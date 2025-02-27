import logging
import os
from argparse import ArgumentParser
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import accelerate
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKLHunyuanVideo,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
)
from PIL import Image
from streaming import MDSWriter, Stream, StreamingDataLoader, StreamingDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
logger = get_logger("cache_latents")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
    force=True,
    handlers=[logging.StreamHandler()],
)

_COMMON_BEGINNING_PHRASES = (
    "This video",
    "The video",
    "This clip",
    "The clip",
    "The animation",
    "This image",
    "The image",
    "This picture",
    "The picture",
)
_COMMON_CONTINUATION_WORDS = ("shows", "depicts", "features", "captures", "highlights", "introduces", "presents")

COMMON_LLM_START_PHRASES = (
    "In the video,",
    "In this video,",
    "In this video clip,",
    "In the clip,",
    "Caption:",
    *(
        f"{beginning} {continuation}"
        for beginning in _COMMON_BEGINNING_PHRASES
        for continuation in _COMMON_CONTINUATION_WORDS
    ),
)


def load_condition_models(
    model_id: str = "hunyuanvideo-community/HunyuanVideo",
    text_encoder_dtype: torch.dtype = torch.float16,
    text_encoder_2_dtype: torch.dtype = torch.float16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision, cache_dir=cache_dir)
    text_encoder = LlamaModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=text_encoder_dtype, revision=revision, cache_dir=cache_dir
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer_2", revision=revision, cache_dir=cache_dir
    )
    text_encoder_2 = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=text_encoder_2_dtype, revision=revision, cache_dir=cache_dir
    )
    if device is not None:
        text_encoder.to(device)
        text_encoder_2.to(device)
    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "tokenizer_2": tokenizer_2,
        "text_encoder_2": text_encoder_2,
    }


def load_latent_models(
    model_id: str = "hunyuanvideo-community/HunyuanVideo",
    vae_dtype: torch.dtype = torch.float16,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> Dict[str, nn.Module]:
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        model_id, subfolder="vae", torch_dtype=vae_dtype, revision=revision, cache_dir=cache_dir
    )
    if device is not None:
        vae.to(device)
    vae.enable_slicing()
    vae.enable_tiling()
    return {
        "vae": vae,
    }


def load_diffusion_models(
    model_id: str = "hunyuanvideo-community/HunyuanVideo",
    transformer_dtype: torch.dtype = torch.bfloat16,
    shift: float = 1.0,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Union[nn.Module, FlowMatchEulerDiscreteScheduler]]:
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=transformer_dtype, revision=revision, cache_dir=cache_dir
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    return {"transformer": transformer, "scheduler": scheduler}


def prepare_conditions(
    tokenizer: LlamaTokenizer,
    text_encoder: LlamaModel,
    tokenizer_2: CLIPTokenizer,
    text_encoder_2: CLIPTextModel,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 256,
    prompt_template: Dict[str, Any] = {
        "template": (
            "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
            "1. The main content and theme of the video."
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
            "4. background environment, light, style and atmosphere."
            "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        ),
        "crop_start": 95,
    },
    **kwargs,
) -> torch.Tensor:
    if isinstance(prompt, str):
        prompt = [prompt]

    conditions = {}
    conditions.update(
        _get_llama_prompt_embeds(tokenizer, text_encoder, prompt, prompt_template, device, dtype, max_sequence_length)
    )
    conditions.update(_get_clip_prompt_embeds(tokenizer_2, text_encoder_2, prompt, device, dtype))

    return conditions


def prepare_latents(
    vae: AutoencoderKLHunyuanVideo,
    image_or_video: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    **kwargs,
) -> torch.Tensor:
    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=dtype)
    image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]

    latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
    latents = latents * vae.config.scaling_factor
    latents = latents.to(dtype=dtype)
    return {"latents": latents}


def _get_llama_prompt_embeds(
    tokenizer: LlamaTokenizer,
    text_encoder: LlamaModel,
    prompt: List[str],
    prompt_template: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = 256,
    num_hidden_layers_to_skip: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(prompt)
    prompt = [prompt_template["template"].format(p) for p in prompt]

    crop_start = prompt_template.get("crop_start", None)
    if crop_start is None:
        prompt_template_input = tokenizer(
            prompt_template["template"],
            padding="max_length",
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=False,
        )
        crop_start = prompt_template_input["input_ids"].shape[-1]
        # Remove <|eot_id|> token and placeholder {}
        crop_start -= 2

    max_sequence_length += crop_start
    text_inputs = tokenizer(
        prompt,
        max_length=max_sequence_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )
    text_input_ids = text_inputs.input_ids.to(device=device)
    prompt_attention_mask = text_inputs.attention_mask.to(device=device)

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_attention_mask,
        output_hidden_states=True,
    ).hidden_states[-(num_hidden_layers_to_skip + 1)]
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    if crop_start is not None and crop_start > 0:
        prompt_embeds = prompt_embeds[:, crop_start:]
        prompt_attention_mask = prompt_attention_mask[:, crop_start:]

    prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

    return {"prompt_embeds": prompt_embeds, "prompt_attention_mask": prompt_attention_mask}


def _get_clip_prompt_embeds(
    tokenizer_2: CLIPTokenizer,
    text_encoder_2: CLIPTextModel,
    prompt: Union[str, List[str]],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = 77,
) -> torch.Tensor:
    text_inputs = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    prompt_embeds = text_encoder_2(text_inputs.input_ids.to(device), output_hidden_states=False).pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    return {"pooled_prompt_embeds": prompt_embeds}


def main(args):
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=36))])

    video_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    mds_streams = []
    if args.recursive:
        for fp in args.mds_data_path:
            for dir in os.listdir(fp):
                dir_path = os.path.join(fp, dir)
                mds_streams.append((Stream(local=dir_path), dir_path))
    else:
        for fp in args.mds_data_path:
            mds_streams.append((Stream(local=fp), fp))

    accelerator.print(f"## mds_streams: {len(mds_streams)}")

    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    logger.info(f"set dtype to {dtype!r}")

    models = load_condition_models(
        args.pretrained_model_name_or_path,
        text_encoder_dtype=dtype,
        text_encoder_2_dtype=torch.bfloat16,
        device=accelerator.device,
    )
    models.update(load_latent_models(args.pretrained_model_name_or_path, vae_dtype=dtype, device=accelerator.device))

    models["vae"].eval().requires_grad_(False)
    models["text_encoder"].eval().requires_grad_(False)
    models["text_encoder_2"].eval().requires_grad_(False)

    accelerator.wait_for_everyone()

    for varname in ["RANK", "LOCAL_WORLD_SIZE", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]:
        assert os.environ.get(varname) is not None, f"{varname} is not set"
        logger.info(f"{varname!r}: {os.environ.get(varname)}")

    for stream, data_path in mds_streams:
        logger.info(f"## Processing {data_path!r}")

        dataset = StreamingDataset(
            streams=[stream],
            batch_size=1,
            num_canonical_nodes=(int(os.environ["WORLD_SIZE"]) // 8),
        )

        save_path = Path(args.output_dir) / f"{Path(data_path).name}_{accelerator.local_process_index:02d}"
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path.as_posix()

        logger.info(f"Saving to => {save_path!r}")

        columns = {
            # "idx": "int32",
            # "item_key": "str",
            # "item": "str",
            # "frame_count": "int32",
            # "bucket_width": "int32",
            # "bucket_height": "int32",
            # "original_width": "int32",
            # "original_height": "int32",
            # "caption_str": "str",
            # "video_arr": "ndarray",
            "prompt_embeds": "ndarray",
            "prompt_attention_mask": "ndarray",
            "pooled_prompt_embeds": "ndarray",
            "latents": "ndarray",
            "latents_cond": "ndarray",
            # "latents_cond_2": "ndarray",
            # "latents_cond_only_first": "ndarray",
            # "latents_cond_only_last": "ndarray",
        }

        os.umask(0o000)
        writer = MDSWriter(
            out=save_path,
            columns=columns,
            compression=args.mds_shard_compression,
            size_limit=256 * (2**20),
            max_workers=64,
        )

        def collate_fn(batch):
            idx = [x["idx"] for x in batch]
            item_key = [x["item_key"] for x in batch]
            item = [x["item"] for x in batch]
            video = [x["video_arr"] for x in batch]
            caption = [x["caption_str"] for x in batch]

            for i in range(len(caption)):
                caption[i] = caption[i].strip()
                for phrase in COMMON_LLM_START_PHRASES:
                    if caption[i].startswith(phrase):
                        caption[i] = caption[i].removeprefix(phrase).strip()

            return {"video_arr": video, "caption_str": caption, "idx": idx, "item_key": item_key, "item": item}

        dl = StreamingDataLoader(
            dataset,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=8,
            prefetch_factor=2,
            pin_memory=True,
        )

        for batch in tqdm(dl, dynamic_ncols=True, desc="Precomputing latents", disable=not accelerator.is_main_process):
            # print(accelerator.process_index, batch["idx"], batch["item_key"], batch["item"])

            assert len(batch["video_arr"]) == 1

            video, caption = batch["video_arr"][0], batch["caption_str"][0]
            video = torch.from_numpy(video.copy()).permute(0, 3, 1, 2)  # n_frames, c, h, w

            control_condition = torch.zeros_like(video)  # create an empty video
            control_condition[0] = video[0]  # keep the first frame
            control_condition[-1] = video[-1]  # keep the last frame

            # control_condition_2 = torch.zeros_like(video)  # create an empty video
            # control_condition_2[0] = video[0]  # keep the first frame
            # control_condition_2[-1] = video[-1]  # keep the last frame
            # control_condition_2[video.shape[0] // 2] = video[video.shape[0] // 2]  # keep the middle frame

            # control_condition_only_first = torch.zeros_like(video)  # create an empty video
            # control_condition_only_first[0] = video[0]  # keep the first frame

            # control_condition_only_last = torch.zeros_like(video)  # create an empty video
            # control_condition_only_last[-1] = video[-1]  # keep the last frame

            video = torch.stack([video_transforms(x) for x in video], dim=0).unsqueeze(0)
            control_condition = torch.stack([video_transforms(x) for x in control_condition], dim=0).unsqueeze(0)
            # control_condition_2 = torch.stack([video_transforms(x) for x in control_condition_2], dim=0).unsqueeze(0)
            # control_condition_only_first = torch.stack([video_transforms(x) for x in control_condition_only_first], dim=0).unsqueeze(0)
            # control_condition_only_last = torch.stack([video_transforms(x) for x in control_condition_only_last], dim=0).unsqueeze(0)

            with torch.inference_mode():  # @TODO: add batch support ?
                latents = prepare_latents(models["vae"], video, device=accelerator.device, dtype=dtype)["latents"]
                latents_cond = prepare_latents(
                    models["vae"], control_condition, device=accelerator.device, dtype=dtype
                )["latents"]
                # latents_cond_2 = prepare_latents(models["vae"], control_condition_2, device=accelerator.device, dtype=dtype)["latents"]
                # latents_cond_only_first = prepare_latents(models["vae"], control_condition_only_first, device=accelerator.device, dtype=dtype)["latents"]
                # latents_cond_only_last = prepare_latents(models["vae"], control_condition_only_last, device=accelerator.device, dtype=dtype)["latents"]

                conditions = prepare_conditions(
                    tokenizer=models["tokenizer"],
                    text_encoder=models["text_encoder"],
                    tokenizer_2=models["tokenizer_2"],
                    text_encoder_2=models["text_encoder_2"],
                    prompt=caption,
                    device=accelerator.device,
                    dtype=dtype,
                )
                prompt_embeds, prompt_attention_mask, pooled_prompt_embeds = (
                    conditions["prompt_embeds"],
                    conditions["prompt_attention_mask"],
                    conditions["pooled_prompt_embeds"],
                )

                # out_batch = {**batch} # only takes extra space
                out_batch = {}
                out_batch["latents"] = latents[0].float().cpu().numpy()
                out_batch["prompt_embeds"] = prompt_embeds[0].float().cpu().numpy()
                out_batch["prompt_attention_mask"] = prompt_attention_mask[0].float().cpu().numpy()
                out_batch["pooled_prompt_embeds"] = pooled_prompt_embeds[0].float().cpu().numpy()
                out_batch["latents_cond"] = latents_cond[0].float().cpu().numpy()
                # out_batch["latents_cond_2"] = latents_cond_2[0].float().cpu().numpy()
                # out_batch["latents_cond_only_first"] = latents_cond_only_first[0].float().cpu().numpy()
                # out_batch["latents_cond_only_last"] = latents_cond_only_last[0].float().cpu().numpy()

                assert (
                    out_batch.keys() == columns.keys()
                ), f"{out_batch.keys()} != {columns.keys()}, missing {set(out_batch.keys()) - set(columns.keys())}"

                os.umask(0o000)
                writer.write(out_batch)

        writer.finish()
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mds_data_path", required=True, type=str, nargs="+")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="hunyuanvideo-community/HunyuanVideo")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--mds_shard_compression", type=str, default=None)
    parser.add_argument("--recursive", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
