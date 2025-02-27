#!/usr/bin/env python3
import argparse
import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel
from diffusers.models.attention import Attention
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoPatchEmbed
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import DEFAULT_PROMPT_TEMPLATE, retrieve_timesteps
from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
from PIL import Image

# Try to import flash attention
try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward, flash_attn_varlen_func
except ImportError:
    flash_attn, _flash_attn_forward, flash_attn_varlen_func = None, None, None

try:
    from sageattention import sageattn, sageattn_varlen
except ImportError:
    sageattn, sageattn_varlen = None, None


def get_cu_seqlens(attention_mask):
    """Calculate cu_seqlens_q, cu_seqlens_kv using attention_mask"""
    batch_size = attention_mask.shape[0]
    text_len = attention_mask.sum(dim=-1, dtype=torch.int)
    max_len = attention_mask.shape[-1]

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i]
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


class HunyuanVideoFlashAttnProcessor:
    def __init__(self, use_flash_attn=True, use_sageattn=False):
        self.use_flash_attn = use_flash_attn
        self.use_sageattn = use_sageattn
        if self.use_flash_attn:
            assert flash_attn is not None, "Flash attention not available"
        if self.use_sageattn:
            assert sageattn is not None, "Sage attention not available"

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        batch_size = hidden_states.shape[0]
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = 0

        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

            txt_seq_len = encoder_hidden_states.shape[1]

        max_seqlen_q = max_seqlen_kv = img_seq_len + txt_seq_len
        cu_seqlens_q = cu_seqlens_kv = get_cu_seqlens(attention_mask)

        query = query.transpose(1, 2).reshape(-1, query.shape[1], query.shape[3])
        key = key.transpose(1, 2).reshape(-1, key.shape[1], key.shape[3])
        value = value.transpose(1, 2).reshape(-1, value.shape[1], value.shape[3])

        if self.use_flash_attn:
            hidden_states = flash_attn_varlen_func(
                query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv
            )
        elif self.use_sageattn:
            hidden_states = sageattn_varlen(query, key, value, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        else:
            raise NotImplementedError("Please set use_flash_attn=True or use_sageattn=True")

        hidden_states = hidden_states.reshape(batch_size, max_seqlen_q, -1)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


def save_video(video, output_path="output.mp4"):
    """Save frames as a video file"""
    width, height = video[0].size

    container = av.open(output_path, mode="w")

    # Create video stream
    codec = "libx264"
    pixel_format = "yuv420p"
    stream = container.add_stream(codec, rate=24)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pixel_format
    stream.bit_rate = 4000000  # 4Mbit/s

    for frame_array in video:
        frame = av.VideoFrame.from_image(frame_array)
        packets = stream.encode(frame)
        for packet in packets:
            container.mux(packet)

    # Flush remaining packets
    for packet in stream.encode():
        container.mux(packet)

    container.close()


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """Resize the image to the bucket resolution."""
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # Crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]

    return image


@torch.inference_mode()
def call_pipe(
    pipe,
    prompt: Union[str, List[str]] = None,
    prompt_2: Union[str, List[str]] = None,
    height: int = 720,
    width: int = 1280,
    num_frames: int = 129,
    num_inference_steps: int = 50,
    sigmas: List[float] = None,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    prompt_attention_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
    max_sequence_length: int = 256,
    image_latents: Optional[torch.Tensor] = None,
):
    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 1. Check inputs
    pipe.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds,
        callback_on_step_end_tensor_inputs,
        prompt_template,
    )

    pipe._guidance_scale = guidance_scale
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False

    device = pipe._execution_device

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 3. Encode input prompt
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_template=prompt_template,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        device=device,
        max_sequence_length=max_sequence_length,
    )

    transformer_dtype = pipe.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
    if pooled_prompt_embeds is not None:
        pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
    )

    # 5. Prepare latent variables
    num_channels_latents = pipe.transformer.config.in_channels
    num_latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    latents = pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_latent_frames,
        torch.float32,
        device,
        generator,
        latents,
    )

    # 6. Prepare guidance condition
    guidance = torch.tensor([guidance_scale] * latents.shape[0], dtype=transformer_dtype, device=device) * 1000.0

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    pipe._num_timesteps = len(timesteps)

    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipe.interrupt:
                continue

            pipe._current_timestep = t
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = pipe.transformer(
                hidden_states=torch.cat([latent_model_input, image_latents], dim=1),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                pooled_projections=pooled_prompt_embeds,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

            # Compute the previous noisy sample x_t -> x_t-1
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(pipe, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # Update progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    pipe._current_timestep = None

    if output_type != "latent":
        latents = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        video = pipe.vae.decode(latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    # Offload all models
    pipe.maybe_free_model_hooks()

    return (video,) if not return_dict else HunyuanVideoPipelineOutput(frames=video)


def setup_pipeline(model_path, lora_path=None):
    """Set up the HunyuanVideo pipeline with optional LoRA weights"""
    pipe = HunyuanVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Apply flash attention to all transformer blocks
    for block in pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks:
        block.attn.processor = HunyuanVideoFlashAttnProcessor(use_flash_attn=True, use_sageattn=False)

    # Modify input channels for conditioning
    with torch.no_grad():
        initial_input_channels = pipe.transformer.config.in_channels
        new_img_in = HunyuanVideoPatchEmbed(
            patch_size=(
                pipe.transformer.config.patch_size_t,
                pipe.transformer.config.patch_size,
                pipe.transformer.config.patch_size,
            ),
            in_chans=pipe.transformer.config.in_channels * 2,
            embed_dim=pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim,
        )
        new_img_in = new_img_in.to(pipe.device, dtype=pipe.dtype)
        new_img_in.proj.weight.zero_()
        new_img_in.proj.weight[:, :initial_input_channels].copy_(pipe.transformer.x_embedder.proj.weight)

        if pipe.transformer.x_embedder.proj.bias is not None:
            new_img_in.proj.bias.copy_(pipe.transformer.x_embedder.proj.bias)

        pipe.transformer.x_embedder = new_img_in
        pipe.transformer.x_embedder.requires_grad_(False)

    # Load LoRA weights if provided
    if lora_path and os.path.exists(lora_path):
        lora_state_dict = pipe.lora_state_dict(lora_path)

        # Load transformer LoRA weights
        transformer_lora_state_dict = {
            k.replace("transformer.", ""): v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.") and "lora" in k
        }
        pipe.load_lora_into_transformer(
            transformer_lora_state_dict, transformer=pipe.transformer, adapter_name="i2v", _pipeline=pipe
        )
        pipe.set_adapters(["i2v"], adapter_weights=[1.0])
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0, adapter_names=["i2v"])
        pipe.unload_lora_weights()

        # Load norm layers if present
        NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]
        transformer_norm_layers_state_dict = {
            k.replace("transformer.", ""): v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
        }

        if transformer_norm_layers_state_dict:
            print("[INFO] Loading normalization layers from state dict...")
            transformer_state_dict = pipe.transformer.state_dict()
            transformer_keys = set(transformer_state_dict.keys())
            state_dict_keys = set(transformer_norm_layers_state_dict.keys())
            extra_keys = list(state_dict_keys - transformer_keys)

            if extra_keys:
                print(f"[WARNING] Ignoring unsupported keys: {extra_keys}")
                for key in extra_keys:
                    transformer_norm_layers_state_dict.pop(key)

            pipe.transformer.load_state_dict(transformer_norm_layers_state_dict, strict=False)
        else:
            print("[INFO] No normalization layers found in state dict")

    return pipe


def prepare_conditioning(pipe, frame1_path, frame2_path, n_frames, height, width):
    """Prepare conditioning frames for the model"""
    video_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    # Load and resize conditioning frames
    cond_frame1 = Image.open(frame1_path).convert("RGB")
    cond_frame2 = Image.open(frame2_path).convert("RGB")

    cond_frame1 = resize_image_to_bucket(cond_frame1, bucket_reso=(width, height))
    cond_frame2 = resize_image_to_bucket(cond_frame2, bucket_reso=(width, height))

    # Create conditioning video tensor
    cond_video = np.zeros(shape=(n_frames, height, width, 3))
    cond_video[0], cond_video[-1] = np.array(cond_frame1), np.array(cond_frame2)

    cond_video = torch.from_numpy(cond_video.copy()).permute(0, 3, 1, 2)
    cond_video = torch.stack([video_transforms(x) for x in cond_video], dim=0).unsqueeze(0)

    # Encode to latent space
    with torch.inference_mode():
        image_or_video = cond_video.to(device="cuda", dtype=pipe.dtype)
        image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
        cond_latents = pipe.vae.encode(image_or_video).latent_dist.sample()
        cond_latents = cond_latents * pipe.vae.config.scaling_factor
        cond_latents = cond_latents.to(dtype=pipe.dtype)

    return cond_latents


def main():
    parser = argparse.ArgumentParser(description="Run HunyuanVideo inference with control frames")
    parser.add_argument(
        "--model", type=str, default="hunyuanvideo-community/HunyuanVideo", help="Path to HunyuanVideo model"
    )
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA weights for image-to-video control")
    parser.add_argument("--frame1", type=str, required=True, help="Path to first control frame")
    parser.add_argument("--frame2", type=str, required=True, help="Path to second control frame")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--height", type=int, default=720, help="Output video height")
    parser.add_argument("--width", type=int, default=1280, help="Output video width")
    parser.add_argument("--frames", type=int, default=77, help="Number of frames to generate")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=6.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--output", type=str, default=None, help="Output video path (default: auto-generated)")

    args = parser.parse_args()

    # Validate inputs
    for path in [args.frame1, args.frame2]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input frame not found: {path}")

    if args.lora and not os.path.exists(args.lora):
        raise FileNotFoundError(f"LoRA weights not found: {args.lora}")

    # Set random seed
    seed = args.seed if args.seed is not None else int(time.time()) % 10000
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Setup pipeline
    print(f"Loading model from {args.model}")
    pipe = setup_pipeline(args.model, args.lora)

    # Prepare conditioning
    print("Preparing conditioning frames...")
    cond_latents = prepare_conditioning(pipe, args.frame1, args.frame2, args.frames, args.height, args.width)

    # Generate video
    print(f"Generating video with prompt: '{args.prompt}'")
    video = call_pipe(
        pipe,
        prompt=args.prompt,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        image_latents=cond_latents,
        width=args.width,
        height=args.height,
        guidance_scale=args.guidance,
        generator=generator,
    ).frames[0]

    # Save output
    if args.output:
        output_path = args.output
    else:
        time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        output_path = f"hv-CL-{args.height}x{args.width}x{args.frames}-{time_flag}.mp4"

    print(f"Saving video to {output_path}")
    save_video(video, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
