from icecream import ic, install

install()
ic.configureOutput(includeContext=True)

import ast
import gc
import importlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import typing
import warnings
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat

import diffusers
import numpy as np
import pyrallis
import torch
import torch.optim.adamw
import transformers
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedType, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoPatchEmbed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm.auto import tqdm

from attn_processor import HunyuanVideoFlashAttnProcessor  # isort: skip
from config import Config  # isort:skip
from mds_dataloaders import build_mds_dataloader  # isort: skip
from optim import get_optimizer, max_gradient  # isort: skip
from ema import EMAModel  # isort: skip

NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
    force=True,
    handlers=[logging.StreamHandler()],
)
warnings.filterwarnings("ignore")  # ignore warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

logger = get_logger(__name__)


def bytes_to_gigabytes(x: int) -> float:
    if x is not None:
        return x / 1024**3


def free_memory() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_memory_statistics(precision: int = 3) -> typing.Dict[str, typing.Any]:
    memory_allocated = None
    memory_reserved = None
    max_memory_allocated = None
    max_memory_reserved = None

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)

    elif torch.backends.mps.is_available():
        memory_allocated = torch.mps.current_allocated_memory()

    else:
        logger.warning("No CUDA, MPS, or ROCm device found. Memory statistics are not available.")

    return {
        "memory_allocated": round(bytes_to_gigabytes(memory_allocated), ndigits=precision),
        "memory_reserved": round(bytes_to_gigabytes(memory_reserved), ndigits=precision),
        "max_memory_allocated": round(bytes_to_gigabytes(max_memory_allocated), ndigits=precision),
        "max_memory_reserved": round(bytes_to_gigabytes(max_memory_reserved), ndigits=precision),
    }


def get_nb_trainable_parameters(mod: torch.nn.Module):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    for _, param in mod.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def get_noisy_model_input_and_timesteps(
    cfg: Config, latents, noise, noise_scheduler, device, weight_dtype, scheduler_sigmas, generator=None
):
    batch_size = latents.size(0)
    if cfg.hparams.flow_match.timestep_sampling == "uniform":
        sigmas = torch.rand((batch_size,), device=device, generator=generator)
    elif cfg.hparams.flow_match.timestep_sampling == "sigmoid":
        # imported from cloneofsimo's minRF trainer: https://github.com/cloneofsimo/minRF
        # also used by: https://github.com/XLabs-AI/x-flux/tree/main
        # and: https://github.com/kohya-ss/sd-scripts/commit/8a0f12dde812994ec3facdcdb7c08b362dbceb0f
        sigmas = torch.sigmoid(
            cfg.hparams.flow_match.sigmoid_scale * torch.randn((batch_size,), device=device, generator=generator)
        )
    elif cfg.hparams.flow_match.timestep_sampling == "logit_normal":
        sigmas = torch.normal(
            cfg.hparams.flow_match.logit_mean,
            cfg.hparams.flow_match.logit_std,
            size=(batch_size,),
            device=device,
            generator=generator,
        )
        sigmas = torch.sigmoid(cfg.hparams.flow_match.sigmoid_scale * sigmas)

    if cfg.hparams.flow_match.discrete_flow_shift is not None and cfg.hparams.flow_match.discrete_flow_shift > 0:
        sigmas = (sigmas * cfg.hparams.flow_match.discrete_flow_shift) / (
            1 + (cfg.hparams.flow_match.discrete_flow_shift - 1) * sigmas
        )

    timesteps, sigmas = (sigmas * 1000.0).long(), sigmas.view(-1, 1, 1, 1, 1)
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
    return noisy_model_input, timesteps


@pyrallis.wrap()
def main(cfg: Config):
    if cfg.experiment.ic_debug:
        ic.enable()
    else:
        ic.disable()

    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    output_dirpath = Path(cfg.experiment.output_dirpath) / cfg.experiment.run_id
    logging_dirpath = output_dirpath / "logs"

    accelerator_project_config = ProjectConfiguration(project_dir=output_dirpath, logging_dir=logging_dirpath)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=cfg.ddp_kwargs.find_unused_parameters,
        gradient_as_bucket_view=cfg.ddp_kwargs.gradient_as_bucket_view,
        static_graph=cfg.ddp_kwargs.static_graph,
    )
    init_kwargs = InitProcessGroupKwargs(backend=cfg.ddp_kwargs.backend, timeout=timedelta(seconds=5400))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.hparams.gradient_accumulation_steps,
        mixed_precision=cfg.hparams.mixed_precision,
        log_with=None,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    print(accelerator.state)

    accelerator.print("\nENVIRONMENT\n")
    accelerator.print(f"  Python .......................... {sys.version}")
    accelerator.print(f"  torch.__version__ ............... {torch.__version__}")
    accelerator.print(f"  torch.version.cuda .............. {torch.version.cuda}")
    accelerator.print(f"  torch.backends.cudnn.version() .. {torch.backends.cudnn.version()}\n")
    accelerator.print("\n")
    accelerator.print(f">> Run ID : {cfg.experiment.run_id!r}")

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.experiment.random_seed is not None:
        set_seed(cfg.experiment.random_seed)

    if accelerator.num_processes > 1:
        logger.info("DDP VARS: ")
        logger.info(f"  WORLD_SIZE: {os.getenv('WORLD_SIZE', 'N/A')}")
        logger.info(f"  LOCAL_WORLD_SIZE: {os.getenv('LOCAL_WORLD_SIZE', 'N/A')}")
        logger.info(f"  RANK: {os.getenv('RANK', 'N/A')}")
        logger.info(f"  MASTER_ADDR: {os.getenv('MASTER_ADDR', 'N/A')}")
        logger.info(f"  MASTER_PORT: {os.getenv('MASTER_PORT', 'N/A')}")

    if accelerator.is_main_process:
        output_dirpath.mkdir(parents=True, exist_ok=True)
    if not accelerator.is_main_process:
        ic.disable()

    if accelerator.is_main_process:
        logger.info(f"Saving config to {output_dirpath / 'config.yaml'}")
        yaml_cfg = pyrallis.dump(cfg)
        with open(output_dirpath / "config.yaml", "w") as f:
            f.write(yaml_cfg)

    logger.info(f"config = \n{pyrallis.dump(cfg)}")

    # ======================================================
    # 2. build model
    # ======================================================

    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=cfg.hparams.flow_match.discrete_flow_shift)

    load_dtype = torch.bfloat16
    logger.info(f"Load transformer model from {cfg.model.pretrained_model_name_or_path!r}")
    t0 = time.time()
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=cfg.model.revision,
        variant=cfg.model.variant,
    )
    logger.info(f"Loaded transformer model from {cfg.model.pretrained_model_name_or_path!r} in {time.time() - t0:.2f}s")

    for blk in transformer.transformer_blocks + transformer.single_transformer_blocks:
        blk.attn.processor = HunyuanVideoFlashAttnProcessor()
    logger.info("transformer patch with flash_attn done ok!")

    with torch.no_grad():
        logger.info("expand transformer x_embedder input channels")
        t0 = time.time()
        initial_input_channels = transformer.config.in_channels
        new_img_in = HunyuanVideoPatchEmbed(
            patch_size=(transformer.config.patch_size_t, transformer.config.patch_size, transformer.config.patch_size),
            in_chans=transformer.config.in_channels * 2,
            embed_dim=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
        )
        new_img_in.proj.weight.zero_()
        new_img_in.proj.weight[:, :initial_input_channels].copy_(transformer.x_embedder.proj.weight)
        if transformer.x_embedder.proj.bias is not None:
            new_img_in.proj.bias.copy_(transformer.x_embedder.proj.bias)
        transformer.x_embedder = new_img_in
        assert torch.all(transformer.x_embedder.proj.weight[:, initial_input_channels:] == 0)
        transformer.register_to_config(in_channels=initial_input_channels * 2, out_channels=initial_input_channels)
        logger.info(f"expanded transformer x_embedder input channels in {time.time() - t0:.2f}s")
    accelerator.wait_for_everyone()

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    logger.info(f"configured weight dtype: {weight_dtype!r}")

    if cfg.hparams.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.requires_grad_(False)

    transformer.to(accelerator.device, dtype=weight_dtype)
    logger.info(f"dit dtype: {next(transformer.parameters()).dtype!r}")

    if cfg.network.lora_layers is not None:
        if cfg.network.lora_layers != "all-linear":
            target_modules = [layer.strip() for layer in cfg.network.lora_layers.split(",")]
            if "x_embedder" not in target_modules:
                target_modules.append("x_embedder.proj")
        elif cfg.network.lora_layers == "all-linear":
            target_modules = set()
            for name, module in transformer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)
            if "x_embedder" not in target_modules:
                target_modules.append("x_embedder.proj")
        target_modules = [t for t in target_modules if "norm" not in t]
    else:
        assert cfg.network.target_modules is not None, "either `lora_layers` or `target_modules` must be specified"
        target_modules = cfg.network.target_modules

    logger.info(f"using LoRA traning mode: ")
    logger.info(f"rank .......................................... {cfg.network.lora_rank!r}")
    logger.info(f"alpha ......................................... {cfg.network.lora_alpha!r}")
    logger.info(f"target_modules ................................ {json.dumps(target_modules, indent=4)}")

    transformer_lora_config = LoraConfig(
        r=cfg.network.lora_rank,
        lora_alpha=cfg.network.lora_alpha,
        lora_dropout=cfg.network.lora_dropout,
        target_modules=target_modules,
        init_lora_weights=cfg.network.init_lora_weights,
    )
    transformer.add_adapter(transformer_lora_config)
    accelerator.wait_for_everyone()

    trainable_params, all_param = get_nb_trainable_parameters(transformer)
    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    if cfg.network.train_norm_layers:
        train_norm_layers = []
        logger.info(f"train norm layers, setting requires_grad to True for layers matching {NORM_LAYER_PREFIXES!r}")
        for name, param in transformer.named_parameters():
            if any(k in name for k in NORM_LAYER_PREFIXES):
                param.requires_grad_(True)
                train_norm_layers.append(name)
        logger.info(f"train norm layers ............................. {json.dumps(train_norm_layers, indent=4)}")

    if cfg.hparams.mixed_precision == "fp16":
        logger.warning("full fp16 training is unstable, casting params to fp32")
        cast_training_params([transformer])

    if cfg.hparams.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    ema_model = None
    if accelerator.is_main_process and cfg.hparams.ema.use_ema:
        logger.info("Using EMA. Creating EMAModel.")
        ema_model_cls, ema_model_config = transformer.__class__, transformer.config
        ema_model = EMAModel(
            cfg.hparams.ema,
            accelerator,
            parameters=transformer_lora_parameters,
            model_cls=ema_model_cls,
            model_config=ema_model_config,
            decay=cfg.hparams.ema.ema_decay,
            foreach=not cfg.hparams.ema.ema_foreach_disable,
        )
        logger.info(f"EMA model creation completed with {ema_model.parameter_count():,} parameters")

    accelerator.wait_for_everyone()

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(
        transformer_lora_parameters,
        optimizer_name=cfg.hparams.optimizer_type,
        learning_rate=cfg.hparams.learning_rate,
        optimizer_args_str=cfg.hparams.optimizer_args,
        use_deepspeed=use_deepspeed_optimizer,
    )

    # ======================================================
    # 3. build dataset and dataloaders
    # ======================================================

    train_dataloader = build_mds_dataloader(
        remote=cfg.data.remote,
        local=cfg.data.local,
        batch_size=cfg.data.batch_size,
        video_key=cfg.data.video_key,
        caption_key=cfg.data.caption_key,
        latents_key=cfg.data.latents_key,
        latents_cond_key=cfg.data.latents_cond_key,
        prompt_embeds_key=cfg.data.prompt_embeds_key,
        prompt_attention_mask_key=cfg.data.prompt_attention_mask_key,
        pooled_prompt_embeds_key=cfg.data.pooled_prompt_embeds_key,
        streaming_kwargs=asdict(cfg.data.streaming_kwargs),
        dataloader_kwargs=asdict(cfg.data.dataloader_kwargs),
        latent_dtype=weight_dtype,
    )

    # =======================================================
    # 4. distributed training preparation with accelerator
    # =======================================================

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        if cfg.hparams.ema.use_ema and ema_model is not None:
            primary_model = unwrap_model(transformer)
            ema_model_path = os.path.join(output_dir, "ema_model.pt")
            logger.info(f"Saving EMA model state to {ema_model_path!r}")
            try:
                ema_model.save_state_dict(ema_model_path)
            except Exception as e:
                logger.error(f"Error saving EMA model: {e!r}")

            # we'll temporarily overwrite the LoRA parameters with the EMA parameters to save it.
            logger.info("Saving EMA model to disk.")
            trainable_parameters = [p for p in primary_model.parameters() if p.requires_grad]
            ema_model.store(trainable_parameters)
            ema_model.copy_to(trainable_parameters)
            transformer_lora_layers = get_peft_model_state_dict(primary_model)
            HunyuanVideoPipeline.save_lora_weights(
                os.path.join(output_dir, "ema"),
                transformer_lora_layers=transformer_lora_layers,
                weight_name=f"{cfg.experiment.name}.sft",
            )
            ema_model.restore(trainable_parameters)

        transformer_lora_layers_to_save = None
        for model in models:
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                model = unwrap_model(model)
                transformer_lora_layers_to_save = get_peft_model_state_dict(model)

                if cfg.network.train_norm_layers:
                    transformer_norm_layers_to_save = {
                        f"transformer.{name}": param
                        for name, param in model.named_parameters()
                        if any(k in name for k in NORM_LAYER_PREFIXES)
                    }
                    transformer_lora_layers_to_save = {
                        **transformer_lora_layers_to_save,
                        **transformer_norm_layers_to_save,
                    }
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            if weights:
                weights.pop()

            HunyuanVideoPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                weight_name=f"{cfg.experiment.name}.sft",
            )

        if hasattr(train_dataloader, "state_dict"):
            torch.save(train_dataloader.state_dict(), os.path.join(output_dir, "train_dataloader_state.pt"))

    def load_model_hook(models, input_dir):
        if hasattr(train_dataloader, "load_state_dict"):
            logger.info(f"Loading train dataloader state from Path: {input_dir!r}")
            train_dataloader.load_state_dict(torch.load(os.path.join(input_dir, "train_dataloader_state.pt")))

        if cfg.hparams.ema.use_ema and ema_model is not None:
            logger.info(f"Loading EMA model from Path: {input_dir!r}")
            try:
                ema_model.load_state_dict(os.path.join(input_dir, "ema_model.pt"))
            except Exception as e:
                logger.error(f"Could not load EMA model: {e!r}")

        transformer_ = None
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_ = HunyuanVideoTransformer3DModel.from_pretrained(
                cfg.model.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.to(accelerator.device, weight_dtype)

            with torch.no_grad():
                initial_input_channels = transformer.config.in_channels
                new_img_in = HunyuanVideoPatchEmbed(
                    patch_size=(
                        transformer.config.patch_size_t,
                        transformer.config.patch_size,
                        transformer.config.patch_size,
                    ),
                    in_chans=transformer.config.in_channels * 2,
                    embed_dim=transformer.config.num_attention_heads
                    * transformer.config.num_attention_heads.attention_head_dim,
                )
                new_img_in.proj.weight.zero_()
                new_img_in.proj.weight[:, :initial_input_channels].copy_(transformer.x_embedder.proj.weight)
                if transformer.x_embedder.proj.bias is not None:
                    new_img_in.proj.bias.copy_(transformer.x_embedder.proj.bias)
                transformer.x_embedder = new_img_in
                transformer.register_to_config(
                    in_channels=initial_input_channels * 2, out_channels=initial_input_channels
                )

            transformer_.add_adapter(transformer_lora_config)

        lora_weight_name = os.path.join(input_dir, f"{cfg.experiment.name}.sft")
        logger.info(f"Loading LoRA weights from Path: {lora_weight_name!r}")
        lora_state_dict = HunyuanVideoPipeline.lora_state_dict(lora_weight_name)
        transformer_lora_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.") and "lora" in k
        }
        # transformer_lora_state_dict = convert_unet_state_dict_to_peft(transformer_lora_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_lora_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if cfg.network.train_norm_layers:
            transformer_norm_layers_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
            }
            for key in list(transformer_norm_layers_state_dict.keys()):
                if key.split(".")[0] == "transformer":
                    transformer_norm_layers_state_dict[
                        key[len(f"transformer.") :]
                    ] = transformer_norm_layers_state_dict.pop(key)

            transformer_state_dict = transformer.state_dict()
            transformer_keys = set(transformer_state_dict.keys())
            state_dict_keys = set(transformer_norm_layers_state_dict.keys())
            extra_keys = list(state_dict_keys - transformer_keys)

            if extra_keys:
                logger.warning(
                    f"Unsupported keys found in state dict when trying to load normalization layers into the transformer. The following keys will be ignored:\n{extra_keys}."
                )

            for key in extra_keys:
                transformer_norm_layers_state_dict.pop(key)

            # We can't load with strict=True because the current state_dict does not contain all the transformer keys
            incompatible_keys = transformer.load_state_dict(transformer_norm_layers_state_dict, strict=False)
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)

            # We shouldn't expect to see the supported norm keys here being present in the unexpected keys.
            if unexpected_keys:
                if any(norm_key in k for k in unexpected_keys for norm_key in NORM_LAYER_PREFIXES):
                    raise ValueError(
                        f"Found {unexpected_keys} as unexpected keys while trying to load norm layers into the transformer."
                    )

        if cfg.hparams.mixed_precision == "fp16":
            cast_training_params([transformer_])

        logger.info(f"Completed loading checkpoint from Path: {input_dir!r}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # The epoch_size attribute of StreamingDataset is the number of samples per epoch of training.
    # The __len__() method returns the epoch_size divided by the number of devices – it is the number of samples seen per device, per epoch.
    # The size() method returns the number of unique samples in the underlying dataset.
    # Due to upsampling/downsampling, size() may not be the same as epoch_size.
    if cfg.hparams.max_train_steps is None:
        len_train_dataloader_after_sharding = len(train_dataloader)
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / cfg.hparams.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            cfg.hparams.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = cfg.hparams.max_train_steps * accelerator.num_processes

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=cfg.hparams.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=num_training_steps_for_scheduler,
            num_warmup_steps=cfg.hparams.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            name=cfg.hparams.lr_scheduler,
            optimizer=optimizer,
            num_training_steps=num_training_steps_for_scheduler,
            num_warmup_steps=cfg.hparams.lr_warmup_steps * accelerator.num_processes,
            num_cycles=cfg.hparams.lr_scheduler_num_cycles,
            power=cfg.hparams.lr_scheduler_power,
        )

    # not need to wrap dataloader because mosaicml-streaming handles it internally
    # the config should be passed via deepspeed json
    if accelerator.state.deepspeed_plugin is not None:
        d = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["reduce_bucket_size"] = d
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = cfg.data.batch_size
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ] = cfg.hparams.gradient_accumulation_steps

    # Prepare everything with our `accelerator`.
    # passing dataloader is required to resolve deepspeed 'auto' params, but we do
    transformer, optimizer, lr_scheduler = accelerator.prepare(transformer, optimizer, lr_scheduler)

    if cfg.hparams.ema.use_ema and ema_model is not None:
        if cfg.hparams.ema.ema_device == "accelerator":
            logger.info("Moving EMA model weights to accelerator...")

        ema_model.to((accelerator.device if cfg.hparams.ema.ema_device == "accelerator" else "cpu"), dtype=weight_dtype)

        if cfg.hparams.ema.ema_device == "cpu" and not cfg.hparams.ema.ema_cpu_only:
            logger.info("Pinning EMA model weights to CPU...")
            try:
                ema_model.pin_memory()
            except Exception as e:
                logger.error(f"Failed to pin EMA model to CPU: {e}")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.hparams.gradient_accumulation_steps)
    if cfg.hparams.max_train_steps is None:
        cfg.hparams.max_train_steps = cfg.hparams.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != cfg.hparams.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )

    # Afterwards we recalculate our number of training epochs
    cfg.hparams.num_train_epochs = math.ceil(cfg.hparams.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = cfg.data.batch_size * accelerator.num_processes * cfg.hparams.gradient_accumulation_steps
    num_trainable_parameters = sum(p.numel() for p in transformer_lora_parameters)

    # fmt: off
    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters ........................................... {num_trainable_parameters}")
    logger.info(f"  Num examples ....................................................... {train_dataloader.dataset.size}")
    logger.info(f"  Num batches each epoch ............................................. {len(train_dataloader)}")
    logger.info(f"  Num epochs ......................................................... {cfg.hparams.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device ................................ {cfg.data.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) ... {total_batch_size}")
    logger.info(f"  Gradient accumulation steps ........................................ {cfg.hparams.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps ........................................... {cfg.hparams.max_train_steps}")
    # fmt: on

    global_step, first_epoch = 0, 0

    if not cfg.checkpointing.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if cfg.checkpointing.resume_from_checkpoint != "latest":
            path = cfg.checkpointing.resume_from_checkpoint
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(cfg.experiment.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            path = os.path.join(cfg.experiment.output_dir, path)

        if path is None:
            accelerator.print(
                f"Checkpoint {cfg.checkpointing.resume_from_checkpoint!r} does not exist. Starting a new training run."
            )
            cfg.checkpointing.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path!r}")
            accelerator.load_state(path)

            global_step = int(path.split("checkpoint-step")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            logger.info(f"Override: global_step={initial_global_step} | first_epoch={first_epoch}")

    memory_statistics = get_memory_statistics()
    logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

    # =======================================================
    # 5. training loop
    # =======================================================
    accelerator.wait_for_everyone()
    progress_bar = tqdm(
        range(0, cfg.hparams.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    generator = torch.Generator(device=accelerator.device)
    scheduler_sigmas = noise_scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
    if cfg.experiment.random_seed is not None:
        generator = generator.manual_seed(cfg.experiment.random_seed)

    for epoch in range(first_epoch, cfg.hparams.num_train_epochs):
        logger.info(f"epoch {epoch+1}/{ cfg.hparams.num_train_epochs}")
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):
                latents, cond_latents = batch[cfg.data.latents_key], batch[cfg.data.latents_cond_key]

                latents = latents.to(accelerator.device, dtype=weight_dtype).contiguous()
                cond_latents = cond_latents.to(accelerator.device, dtype=weight_dtype).contiguous()

                batch_size = latents.size(0)

                prompt_embeds = batch[cfg.data.prompt_embeds_key].to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = batch[cfg.data.pooled_prompt_embeds_key].to(
                    accelerator.device, dtype=weight_dtype
                )
                prompt_attention_mask = batch[cfg.data.prompt_attention_mask_key].to(
                    accelerator.device, dtype=torch.bool
                )

                if random.random() < cfg.hparams.caption_dropout_p:
                    prompt_embeds.fill_(0)
                    pooled_prompt_embeds.fill_(0)
                    prompt_attention_mask.fill_(False)

                noise = torch.randn(latents.shape, device=accelerator.device, dtype=weight_dtype, generator=generator)

                noisy_model_input, timesteps = get_noisy_model_input_and_timesteps(
                    cfg=cfg,
                    latents=latents,
                    noise=noise,
                    noise_scheduler=noise_scheduler,
                    device=accelerator.device,
                    weight_dtype=weight_dtype,
                    generator=generator,
                    scheduler_sigmas=scheduler_sigmas,
                )
                noisy_model_input = noisy_model_input.to(weight_dtype)

                weighting = compute_loss_weighting_for_sd3(
                    cfg.hparams.flow_match.weighting_scheme, sigmas=scheduler_sigmas
                )
                while len(weighting.shape) < latents.ndim:
                    weighting = weighting.unsqueeze(-1)
                guidance_vec = (
                    torch.full((batch_size,), float(cfg.hparams.guidance_scale), device=accelerator.device) * 1000.0
                )

                ic(noisy_model_input.shape, cond_latents.shape)
                ic(
                    step,
                    guidance_vec,
                    weighting,
                    timesteps,
                    prompt_embeds.shape,
                    prompt_attention_mask.shape,
                    pooled_prompt_embeds.shape,
                )

                denoised_latents = transformer(
                    hidden_states=torch.cat([noisy_model_input, cond_latents], dim=1),
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    guidance=guidance_vec,
                    return_dict=False,
                )[0]

                target = noise - latents
                loss = torch.nn.functional.mse_loss(denoised_latents.to(weight_dtype), target, reduction="none")

                if weighting is not None:
                    loss = loss * weighting

                loss = loss.mean()

                assert torch.isnan(loss) == False, "NaN loss detected"

                accelerator.backward(loss)

                if cfg.hparams.gradient_precision == "fp32":
                    for param in transformer_lora_parameters:
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(torch.float32)

                grad_norm = max_gradient(transformer_lora_parameters)
                if accelerator.sync_gradients:
                    if accelerator.distributed_type == DistributedType.DEEPSPEED:
                        grad_norm = transformer.get_global_grad_norm()

                    elif cfg.hparams.max_grad_norm > 0:
                        if cfg.hparams.grad_clip_method == "norm":
                            grad_norm = accelerator.clip_grad_norm_(
                                transformer_lora_parameters, cfg.hparams.max_grad_norm
                            )
                        elif cfg.hparams.grad_clip_method == "value":
                            grad_norm = accelerator.clip_grad_value_(
                                transformer_lora_parameters, cfg.hparams.max_grad_norm
                            )

                if torch.is_tensor(grad_norm):
                    grad_norm = grad_norm.item()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if cfg.hparams.ema.use_ema and ema_model is not None:
                    ema_model.step(parameters=transformer_lora_parameters, global_step=global_step)

                if accelerator.is_main_process:
                    if global_step % cfg.checkpointing.save_every_n_steps == 0:
                        save_path = os.path.join(output_dirpath, f"checkpoint-step{global_step:08d}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path!r}")

                        if cfg.checkpointing.save_last_n_steps is not None:
                            remove_step_no = global_step - cfg.checkpointing.save_last_n_steps - 1
                            remove_step_no = remove_step_no - (remove_step_no % cfg.checkpointing.save_every_n_steps)
                            if remove_step_no < 0:
                                remove_step_no = None
                            if remove_step_no is not None:
                                remove_ckpt_name = os.path.join(output_dirpath, f"checkpoint-step{remove_step_no:08d}")
                                if os.path.exists(remove_ckpt_name):
                                    logger.info(f"removing old checkpoint: {remove_ckpt_name!r}")
                                    shutil.rmtree(remove_ckpt_name)

            logs = {}
            logs["loss"] = accelerator.reduce(loss.detach().clone(), reduction="mean").item()
            logs["grad_norm"] = grad_norm
            logs["lr"] = lr_scheduler.get_last_lr()[0]
            if ema_model is not None:
                logs["ema_decay"] = ema_model.get_decay()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= cfg.hparams.max_train_steps:
                logger.info(f"max training steps={cfg.hparams.max_train_steps!r} reached.")
                break

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if cfg.hparams.ema.use_ema and ema_model is not None:
            ema_model.copy_to(transformer_lora_parameters)

        transformer = unwrap_model(transformer)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        HunyuanVideoPipeline.save_lora_weights(
            output_dirpath,
            transformer_lora_layers=transformer_lora_layers,
            safe_serialization=True,
            weight_name=f"{cfg.experiment.name}.sft",
        )
    accelerator.wait_for_everyone()

    memory_statistics = get_memory_statistics()
    logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
