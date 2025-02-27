import logging
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from streaming import Stream, StreamingDataLoader, StreamingDataset
from torch.utils.data import DataLoader


def make_streams(remote, local=None, proportion=None, repeat=None, choose=None):
    """Helper function to create a list of Stream objects from a set of remotes and stream weights.

    Args:
        remote (Union[str, Sequence[str]]): The remote path or paths to stream from.
        local (Union[str, Sequence[str]], optional): The local path or paths to cache the data. If not provided, the
            default local path is used. Default: ``None``.
        proportion (list, optional): Specifies how to sample this Stream relative to other Streams. Default: ``None``.
        repeat (list, optional): Specifies the degree to which a Stream is upsampled or downsampled. Default: ``None``.
        choose (list, optional): Specifies the number of samples to choose from a Stream. Default: ``None``.

    Returns:
        List[Stream]: A list of Stream objects.
    """
    remote, local = _make_remote_and_local_sequences(remote, local)
    proportion, repeat, choose = _make_weighting_sequences(remote, proportion, repeat, choose)

    streams = []
    for i, (r, l) in enumerate(zip(remote, local)):
        streams.append(Stream(remote=r, local=l, proportion=proportion[i], repeat=repeat[i], choose=choose[i]))
    return streams


def _make_remote_and_local_sequences(remote, local=None):
    if isinstance(remote, str):
        remote = [remote]
    if isinstance(local, str):
        local = [local]
    if not local:
        local = [_make_default_local_path(r) for r in remote]

    if isinstance(remote, Sequence) and isinstance(local, Sequence):
        if len(remote) != len(local):
            ValueError(
                f"remote and local Sequences must be the same length, got lengths {len(remote)} and {len(local)}"
            )
    else:
        ValueError(f"remote and local must be both Strings or Sequences, got types {type(remote)} and {type(local)}.")
    return remote, local


def _make_default_local_path(remote_path):
    return str(Path(*["/tmp"] + list(Path(remote_path).parts[1:])))


def _make_weighting_sequences(remote, proportion=None, repeat=None, choose=None):
    weights = {"proportion": proportion, "repeat": repeat, "choose": choose}
    for name, weight in weights.items():
        if weight is not None and len(remote) != len(weight):
            ValueError(f"{name} must be the same length as remote, got lengths {len(remote)} and {len(weight)}")
    proportion = weights["proportion"] if weights["proportion"] is not None else [None] * len(remote)
    repeat = weights["repeat"] if weights["repeat"] is not None else [None] * len(remote)
    choose = weights["choose"] if weights["choose"] is not None else [None] * len(remote)
    return proportion, repeat, choose


class StreamingVideoCaptionLatentsDataset(StreamingDataset):
    def __init__(
        self,
        streams: Sequence[Stream],
        video_key: str = "video",
        caption_key: str = "caption",
        latents_key: str = "latents",
        latents_cond_key: str = "latents_cond",
        prompt_embeds_key: str = "prompt_embeds",
        prompt_attention_mask_key: str = "prompt_attention_mask",
        pooled_prompt_embeds_key: str = "pooled_prompt_embeds",
        latent_dtype: torch.dtype = torch.bfloat16,
        batch_size: int = None,
        **streaming_kwargs,
    ):
        streaming_kwargs.setdefault("shuffle_block_size", 1 << 18)
        streaming_kwargs.setdefault("shuffle_algo", "py1s")
        super().__init__(streams=streams, batch_size=batch_size, **streaming_kwargs)

        self.video_key = video_key
        self.caption_key = caption_key
        self.latents_key = latents_key
        self.prompt_embeds_key = prompt_embeds_key
        self.latents_cond_key = latents_cond_key
        self.prompt_attention_mask_key = prompt_attention_mask_key
        self.pooled_prompt_embeds_key = pooled_prompt_embeds_key
        self.latent_dtype = latent_dtype

    def __getitem__(self, index):
        sample = super().__getitem__(index)

        out = {}

        latents = torch.from_numpy(sample[self.latents_key].copy()).to(dtype=self.latent_dtype)
        latents_cond = torch.from_numpy(sample[self.latents_cond_key].copy()).to(dtype=self.latent_dtype)

        prompt_embeds = torch.from_numpy(sample[self.prompt_embeds_key].copy()).to(dtype=self.latent_dtype)
        pooled_prompt_embeds = torch.from_numpy(sample[self.pooled_prompt_embeds_key].copy()).to(
            dtype=self.latent_dtype
        )
        prompt_attention_mask = torch.from_numpy(sample[self.prompt_attention_mask_key].copy()).to(dtype=torch.bool)

        out[self.latents_key] = latents
        out[self.latents_cond_key] = latents_cond
        out[self.prompt_embeds_key] = prompt_embeds
        out[self.pooled_prompt_embeds_key] = pooled_prompt_embeds
        out[self.prompt_attention_mask_key] = prompt_attention_mask

        return out


def build_mds_dataloader(
    remote: Union[str, List],
    local: Union[str, List],
    batch_size: int,
    video_key: str = "video",
    caption_key: str = "caption",
    latents_key: str = "latents",
    latents_cond_key: str = "latents_cond",
    prompt_embeds_key: str = "prompt_embeds",
    prompt_attention_mask_key: str = "prompt_attention_mask",
    pooled_prompt_embeds_key: str = "pooled_prompt_embeds",
    latent_dtype: torch.dtype = torch.bfloat16,
    proportion: Optional[list] = None,
    repeat: Optional[list] = None,
    choose: Optional[list] = None,
    streaming_kwargs: Optional[Dict] = None,
    dataloader_kwargs: Optional[Dict] = None,
):
    if streaming_kwargs is None:
        streaming_kwargs = {}
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    # streams = make_streams(remote, local=local, proportion=proportion, repeat=repeat, choose=choose)
    if isinstance(local, str):
        local = [local]
    streams = [Stream(local=l) for l in local]

    dataset = StreamingVideoCaptionLatentsDataset(
        streams=streams,
        video_key=video_key,
        caption_key=caption_key,
        latents_key=latents_key,
        latents_cond_key=latents_cond_key,
        prompt_embeds_key=prompt_embeds_key,
        prompt_attention_mask_key=prompt_attention_mask_key,
        pooled_prompt_embeds_key=pooled_prompt_embeds_key,
        latent_dtype=latent_dtype,
        batch_size=batch_size,
        **streaming_kwargs,
    )

    dataloader = StreamingDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=None,
        **dataloader_kwargs,
    )

    return dataloader
