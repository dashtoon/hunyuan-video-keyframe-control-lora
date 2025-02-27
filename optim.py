import inspect
from typing import Any, Dict

import torch
from accelerate.logging import get_logger

try:
    from optimi import AdamW as OptimiAdamW
    from optimi import StableAdamW as OptimiStableAdamW
except ImportError:
    OptimiAdamW, OptimiStableAdamW = None, None

try:
    from bitsandbytes.optim import AdamW8bit, Lion8bit
except ImportError:
    AdamW8bit, Lion8bit = None, None

try:
    from came_pytorch import CAME
except ImportError:
    CAME = None

import ast

logger = get_logger(__name__)


OPTIMIZER_FUNC_TO_NAME = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "optimi-adamw": OptimiAdamW,
    "optimi-stableadamw": OptimiStableAdamW,
    "bnb-adamw8bit": AdamW8bit,
    "bnb-lion8bit": Lion8bit,
    "came": CAME,
}


def get_optimizer(
    params_to_optimize,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    optimizer_args_str: str | None = None,
    use_deepspeed: bool = False,
    # use_cpu_offload_optimizer: bool = False,
    # offload_gradients: bool = False,
) -> torch.optim.Optimizer:
    optimizer_kwargs = {}

    if optimizer_args_str is not None and len(optimizer_args_str) > 0:
        for arg in optimizer_args_str:
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value

    optimizer_name = optimizer_name.lower()

    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(params_to_optimize, lr=learning_rate, **optimizer_kwargs)

    assert optimizer_name in OPTIMIZER_FUNC_TO_NAME, f"Unknown optimizer: {optimizer_name!r}"

    optimizer_class = OPTIMIZER_FUNC_TO_NAME[optimizer_name]
    assert optimizer_class is not None

    optimizer = optimizer_class(params_to_optimize, lr=learning_rate, **optimizer_kwargs)

    logger.info(f"Use {optimizer.__class__.__name__!r} | {optimizer_kwargs!r}")
    return optimizer


def gradient_norm(parameters):
    norm = 0
    for param in parameters:
        if param.grad is None:
            continue
        local_norm = param.grad.detach().data.norm(2)
        norm += local_norm.item() ** 2
    norm = norm**0.5
    return norm


def max_gradient(parameters):
    max_grad_value = float("-inf")
    for param in parameters:
        if param.grad is None:
            continue
        local_max_grad = param.grad.detach().data.abs().max()
        max_grad_value = max(max_grad_value, local_max_grad.item())
    return max_grad_value
