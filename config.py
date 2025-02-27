import json
import typing
import uuid
from dataclasses import asdict, dataclass

from peft import LoraConfig
from pyrallis import field


@dataclass
class BaseConfig:
    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def pop(self, attribute_name, default=None):
        if hasattr(self, attribute_name):
            value = getattr(self, attribute_name)
            delattr(self, attribute_name)
            return value
        else:
            return default

    def __str__(self):
        return json.dumps(asdict(self), indent=4)


@dataclass
class DataLoaderKwargs(BaseConfig):
    """Configuration for data loading parameters"""

    drop_last: bool = field(default=True)  # Whether to drop the last incomplete batch
    num_workers: int = field(default=8)  # Number of worker processes for data loading
    persistent_workers: bool = field(default=True)  # Keep worker processes alive between epochs
    pin_memory: bool = field(default=True)  # Pin memory for faster data transfer to GPU
    prefetch_factor: int = field(default=2)  # Number of batches to prefetch per worker


@dataclass
class StreamingKwargs(BaseConfig):
    """Configuration for data streaming parameters"""

    cache_limit: str = field(default="5tb")  # Maximum cache size limit
    download_timeout: int = field(default=12000)  # Timeout in seconds for downloads
    num_canonical_nodes: typing.Optional[int] = field(default=None)  # Number of canonical nodes to use
    shuffle: bool = field(default=True)  # Whether to shuffle the data
    batching_method: str = field(default="per_stream")  # Method used for batching data


@dataclass
class DataConfig(BaseConfig):
    """Configuration for data sources and processing"""

    remote: typing.Optional[typing.List[typing.Optional[str]]] = field(default=None)  # Remote data source paths
    local: typing.Optional[typing.List[typing.Optional[str]]] = field(default=None)  # Local data source paths
    batch_size: int = field(default=1)  # Training batch size
    video_key: str = field(default="video")  # Key for video data in dataset
    caption_key: str = field(default="caption")  # Key for caption data in dataset
    latents_key: str = field(default="latents")  # Key for latents in dataset
    prompt_embeds_key: str = field(default="prompt_embeds")  # Key for prompt embeddings
    latents_cond_key: str = field(default="latents_cond")  # Key for conditional latents
    prompt_attention_mask_key: str = field(default="prompt_attention_mask")  # Key for prompt attention mask
    pooled_prompt_embeds_key: str = field(default="pooled_prompt_embeds")  # Key for pooled prompt embeddings
    repeat: typing.Optional[typing.List] = field(default=None, is_mutable=True)  # Number of times to repeat dataset
    choose: typing.Optional[typing.List] = field(default=None, is_mutable=True)  # Indices to choose from dataset
    streaming_kwargs: StreamingKwargs = field(
        default_factory=StreamingKwargs, is_mutable=True
    )  # Streaming configuration
    dataloader_kwargs: DataLoaderKwargs = field(
        default_factory=DataLoaderKwargs, is_mutable=True
    )  # DataLoader configuration


@dataclass
class PretrainedModelConfig(BaseConfig):
    """Configuration for pretrained model loading"""

    pretrained_model_name_or_path: str = "hunyuanvideo-community/HunyuanVideo"  # Path or name of pretrained model
    revision: typing.Optional[str] = field(default=None)  # Specific model revision to use
    variant: typing.Optional[str] = field(default=None)  # Specific model variant to use


@dataclass
class NetworkConfig(BaseConfig):
    """Configuration for network architecture"""

    lora_rank: int = field(default=16)  # Rank for LoRA adaptation
    lora_alpha: int = field(default=16)  # Alpha scaling for LoRA
    target_modules: typing.Optional[typing.List[str]] = field(default=None, is_mutable=True)  # Target modules for LoRA
    lora_dropout: float = field(default=0.0)  # Dropout probability for LoRA layers
    train_norm_layers: bool = field(default=False)  # Whether to train normalization layers
    init_lora_weights: typing.Union[bool, str] = field(
        default=True
    )  # typing.Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "loftq"]
    lora_layers: typing.Optional[str] = field(default=None)


@dataclass
class ExperimentConfig(BaseConfig):
    """Configuration for experiment tracking"""

    output_dirpath: str = field(default="./outputs")  # Directory path for outputs
    random_seed: int = field(default=42)  # Random seed for reproducibility
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8], is_mutable=True)  # Unique run identifier
    name: str = field(default="whatever")  # Name of the experiment
    ic_debug: bool = field(default=False)  # Whether to use ic for debugging


@dataclass
class FlowMatchConfig(BaseConfig):
    """Configuration for flow matching parameters"""

    discrete_flow_shift: float = field(default=7.0)  # Shift for discrete flow
    timestep_sampling: str = field(default="sigma")  # "uniform", "sigmoid", "logit_normal"
    weighting_scheme: str = field(default="none")  # "sigma_sqrt", "cosmap", "none"
    sigmoid_scale: float = field(default=1.0)  # Scale for sigmoid function
    logit_mean: float = field(default=0.0)  # Mean for logit distribution
    logit_std: float = field(default=1.0)  # Standard deviation for logit distribution


@dataclass
class EmaConfig(BaseConfig):
    """Ema configuration"""

    use_ema: bool = field(default=False)
    ema_decay: float = field(default=0.99)
    ema_foreach_disable: bool = field(default=False)
    ema_device: str = field(default="accelerator")  # | typing.Literal["accelerator", "cpu"]
    ema_cpu_only: bool = field(default=False)
    ema_update_interval: typing.Optional[int] = field(default=None)


@dataclass
class TrainingHyperParametersConfig(BaseConfig):
    """Configuration for training hyperparameters"""

    mixed_precision: str = field(default="bf16")  # Mixed precision training type
    gradient_checkpointing: bool = field(default=True)  # Whether to use gradient checkpointing
    gradient_accumulation_steps: int = field(default=1)  # Number of gradient accumulation steps
    learning_rate: float = field(default=1e-04)  # Learning rate for training
    optimizer_type: str = field(default="torch.optim.AdamW")  # Type of optimizer to use
    optimizer_args: typing.List[str] = field(default=[], is_mutable=True)  # Additional optimizer arguments
    max_grad_norm: int = field(default=1.0)  # Maximum gradient norm for clipping
    grad_clip_method: str = field(default="norm")
    lr_scheduler: str = field(default="constant")  # Learning rate scheduler type
    lr_warmup_steps: int = field(default=0)  # Number of warmup steps
    lr_scheduler_num_cycles: int = field(default=1)  # Number of scheduler cycles
    lr_scheduler_power: float = field(default=0.9)  # Power for scheduler
    guidance_scale: int = field(default=1.0)  # Scale for guidance
    flow_match: FlowMatchConfig = field(default_factory=FlowMatchConfig, is_mutable=True)  # Flow matching configuration
    num_train_epochs: typing.Optional[int] = field(default=None)  # Number of training epochs
    max_train_steps: typing.Optional[int] = field(default=None)  # Maximum number of training steps
    caption_dropout_p: float = field(default=0.0)  # Dropout probability for captions
    ema: EmaConfig = field(default_factory=EmaConfig, is_mutable=True)  # EMA configuration
    gradient_precision: str = field(
        default="accelerator"
    )  # gradient precision from LLAMA paper | typing.Literal["accelerator", "fp32"]


@dataclass
class CheckpointConfig(BaseConfig):
    """Configuration for model checkpointing"""

    save_every_n_steps: typing.Optional[int] = field(default=None)  # Save checkpoint every N steps
    save_last_n_steps: typing.Optional[int] = field(default=None)  # Keep last N checkpoints
    resume_from_checkpoint: typing.Optional[str] = field(default=None)  # Path to checkpoint to resume from


@dataclass
class TorchDDPKwargs(BaseConfig):
    """Configuration for torch distributed parameters"""

    backend: str = field(default="nccl")
    find_unused_parameters: bool = field(default=False)
    gradient_as_bucket_view: bool = field(default=False)
    static_graph: bool = field(default=False)


@dataclass
class Config(BaseConfig):
    """Main configuration class combining all sub-configurations"""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig, is_mutable=True)  # Experiment configuration
    data: DataConfig = field(default_factory=DataConfig, is_mutable=True)  # Data configuration
    model: PretrainedModelConfig = field(default_factory=PretrainedModelConfig, is_mutable=True)  # Model configuration
    network: NetworkConfig = field(default_factory=NetworkConfig, is_mutable=True)  # Network configuration
    hparams: TrainingHyperParametersConfig = field(
        default_factory=TrainingHyperParametersConfig, is_mutable=True
    )  # Training hyperparameters
    checkpointing: CheckpointConfig = field(
        default_factory=CheckpointConfig, is_mutable=True
    )  # Checkpointing configuration
    ddp_kwargs: TorchDDPKwargs = field(default_factory=TorchDDPKwargs, is_mutable=True)


if __name__ == "__main__":
    import pyrallis

    cfg = pyrallis.parse(config_class=Config)
    print(f"Training {cfg}")
