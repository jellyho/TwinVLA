from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, List, Any
import transformers

@dataclass
class ModelArguments:
    model_type: str = field(default='Eagle2_1BVLA')
    model_path: Optional[str] = field(default=None)
    action_head: str = field(default='DiT')
    action_head_hidden_dim: int = field(default=1024)
    action_dim: int = field(default=7)
    action_len: int = field(default=32)
    state_dim: int = field(default=8)

    # Tokenization
    vocab_start: int = field(default=None)

    # Denoising
    denoiser: str = field(default='FM')
    train_denoising_steps: int = field(default=100)
    test_denoising_steps: int = field(default=10)
    num_readouts: int = field(default=1)
    readout_token_as_eos: bool = field(default=True)
    aggregation: str = field(default='None')
    dit_size: str = field(default='DiT-L')
    enable_cfg: bool = field(default=True)
    knowledge_insulation: bool = field(default=False)
    diffusion_batch: int = field(default=16)
    
    image_size: int = field(default=224)
    normalization: str = field(default='quantile')
    global_normalization: bool = field(default=True) # apply normalization across all dataset
    dataset_statistics_path: str = field(default=None) # path to dataset statistics for normalization

    # action herz interpolation
    hz_interpolate: int = field(default=None) # disable if None
    interpolate_gripper: bool = field(default=False) # if True, gripper action will be interpolated to match the hz_interpolate

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # resume?
    resume: bool = field(default=False)
    num_workers: int = field(default=8)

    # Directory Paths
    output_dir: str = field(default='checkpoints/singlevla_libero_suite')
    data_root_dir: str = field(default='./OXE')
    data_mix: str = field(default='fractal20220817_data_gresearch')

    # Wandb
    wandb_project: str = field(default='SingleVLA')
    wandb_entity: Optional[str] = field(default=None)

    # LoRA
    lora_enable: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    use_rslora: bool = False
    lora_dropout: float = 0.01
    lora_weight_path: str = ""
    lora_bias: str = "none"

    # Training
    bf16: bool = field(default=True)
    seed: int = field(default=42)
    batch_size: int = field(default=32)
    shuffle_buffer_size: int = field(default=10000)
    enable_autotune: bool = field(default=True)
    num_parallel_calls: int = field(default=16)
    traj_transform_threads: int = field(default=10)
    traj_read_treads: int = field(default=10)
    image_aug: bool = field(default=False)
    max_steps: int = field(default=50000)  
    save_steps: int = field(default=1000)
    log_grad: bool = field(default=False)
    log_steps: int = field(default=100)

    learning_rate: float = field(default=1e-4)
    lr_scheduler_type: str = field(default='constant')
    freeze_vision_backbone: bool = field(default=False)
    warmup_ratio: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=0.01)
    adam_epsilon: float= field(default=1e-8)
    gradient_accumulation_steps: int = field(default=1)

    # Quantization - curretnly not supported
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)

    # misc
    group_by_modality_length: bool = field(default=False)
    pretrained_ckpt: str = field(default='')