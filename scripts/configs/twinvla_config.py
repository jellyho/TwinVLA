from dataclasses import dataclass, field
from typing import Optional
import transformers

@dataclass
class ModelArguments:
    model_type: str = field(default='Eagle2_1BTwinVLA')
    model_path: Optional[str] = field(default=None)
    singlevla_pretrained_path: Optional[str] = field(default='jellyho/TwinVLA') # path to pretrained singlevla model
    share_vision: bool = field(default=True) # use shared vision encoder
    share_decoder: bool = field(default=True) # use shared action decoder
    share_embed_tokens: bool = field(default=True) # use shared embedding tokens

    global_normalization: bool = field(default=False)
    action_len: int = field(default=20)
    hz_interpolate: int = field(default=None) # disable if None
    interpolate_gripper: bool = field(default=False)
    knowledge_insulation: bool = field(default=False) #
    normalization: str = field(default='quantile')
    attn_reweighting: bool = field(default=True) # For ablation
    dit_scratch: bool = field(default=False) # For ablation
    enable_moe: bool = field(default=True) # For ablation
    enable_joint_attn: bool = field(default=True) # For ablation


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # resume?
    resume: bool = field(default=False)
    pretrained_path: Optional[str] = field(default=None)
    num_workers: int = field(default=8)
    # Directory Paths
    output_dir: str = field(default='checkpoints/singlevla_libero_suite')
    data_root_dir: str = field(default='./tabletop-simulation-rlds-v3')
    data_mix: str = field(default='jellyho/aloha_handover_box')

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
    batch_size: int = field(default=8)
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
    lr_scheduler_type: str = field(default='cosine')
    freeze_vision_backbone: bool = field(default=False)
    warmup_ratio: float = field(default=0.01)
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=1e-5)
    adam_epsilon: float= field(default=1e-8)
    adam_beta1: float= field(default=0.95)
    adam_beta2: float= field(default=0.999)
    gradient_accumulation_steps: int = field(default=1)

    # etc
    group_by_modality_length: bool = field(default=False)