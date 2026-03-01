import torch
import tensorflow as tf
import torch.nn as nn
import wandb
import os
import numpy as np
from tqdm import tqdm
import transformers
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup, SchedulerType
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names
from accelerate import PartialState
from dataclasses import  asdict
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from configs.twinvla_config import ModelArguments, TrainingArguments
from twinvla.datasets.rlds.utils.data_utils import save_dataset_statistics
from twinvla.model.twinvla import TwinVLA
from twinvla.datasets import load_datasets
from utils import cal_token_acc

# Ensure dataloader does not access GPU to avoid memory issues
tf.config.set_visible_devices([], "GPU")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "online"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

"""
Training script for TwinVLA.
This script handles the training loop, including data loading, model initialization,
optimization, and logging.
"""

distributed_state = PartialState()
device_id = distributed_state.local_process_index
torch.cuda.set_device(device_id)
torch.cuda.empty_cache()

parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()
dtype = torch.bfloat16 if training_args.bf16 else torch.float32
resumed = False

if training_args.resume or os.path.exists(training_args.output_dir):
    try:
        vla = TwinVLA(pretrained_path=training_args.output_dir, device=device_id, dtype=dtype)
        resumed = True
    except:
        vla = TwinVLA(model_args=model_args, device=device_id, dtype=dtype)
else:
    if training_args.pretrained_path is not None:
        vla = TwinVLA(pretrained_path=training_args.pretrained_path, device=device_id, dtype=dtype)
    else:
        vla = TwinVLA(model_args=model_args, device=device_id, dtype=dtype)

torch.cuda.empty_cache()

dataloader, dataset_statistics = load_datasets(vla.model, model_args, training_args, single_arm=False)
## SAVE Dataset Statistics
if not os.path.exists(training_args.output_dir):
    try:
        os.makedirs(training_args.output_dir)
    except:
        pass
if distributed_state.is_main_process:
    save_dataset_statistics(dataset_statistics, training_args.output_dir)
print('✈️ Dataset Loaded and Statistics Saved.')

if training_args.freeze_vision_backbone:
    for param in vla.model.vision_c.parameters():
        param.requires_grad = False

vla = DDP(vla, device_ids=[device_id], find_unused_parameters=False, gradient_as_bucket_view=True)
vla.train()
total_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
total_trainable_params_in_billion = total_trainable_params / (10**9)  # 1 Billion = 10^9
print(f'Total number of trainable parameters: {total_trainable_params_in_billion:.4f} billion')

if vla.module.config.modeling != 'tokenization':
    if vla.module.config.share_decoder:
        head_trainable_params = sum(p.numel() for p in vla.module.model.decoder_c.parameters() if p.requires_grad)
    else:
        head_trainable_params = sum(p.numel() for p in vla.module.model.decoder_r.parameters() if p.requires_grad)
        head_trainable_params += sum(p.numel() for p in vla.module.model.decoder_l.parameters() if p.requires_grad)
    head_trainable_params_in_billion = head_trainable_params / (10**9)  # 1 Billion = 10^9
    print(f'Total number of trainable Action head parameters: {head_trainable_params_in_billion:.4f} billion')

decay_parameters = get_parameter_names(vla, ALL_LAYERNORM_LAYERS)
decay_parameters_names = [name for name in decay_parameters if "bias" not in name]
decay_params = [p for n, p in vla.named_parameters() if (n in decay_parameters_names and p.requires_grad)]
nondecay_params = [p for n, p in vla.named_parameters() if (n not in decay_parameters_names and p.requires_grad)]
optimizer_grouped_parameters = [
    {
        "params": decay_params,
        "weight_decay": training_args.weight_decay,
        "lr": training_args.learning_rate,
        "eps":training_args.adam_epsilon,
        "betas":(training_args.adam_beta1, training_args.adam_beta2)
    },
    {
        "params": nondecay_params,
        "weight_decay": 0.0,
        "lr": training_args.learning_rate,
        "eps":training_args.adam_epsilon,
        "betas":(training_args.adam_beta1, training_args.adam_beta2)
    },
]
optimizer = AdamW(optimizer_grouped_parameters)

if training_args.lr_scheduler_type == 'linear':
    warmup_steps = int(training_args.max_steps * training_args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_args.max_steps)
elif training_args.lr_scheduler_type == 'cosine':
    warmup_steps = int(training_args.max_steps * training_args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_args.max_steps)
elif training_args.lr_scheduler_type == 'constant_with_warmup':
    warmup_steps = int(training_args.max_steps * training_args.warmup_ratio)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
print('🚀 Model Loaded,', vla.module.device, vla.module.dtype)

## Wandb Init
if distributed_state.is_main_process:
    wandb.init(
        entity=training_args.wandb_entity,
        project=training_args.wandb_project,
        name=f"TwinVLA_{model_args.model_type}_{training_args.data_mix}",
        config={**asdict(training_args), **asdict(model_args)}
    )

if resumed:
    if os.path.exists(f'{training_args.output_dir}/training_states.pth'):
        ckpt = torch.load(f'{training_args.output_dir}/training_states.pth', map_location="cpu")
        optimizer.load_state_dict(ckpt['optim'])
        print(training_args.lr_scheduler_type)
        if 'scheduler' in ckpt.keys() and training_args.lr_scheduler_type != SchedulerType.CONSTANT:
            scheduler.load_state_dict(ckpt['scheduler'])
        step = ckpt['step']
        print(f"Resuming from step {step}")
        del ckpt  # Free up memory
        torch.cuda.empty_cache()  # Ensure no VRAM consumption
    else:
        step = 0
else:
    step = 0

print('🚗 Training Start 🚕')
with tqdm(total=training_args.max_steps, initial=step, leave=False) as progress:
    vla.train()
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader, start=int(step * training_args.gradient_accumulation_steps)):
        vla.train()
        optimizer.zero_grad()
        # Use mixed precision training
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = vla(batch)
            loss = outputs['loss']
        normalized_loss = loss / training_args.gradient_accumulation_steps
        normalized_loss.backward()
        gradient_step_idx = batch_idx // training_args.gradient_accumulation_steps

        ## vla Update
        if (batch_idx + 1) % training_args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(vla.parameters(), max_norm=training_args.max_grad_norm)
            optimizer.step()
            if training_args.lr_scheduler_type != SchedulerType.CONSTANT:
                scheduler.step()
            progress.update()

        ## Logging        
        log_dict = {}
        if distributed_state.is_main_process and gradient_step_idx % training_args.log_steps == 0:
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if vla.module.config.modeling == 'tokenization':
                        token_acc = cal_token_acc(batch, outputs['logits'].detach())
                        log_dict['token_acc'] = token_acc
                    else:
                        action = batch['action']
                        action_len = action.shape[1]
                        action_dim = action.shape[2]
                        output_action = vla.module.model.inference(batch=batch, action_len=action_len, action_dim=action_dim)[0] # only action output
                        action_loss = nn.MSELoss()(output_action, action)
                        log_dict['action_loss'] = action_loss.item()
        
        if distributed_state.is_main_process:
            log_dict['loss'] = loss.item()
            log_dict['learning_rate'] = scheduler.get_last_lr()[0] if training_args.lr_scheduler_type != 'constant' else training_args.learning_rate
            wandb.log(log_dict, step=gradient_step_idx)

        # Save Checkpoint
        if gradient_step_idx > 0 and gradient_step_idx % training_args.save_steps == 0:
            if distributed_state.is_main_process:
                print(f"⭐ Saving Model Checkpoint for Step {gradient_step_idx} ⭐")
                ## SAVE - step specific
                save_dir = f'{training_args.output_dir}-{gradient_step_idx}'
                if not os.path.exists(save_dir):
                    try:
                        os.makedirs(save_dir)
                    except:
                        pass
                save_dataset_statistics(dataset_statistics, save_dir)
                vla.module.save_pretrained(save_dir)
                ## SAVE
                save_dir = f'{training_args.output_dir}'
                if not os.path.exists(save_dir):
                    try:
                        os.makedirs(save_dir)
                    except:
                        pass
                save_dataset_statistics(dataset_statistics, save_dir)
                vla.module.save_pretrained(save_dir)
                other_states = {
                    'step': gradient_step_idx,
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if training_args.lr_scheduler_type != SchedulerType.CONSTANT else torch.zeros(1)
                }
                torch.save(other_states, f'{save_dir}/training_states.pth')
                del other_states
            dist.barrier()

        if gradient_step_idx >= training_args.max_steps:
            print(f"🔥Max step {training_args.max_steps} reached! Stopping training... 🤗")
            break