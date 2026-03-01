import torch
import torch.nn as nn
import numpy as np
from twinvla.model.utils import initialize_weights
from twinvla.model.base_models import modeling_head_type
import twinvla.model.singlevlas
from twinvla.datasets.rlds.utils.data_utils import load_statistics_from_json
from transformers.feature_extraction_utils import BatchFeature
from transformers import (
    AutoConfig,
    AutoModel,
)

## Wrapper class
class SingleVLA(nn.Module):
    """
    SingleVLA model wrapper for loading and inference.
    """
    def __init__(self, pretrained_path=None, model_args=None, device='cuda', dtype=torch.bfloat16):
        super().__init__()
        try:
            dataset_statistics = load_statistics_from_json(pretrained_path)
        except Exception:
            print('No dataset statistics found')
            dataset_statistics = None

        self.dataset_statistics = dataset_statistics
        self.device = device
        self.dtype = dtype
        first_load = False
        # First Loading
        if pretrained_path is None:
            assert model_args is not None, "Need model_args to initialize VLA"
            config_cls = getattr(twinvla.model.singlevlas, f'{model_args.model_type}Config')
            pretrained_path = config_cls.pretrained_path
            revision = getattr(config_cls, "revision", "main")
            self.config = config_cls.from_pretrained(pretrained_path, revision=revision)
            print(f'Initialize from pretrained VLM {pretrained_path} to create {model_args.model_type}')
            first_load = True
            # Update config
            for key, values in model_args.__dict__.items():
                setattr(self.config, key, values)
            self.config.modeling = modeling_head_type[self.config.action_head]
            self.config.model_path = pretrained_path
            if hasattr(self.config, 'auto_map'):
                self.config.auto_map = {}
            # print(self.config)
        else:
            self.config = AutoConfig.from_pretrained(pretrained_path)
            if hasattr(self.config, 'auto_map'):
                self.config.auto_map = {}
        revision = 'main' if pretrained_path == 'jellyho/TwinVLA' else getattr(self.config, "revision", "main")
        if self.config.knowledge_insulation:
            self.config.vision_config._attn_implementation = 'eager'
            self.config.llm_config._attn_implementation = 'eager'
        else:
            try:
                import flash_attn
            except ImportError:
                print("FlashAttention is not installed, using eager attention implementation.")
                self.config.vision_config._attn_implementation = 'eager'
                self.config.llm_config._attn_implementation = 'eager'
        # print(self.config)
        self.model = AutoModel.from_pretrained(
            pretrained_path,
            config=self.config,
            low_cpu_mem_usage=False,
            torch_dtype=dtype,
            trust_remote_code=True,
            revision=revision
        )
        self.model.to(device=device)
        ## Additional init logics
        if first_load and self.config.modeling != 'tokenization':
            initialize_weights(self.model.action_head)

        # Check for NaN parameters
        NaN = []
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                NaN.append(name)
        if len(NaN) > 0:
            print("NaN detected in parameters:")
            for name in NaN:
                print(name)
            raise ValueError(f"Make sure you init weights properly!")

        total_params = sum(p.numel() for p in self.model.parameters())
        total_params_in_billion = total_params / (10**9)  # 1 Billion = 10^9
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_trainable_params_in_billion = total_trainable_params / (10**9)  # 1 Billion = 10^9
        print(f'Total number of parameters: {total_params_in_billion:.4f} billion')
        print(f'Total number of trainable parameters: {total_trainable_params_in_billion:.4f} billion')

    def forward(self, batch):
        return self.model.base_forward(batch)

    # this assumes single-batch inference
    def predict_action(self, unnorm_key, instruction, proprio, image, image_wrist, action_len=None, cfg=1.0, num_denoising_steps=10, return_action_token=False):
        assert self.dataset_statistics is not None
    
        with torch.no_grad():
            with torch.autocast('cuda', dtype=self.dtype):
                action_dim = self.dataset_statistics[unnorm_key]['action']['mean'].shape[-1]
                action_len = action_len if action_len is not None else self.config.action_len
                normalized_proprio = self.normalize_state(unnorm_key, proprio)
                batch = BatchFeature(self.model.preprocess_inputs(image[np.newaxis, :].copy(), image_wrist[np.newaxis, :].copy(), instruction, action=None))
                ## Make it batch
                for key in batch.keys():
                    batch[key] = batch[key].unsqueeze(0)
                batch['proprio'] = torch.tensor(normalized_proprio).unsqueeze(0).unsqueeze(0) #assume only inputs one dim state vector
                batch = batch.to(device=self.model.device, dtype=self.dtype)
                if return_action_token:
                    normalized_action, action_token = self.model.inference(batch, action_len=action_len, action_dim=action_dim, cfg=cfg, num_denoising_steps=num_denoising_steps, return_action_token=return_action_token)
                else:
                    normalized_action = self.model.inference(batch, action_len=action_len, action_dim=action_dim, cfg=cfg, num_denoising_steps=num_denoising_steps)

        if type(normalized_action) == torch.Tensor:
            normalized_action = normalized_action.cpu().float().numpy()
        unnormalized_action = self.unnormalize_action(unnorm_key, normalized_action)

        if return_action_token:
            return unnormalized_action[0], action_token.detach().cpu().numpy()
        return unnormalized_action[0]

    def normalize_state(self, unnorm_key, state):
        if self.config.normalization == 'normal':
            pass
        elif self.config.normalization == 'quantile':
            mask = self.dataset_statistics[unnorm_key]['action']['mask']
            low = self.dataset_statistics[unnorm_key]['proprio']['q01']
            high = self.dataset_statistics[unnorm_key]['proprio']['q99']
            state= np.where(
                mask,  # Condition: apply unnormalization where mask is True
                (state - low) * 2 / (high - low + 1e-6) - 1,
                state  # Original state where mask is False
            )
        return state

    def unnormalize_action(self, unnorm_key, action):
        if self.config.normalization == 'normal':
            pass
        elif self.config.normalization == 'quantile':
            mask = self.dataset_statistics[unnorm_key]['action']['mask']
            low = self.dataset_statistics[unnorm_key]['action']['q01']
            high = self.dataset_statistics[unnorm_key]['action']['q99']
            
            unnormalized_action = np.where(
                mask,  # Condition: apply unnormalization where mask is True
                (action + 1) * (high - low + 1e-6) / 2 + low,
                action  # Original action where mask is False
            )
        return unnormalized_action

    def save_pretrained(self, directory):
        self.config.save_pretrained(directory)
        self.model.save_pretrained(directory)

class DummyVLAForDebug:
    def __init__(self, processor, tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer

    def preprocess_inputs(self, image, image_wrist, proprio, instruction, action=None):
        pass