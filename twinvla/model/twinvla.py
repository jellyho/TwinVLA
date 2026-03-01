import torch
import torch.nn as nn
import numpy as np
from twinvla.datasets.rlds.utils.data_utils import load_statistics_from_json
from transformers.feature_extraction_utils import BatchFeature
from transformers import (
    AutoConfig,
    AutoModel,
)
import twinvla.model.twinvlas
import twinvla.model.singlevlas # Import SingleVLA configs
from huggingface_hub import hf_hub_download

## Wrapper class
class TwinVLA(nn.Module):
    """
    TwinVLA model wrapper for loading and inference.
    Handles initialization from SingleVLA checkpoints or existing TwinVLA checkpoints.
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
        load_from_singlevla = model_args is not None # If you give model_args, it will load from singlevla pretrained path
        
        if load_from_singlevla:
            singlevla_pretrained_path = model_args.singlevla_pretrained_path
            twinvla_config_cls = getattr(twinvla.model.twinvlas, f'{model_args.model_type}Config')
            self.config = twinvla_config_cls()
            self.config.singlevla_pretrained_path = singlevla_pretrained_path
            self.config.singlevla_config_path = singlevla_pretrained_path
            print(f'Initialize from pretrained SingleVLA {singlevla_pretrained_path} to create {model_args.model_type}')
            # Update config
            for key, values in model_args.__dict__.items():
                setattr(self.config, key, values)
            if hasattr(self.config, 'auto_map'):
                self.config.auto_map = {}
            self.model = AutoModel.from_config(
                config=self.config,
            )
        else:
            print('Loading from trained TwinVLA checkpoints')
            self.config = AutoConfig.from_pretrained(pretrained_path)
            if hasattr(self.config, 'auto_map'):
                self.config.auto_map = {}

            ## HF resolving the config path
            try:
                hf_config_path = hf_hub_download(
                    repo_id=pretrained_path,
                    filename="config.json",
                    subfolder="singlevla_config",     
                    revision="main"
                )
                self.config.singlevla_config_path = hf_config_path
            except:
                self.config.singlevla_config_path = f'{pretrained_path}/singlevla_config'
                
            self.model = AutoModel.from_pretrained(
                pretrained_path,
                config=self.config,
                low_cpu_mem_usage=False,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        self.model.to(device=device)
        self.config = self.model.config
        self.config.singlevla_pretrained_path = None

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
        batch = batch.to(device=self.device, dtype=self.dtype)
        return self.model.base_forward(batch)[0] # only take hidden states

    def predict_action(self, unnorm_key, instruction, proprio, image, image_wrist_r, image_wrist_l, action_len=None, cfg=1.5, output_attn=False, output_action_token=False):
        """
        Runs inference to predict actions given an instruction and observation images.
        """
        assert self.dataset_statistics is not None
        self.model.eval()
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=self.dtype):
                action_dim = self.dataset_statistics[unnorm_key]['action']['mean'].shape[-1]
                action_len = action_len if action_len is not None else self.config.action_len
                normalized_proprio = self.normalize_state(unnorm_key, proprio)
                batch = BatchFeature(self.model.preprocess_inputs(image[np.newaxis, :].copy(), image_wrist_r[np.newaxis, :].copy(), image_wrist_l[np.newaxis, :].copy(), instruction, action=None))
                ## Make it batch
                for key in batch.keys():
                    batch[key] = batch[key].unsqueeze(0)
                batch['proprio'] = torch.tensor(normalized_proprio).unsqueeze(0).unsqueeze(0) #assume only inputs one dim state vector
                batch = batch.to(device=self.model.device, dtype=self.dtype)
                normalized_action, val = self.model.inference(batch, action_len=action_len, action_dim=action_dim, output_attn=output_attn, cfg=cfg, output_action_token=output_action_token)
        if type(normalized_action) == torch.Tensor:
            normalized_action = normalized_action.cpu().float().numpy()
        unnormalized_action = self.unnormalize_action(unnorm_key, normalized_action)
        if output_attn:
            return unnormalized_action[0], val, batch['modal_ids'].detach().cpu().numpy()
        if output_action_token:
            return unnormalized_action[0], val.detach().cpu().float().numpy()
        return unnormalized_action[0]

    def normalize_state(self, unnorm_key, state):
        if self.model.config.normalization == 'normal':
            pass
        elif self.model.config.normalization == 'minmax':
            low = self.dataset_statistics[unnorm_key]['proprio']['min']
            high = self.dataset_statistics[unnorm_key]['proprio']['max']
            mask = self.dataset_statistics[unnorm_key]['proprio']['mask']
            state = np.where(
                mask,  # Condition: apply normalization where mask is True
                (state - low) * 2 / (high - low + 1e-6) - 1,
                state  # Original state where mask is False
            )
        elif self.model.config.normalization == 'quantile':
            # proprio statistics
            low = self.dataset_statistics[unnorm_key]['proprio']['q01']
            high = self.dataset_statistics[unnorm_key]['proprio']['q99']
            mask = self.dataset_statistics[unnorm_key]['proprio']['mask']
            # If statistics are half the size of state, apply to each half
            if low.shape[-1] * 2 == state.shape[-1]:
                half = state.shape[-1] // 2
                state_1 = np.where(
                    mask, 
                    (state[..., :half] - low) * 2 / (high - low + 1e-6) - 1,
                    state[..., :half]
                )
                state_2 = np.where(
                    mask, 
                    (state[..., half:] - low) * 2 / (high - low + 1e-6) - 1,
                    state[..., half:]
                )
                state = np.concatenate([state_1, state_2], axis=-1)
            else:
                state = np.where(
                    mask,  # Condition: apply normalization where mask is True
                    (state - low) * 2 / (high - low + 1e-6) - 1,
                    state  # Original state where mask is False
                )
        return state

    def unnormalize_action(self, unnorm_key, action):
        if self.model.config.normalization == 'normal':
            pass
        elif self.model.config.normalization == 'quantile':
            low = self.dataset_statistics[unnorm_key]['action']['q01']
            high = self.dataset_statistics[unnorm_key]['action']['q99']
            mask = self.dataset_statistics[unnorm_key]['action']['mask']
            # If statistics are half the size of action, apply to each half
            if low.shape[-1] * 2 == action.shape[-1]:
                half = action.shape[-1] // 2
                action_1 = np.where(
                    mask,
                    (action[..., :half] + 1) * (high - low + 1e-6) / 2 + low,
                    action[..., :half]
                )
                action_2 = np.where(
                    mask,
                    (action[..., half:] + 1) * (high - low + 1e-6) / 2 + low,
                    action[..., half:]
                )
                action = np.concatenate([action_1, action_2], axis=-1)
            else:
                action = np.where(
                    mask,  # Condition: apply unnormalization where mask is True
                    (action + 1) * (high - low + 1e-6) / 2 + low,
                    action  # Original action where mask is False
                )
        return action

    def save_pretrained(self, directory):
        self.config.save_pretrained(directory)
        self.model.singlevla_config.save_pretrained(f'{directory}/singlevla_config')
        self.model.save_pretrained(directory)