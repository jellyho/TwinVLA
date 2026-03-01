import torch
import torch.nn as nn
import torch.nn.functional as F
from twinvla.model.heads.utils import get_noise_scheduler
from twinvla.model.heads.DiT_policy import DiT

class MLPHead(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPHead, self).__init__()
        
        # Define a list to hold each layer
        layers = []
        
        # First hidden layer (input to first hidden size)
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Add activation
        
        # Additional hidden layers (for each pair in hidden_sizes)
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        
        # Output layer (last hidden size to output)
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)

    def _init_weights(self):
        # Apply custom initialization to each module in self.modules()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)    

# Create model sizes of ActionModels
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

# Model size
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L}

# Create ActionModel
class DiTPolicy(nn.Module):
    def __init__(self, 
                 token_size=2048, 
                 model_type='DiT-L', 
                 in_channels=7, 
                 state_dim=8,
                 hidden_dim=2048,
                 future_action_window_size=16, 
                 past_action_window_size=0,
                 diffusion_steps=100,
                 test_denoising_steps=10,
                 denoiser='DDIM',
                 enable_cfg=False,
                 diffusion_batch=16,
                 ):
        super().__init__()
        self.in_channels = in_channels
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.denoiser = denoiser
        self.noise_scheduler = get_noise_scheduler(denoiser, diffusion_steps)
        self.test_denoising_steps=test_denoising_steps
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.net = DiT_models[model_type](
                                        token_size=token_size, 
                                        in_channels=in_channels, 
                                        class_dropout_prob=0.1 if enable_cfg else 0.0, 
                                        learn_sigma=False, 
                                        future_action_window_size=future_action_window_size, 
                                        past_action_window_size=past_action_window_size
                                        )
        self.combine = nn.Linear(token_size + state_dim, token_size)
        self.repeated_diffusion_steps = diffusion_batch
        self.dtype = next(self.parameters()).dtype

        # self.proj = nn.Linear(token_size, hidden_dim)

    # Given condition z and ground truth token x, compute loss
    def forward(self, x, z, state):
        self.net.training=True
        z = self.combine(torch.cat([z.squeeze(1), state], dim=-1))
        z = z.unsqueeze(1) # [B, 1, H]
        actions_repeated = x.repeat(self.repeated_diffusion_steps, 1, 1)
        conditions_repeated = z.repeat(self.repeated_diffusion_steps, 1, 1)                
        noise = torch.randn_like(actions_repeated) # [BS, T, C]
        BS = noise.shape[0]

        timestep = self.noise_scheduler.sample_time(BS).to(device=actions_repeated.device) # BS,

        x_t = self.noise_scheduler.add_noise(actions_repeated, noise, timestep)
        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, conditions_repeated)

        assert noise_pred.shape == noise.shape == actions_repeated.shape
        # Compute L2 loss
        noise_target = self.noise_scheduler.target(actions_repeated, noise)
        loss = ((noise_pred - noise_target) ** 2).mean()
        return loss
    
    def denoise(self, z, state, denoising_steps=None, cfg=1.0):
        self.net.training = False
        if cfg > 1.0:
            return self.denoise_cfg(z, state, denoising_steps, cfg)
        denoising_steps = self.test_denoising_steps if denoising_steps is None else denoising_steps
        # encode
        z = self.combine(torch.cat([z.squeeze(1), state], dim=-1)) # Should be change to be deal with agg
        z = z.unsqueeze(1)      
        B = z.shape[0]
        noisy_action = torch.randn((B, self.future_action_window_size, self.in_channels)).to(z.device)
        naction = noisy_action.to(dtype=z.dtype)
        self.noise_scheduler.set_timesteps(denoising_steps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            kt = self.noise_scheduler.time_tensor(k).to(naction.device)
            noise_pred = self.net(naction, kt, z)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

        return naction
    
    def denoise_cfg(self, z, state, denoising_steps=None, cfg_scale=1.0):
        self.net.training = False
        denoising_steps = self.test_denoising_steps if denoising_steps is None else denoising_steps
        # encode
        B = z.shape[0]
        z = self.combine(torch.cat([z.squeeze(1), state], dim=-1)) # Should be change to be deal with agg
        z = z.unsqueeze(1)
        # print(z.shape, state.shape)
        uncondition = self.net.z_embedder.uncondition.unsqueeze(0).expand(B, 1, -1)
        z = torch.cat([z, uncondition], 0)
        noisy_action = torch.randn((2*B, self.future_action_window_size, self.in_channels)).to(z.device)
        naction = noisy_action.to(dtype=z.dtype)
        self.noise_scheduler.set_timesteps(denoising_steps)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            kt = self.noise_scheduler.time_tensor(k).to(naction.device)

            half = naction[:len(naction) // 2] # B, L, D
            combined = torch.cat([half, half], dim=0).to(next(self.net.x_embedder.parameters()).dtype) #2B L D
            noise_pred = self.net(combined, kt, z)
            cond_eps, uncond_eps = torch.split(noise_pred, len(noise_pred) // 2, dim=0) # B L D
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            naction = self.noise_scheduler.step(
                model_output=eps,
                timestep=k,
                sample=naction
            ).prev_sample


            naction = torch.clamp(naction, -5, 5)
        return naction[:len(naction) // 2, :, :self.in_channels]