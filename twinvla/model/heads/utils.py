import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from dataclasses import dataclass, field
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

class Hidden_Projector(nn.Module):
    def __init__(self, hidden_dim, middle_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, middle_dim),
            nn.LayerNorm(middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    def forward(self, x):
        return self.model(x)
    
class MultiheadAttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention_layer = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        query = self.query.expand(x.shape[0], -1, -1)
        pooled, _ = self.attention_layer(query, x, x)  # (1, 1, H)
        pooled = self.layer_norm(pooled)
        return pooled  # (1, H) returned
    
class AveragePooling(nn.Module):
    def __init__(self, embed_dim):
        self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.norm_after_pool = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.global_1d_pool(x.permute(0, 2, 1)).squeeze(-1)
        x = self.norm_after_pool(x).unsqueeze(1)
        return x
    
def swish(x):
    return x * torch.sigmoid(x)

class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class CategorySpecificMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = F.relu(self.layer1(x))
        return self.layer2(hidden)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = nn.Linear(action_dim, hidden_size)  # (d -> w)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = nn.Linear(hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x)
        return x
    

## Wrappers for denoising methods #######################################################################

class DDIM_Scheduler:
    def __init__(self, num_train_timesteps):
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
    
    def sample_time(self, batch_size):
        return torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (batch_size,))
    
    def add_noise(self, actions, noise, timesteps):
        return self.noise_scheduler.add_noise(actions, noise, timesteps)
    
    def target(self, actions, noise):
        return noise
    
    def step(self, model_output, timestep, sample):
        return self.noise_scheduler.step(model_output, timestep, sample)

    def time_tensor(self, timestep):
        return torch.tensor([timestep], dtype=torch.long)
    
    def set_timesteps(self, denoising_steps):
        self.noise_scheduler.set_timesteps(denoising_steps)
        self.timesteps = self.noise_scheduler.timesteps      

class FlowMatchingScheduler:
    def __init__(self, num_train_timesteps):
        self.flow_sig_min = 0.001
        flow_alpha = 1.5
        flow_beta = 1.0
        self.flow_t_max = 1 - self.flow_sig_min
        self.flow_beta_dist = torch.distributions.Beta(torch.tensor(flow_alpha, dtype=torch.float32), torch.tensor(flow_beta, dtype=torch.float32))

    def sample_time(self, batch_size):
        z = self.flow_beta_dist.sample((batch_size, ))
        t = self.flow_t_max * z + self.flow_sig_min
        return t
    
    def add_noise(self, actions, noise, timesteps):
        timesteps = timesteps[:, None, None]
        noise = timesteps * noise + (1 - timesteps) * actions
        return noise
    
    def target(self, actions, noise):
        return noise - actions
    
    def step(self, model_output, timestep, sample):
        sample += self.delta_t * model_output
        return SimpleNamespace(prev_sample=sample)
    
    def time_tensor(self, timestep):
        return torch.tensor([1 - (timestep) / self.denoising_steps])

    def set_timesteps(self, denoising_steps):
        assert denoising_steps < 1000, 'You cannot exceed 1000'
        self.denoising_steps = denoising_steps
        self.delta_t = - 1 / self.denoising_steps
        self.timesteps = range(self.denoising_steps)
    
class DPMSolverScheduler:
    def __init__(self, num_train_timesteps):
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            prediction_type='sample'
        )
        self.scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            prediction_type='sample'
        )

    def sample_time(self, batch_size):
        return torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,))

    def add_noise(self, actions, noise, timesteps):
        return self.scheduler.add_noise(actions, noise, timesteps)

    def target(self, actions, noise):
        return actions

    def step(self, model_output, timestep, sample):
        return self.scheduler_sample.step(model_output, timestep, sample)
    
    def time_tensor(self, timestep):
        return torch.tensor([timestep], dtype=torch.long)
    
    def set_timesteps(self, denoising_steps):
        self.scheduler_sample.set_timesteps(denoising_steps)
        self.timesteps = self.scheduler_sample.timesteps


def get_noise_scheduler(denoiser, training_steps):
    if denoiser=='DDIM':
        noise_scheduler = DDIM_Scheduler(
            num_train_timesteps=training_steps
        )
    elif denoiser=='FM':
        noise_scheduler = FlowMatchingScheduler(
            num_train_timesteps=training_steps
        )
    elif denoiser=='DPM':
        noise_scheduler = DPMSolverScheduler(
            num_train_timesteps=training_steps
        )
    return noise_scheduler