import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
from twinvla.model.base_models import TwinVLAMetaModel, TwinVLAConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

class Eagle2_1BTwinVLAConfig(TwinVLAConfig):
    model_type = "Eagle2_1BTwinVLA"
    pretrained_path = "nvidia/Eagle2-1B"
    revision = "2d2621147c8f3b55676cb68a8354a62f1d8df877"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Eagle2_1BTwinVLA(PreTrainedModel, TwinVLAMetaModel):
    config_class = Eagle2_1BTwinVLAConfig
    
    def __init__(self, config):
        super(Eagle2_1BTwinVLA, self).__init__(config)
        self.construct_twinvla(config.singlevla_config_path)

    def init_processor_tokenizer(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path, use_fast=True)

    def hidden_dim(self):
        return self.singlevla_config.llm_config.hidden_size

    def image_seq_len(self):
        return 256

    def image_start_token(self):
        return 151665

    def image_end_token(self):
        return 151666
    
    def process_image(self, image):
        if not torch.is_tensor(image):
            output = torch.tensor(image[0], dtype=torch.float32) / 255.0
        else:
            output = image.float()[0] / 255.0
        # output = torch.tensor(image[0], dtype=torch.float32) / 255.0  # Convert to float and normalize in-place
        output = T.functional.resize(output.permute(2, 0, 1), (448, 448), interpolation=InterpolationMode.BICUBIC)
        output = T.functional.normalize(output, mean=SIGLIP_MEAN, std=SIGLIP_STD)  # Normalize using PyTorch
        return output
    
    def system_prompt(self):
        return '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    
    def in_layernorm(self, model, idx):
        return model[idx].input_layernorm
    
    def out_layernorm(self, model, idx):
        return model[idx].post_attention_layernorm
    
    def attn(self, model, idx):
        return model[idx].self_attn
    
    def mlp(self, model, idx):
        return model[idx].mlp

    def copy_backbone_target(self):
        return 'language_model.model.layers'
    
    def copy_vision_target(self):
        return 'vision_model'

    def copy_embed_tokens_target(self):
        return 'language_model.model.embed_tokens'
    
    def copy_norm_target(self):
        return 'language_model.model.norm'
    
    def copy_pos_embed_target(self):
        return 'language_model.model.rotary_emb'
    
    def copy_other_target(self):
        ## This is for the case that we need to copy other modules ex(. vision projector)
        return ['mlp1']

    def image_embeds(self, pixel_values):
        ## TODO : deal with split vision encoder
        vit_embeds = self.vision_c(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True)
        # if there is vit_embeds.last_hidden_state, use it.
        if hasattr(vit_embeds, 'last_hidden_state'):
            vit_embeds = vit_embeds.last_hidden_state
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=0.5) # torch.Size([B, 1024, 1024]) -> torch.Size([B, 16, 16, 4096])
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]) # torch.Size([B, 16, 16, 4096]) -> torch.Size([B, 256, 4096])
        vit_embeds = self.mlp1(vit_embeds)#.to(pixel_values.device)

        return vit_embeds

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


AutoConfig.register("Eagle2_1BTwinVLA", Eagle2_1BTwinVLAConfig)
AutoModel.register(Eagle2_1BTwinVLAConfig, Eagle2_1BTwinVLA)