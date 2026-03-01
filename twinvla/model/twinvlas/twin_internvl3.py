import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
from twinvla.model.base_models import TwinVLAMetaModel, TwinVLAConfig
import numpy as np
from PIL import Image

MEAN = np.array((0.485, 0.456, 0.406)).reshape(1, 1, 3)
STD = np.array((0.229, 0.224, 0.225)).reshape(1, 1, 3)

class InternVL3_1BTwinVLAConfig(TwinVLAConfig):
    model_type = "InternVL3_1BTwinVLA"
    pretrained_path = "OpenGVLab/InternVL3-1B"
    def __init__(self, **kwargs):   
        super().__init__(**kwargs)

class InternVL3_1BTwinVLA(PreTrainedModel, TwinVLAMetaModel):
    config_class = InternVL3_1BTwinVLAConfig
    
    def __init__(self, config):
        super(InternVL3_1BTwinVLA, self).__init__(config)
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
        output = Image.fromarray(image[0]).resize((448, 448))  # Efficient resize
        output = np.asarray(output, dtype=np.float32) / 255.0  # Avoid extra np.array()
        output = (output - MEAN) / STD  # Normalize directly
        output = torch.tensor(output, dtype=torch.float32).permute(2, 0, 1)
        return output
    
    def system_prompt(self):
        return '<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>\n<|im_start|>user\n'
    
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
        vit_embeds = vit_embeds[:, 1:, :]
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


AutoConfig.register("InternVL3_1BTwinVLA", InternVL3_1BTwinVLAConfig)
AutoModel.register(InternVL3_1BTwinVLAConfig, InternVL3_1BTwinVLA)