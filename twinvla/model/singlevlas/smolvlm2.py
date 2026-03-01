import torch
import torch.nn as nn
import numpy as np
from transformers import AutoProcessor, AutoConfig, AutoModel
from twinvla.model.base_models import SingleVLAMetaModel, SingleVLAConfig
from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration
    
class SmolVLM2VLAConfig(SingleVLAConfig, SmolVLMConfig):
    model_type = "SmolVLM2VLA"
    pretrained_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SmolVLM22BVLAConfig(SingleVLAConfig, SmolVLMConfig):
    model_type = "SmolVLM22BVLA"
    pretrained_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


## Make sure you commented modeling_smolvlm.py line 896, 897
class SmolVLM2VLA(SmolVLMForConditionalGeneration, SingleVLAMetaModel):
    config_class = SmolVLM2VLAConfig
    
    def __init__(self, config):
        super(SmolVLM2VLA, self).__init__(config)
        self.init_model(config) 
        if config.modeling != 'tokenization':
            self.lm_head = nn.Identity()

    def init_processor_tokenizer(self, config):
        self.processor = AutoProcessor.from_pretrained(config.pretrained_path, use_fast=True)
        self.processor.do_image_splitting = False
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor

    def text_backbone(self):
        return self

    def hidden_dim(self):
        return self.config.text_config.hidden_size

    def vision_backbone(self):
        return self.model.vision_model

    def image_seq_len(self):
        return self.processor.image_seq_len

    def image_start_token(self):
        return self.tokenizer("<fake_token_around_image>")['input_ids'][0]

    def image_end_token(self):
        return self.tokenizer("<fake_token_around_image>")['input_ids'][0]

    def process_image(self, image):
        pixel_values = torch.tensor(self.image_processor(image, do_image_splitting=False)['pixel_values'][0][0])
        return pixel_values

    def image_embeds(self, pixel_values):
        # B, C, H, W
        pixel_attention_mask = torch.ones(
            size=[pixel_values.shape[i] for i in (0, 2, 3)],
            dtype=torch.bool,
            device=pixel_values.device,
        )
        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
        patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        image_hidden_states = self.vision_backbone()(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state
        image_embeds = self.model.connector(image_hidden_states)
        return image_embeds

    def system_prompt(self):
        return '<|im_start|>User:'
    
## Make sure you commented modeling_smolvlm.py line 896, 897
class SmolVLM22BVLA(SmolVLM2VLA):
    config_class = SmolVLM22BVLAConfig

AutoConfig.register("SmolVLM2VLA", SmolVLM2VLAConfig)
AutoModel.register(SmolVLM2VLAConfig, SmolVLM2VLA)
AutoConfig.register("SmolVLM22BVLA", SmolVLM22BVLAConfig)
AutoModel.register(SmolVLM22BVLAConfig, SmolVLM22BVLA)
