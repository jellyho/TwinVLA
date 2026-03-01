from twinvla.model.base_models import SingleVLAMetaModel, SingleVLAConfig
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoProcessor, AutoConfig, AutoModel
from transformers import Kosmos2ForConditionalGeneration, Kosmos2Config

class Kosmos2VLAConfig(SingleVLAConfig, Kosmos2Config):
    model_type = "Kosmos2VLA"
    pretrained_path = "microsoft/kosmos-2-patch14-224"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Kosmos2VLA(Kosmos2ForConditionalGeneration, SingleVLAMetaModel):
    config_class = Kosmos2VLAConfig
    # _tied_weights_keys = []
    def __init__(self, config):
        super(Kosmos2VLA, self).__init__(config)
        self.init_model(config) 
        if config.modeling != 'tokenization':
            self.lm_head = nn.Identity()

    def init_processor_tokenizer(self, config):
        self.processor = AutoProcessor.from_pretrained(config.pretrained_path)
        self.tokenizer = self.processor.tokenizer

    def text_backbone(self):
        return self.text_model

    def hidden_dim(self):
        return self.config.text_config.embed_dim

    def vision_backbone(self):
        return self.vision_model

    def image_seq_len(self):
        return self.config.latent_query_num

    def image_start_token(self):
        return self.tokenizer("<image>")['input_ids'][1]

    def image_end_token(self):
        return self.tokenizer("</image>")['input_ids'][1]

    def process_image(self, image):
        return torch.tensor(self.processor.image_processor(image)['pixel_values'][0])

    def image_embeds(self, pixel_values):
        vision_model_output = self.vision_model(
            pixel_values=pixel_values,
        )
        image_embeds = self.vision_model.model.post_layernorm(vision_model_output[0])
        image_embeds = nn.functional.normalize(image_embeds, dim=-1)
        image_embeds, _ = self.image_to_text_projection(image_embeds)
        return image_embeds
    
    def system_prompt(self):
        return '<s>'

AutoConfig.register("Kosmos2VLA", Kosmos2VLAConfig)
AutoModel.register(Kosmos2VLAConfig, Kosmos2VLA)