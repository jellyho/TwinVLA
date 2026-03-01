import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoConfig, AutoModel, AutoTokenizer
from twinvla.model.base_models import SingleVLAMetaModel, SingleVLAConfig
from twinvla.model.modeling.mobilevlmv2.model.mobilellama import MobileVLMConfig, MobileLlamaForCausalLM
from twinvla.model.modeling.mobilevlmv2.utils import expand2square
    
class MobileVLMVLAConfig(SingleVLAConfig, MobileVLMConfig):
    model_type = "MobileVLMVLA"
    pretrained_path = "mtgv/MobileVLM_V2-1.7B"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MobileVLMVLA(MobileLlamaForCausalLM, SingleVLAMetaModel):
    config_class = MobileVLMVLAConfig
    def __init__(self, config):
        super(MobileVLMVLA, self).__init__(config)
        self.init_model(config)

    def init_processor_tokenizer(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained("mtgv/MobileVLM_V2-1.7B", use_fast=False)
        self.model.get_vision_tower().load_model()
        self.processor = self.model.get_vision_tower().image_processor
        if config.modeling != 'tokenization':
            self.lm_head = nn.Identity()

    def text_backbone(self):
        return self

    def hidden_dim(self):
        return self.config.hidden_size

    def vision_backbone(self):
        return self.model.get_vision_tower()

    def image_seq_len(self):
        return 144

    def image_start_token(self):
        return None

    def image_end_token(self):
        return None

    def process_image(self, image):
        image = Image.fromarray(image[0])
        image = expand2square(image, tuple(int(x*255) for x in self.processor.image_mean))
        image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image

    def image_embeds(self, pixel_values):
        image_features = self.get_model().get_vision_tower()(pixel_values)
        image_features = self.get_model().mm_projector(image_features).contiguous()
        return image_features

AutoConfig.register("MobileVLMVLA", MobileVLMVLAConfig)
AutoModel.register(MobileVLMVLAConfig, MobileVLMVLA)
