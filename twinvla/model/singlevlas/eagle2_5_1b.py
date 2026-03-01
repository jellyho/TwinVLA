import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoConfig, AutoModel
from twinvla.model.base_models import SingleVLAMetaModel, SingleVLAConfig
# from transformers import Eagle2_1BConfig, Eagle2_1BForConditionalGeneration
# from twinvla.model.modeling.eagle2.modeling_eagle_chat import Eagle2ChatModel
# from twinvla.model.modeling.eagle2.configuration_eagle_chat import Eagle2ChatConfig
from twinvla.model.modeling.Eagle2_5_1B.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration
from twinvla.model.modeling.Eagle2_5_1B.configuration_eagle2_5_vl import Eagle2_5_VLConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
    
class Eagle2_1BVLAConfig(SingleVLAConfig, Eagle2_5_VLConfig):
    model_type = "Eagle2_1BVLA"
    pretrained_path = "nvidia/Eagle2-1B"
    def __init__(self, **kwargs):   
        super().__init__(**kwargs)

class Eagle2_2BVLAConfig(SingleVLAConfig, Eagle2_5_VLConfig):
    model_type = "Eagle2_2BVLA"
    pretrained_path = "nvidia/Eagle2-2B"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Eagle2_1BVLA(Eagle2_5_VLForConditionalGeneration, SingleVLAMetaModel):
    config_class = Eagle2_1BVLAConfig
    
    def __init__(self, config, **kwargs):
        super(Eagle2_1BVLA, self).__init__(config, **kwargs)
        self.init_model(config) 
        if config.modeling != 'tokenization':
            self.language_model.lm_head = nn.Identity()

    def init_processor_tokenizer(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path, use_fast=True)

    def text_backbone(self):
        return self.language_model

    def hidden_dim(self):
        return self.config.text_config.hidden_size

    def vision_backbone(self):
        return self.vision_model

    def image_seq_len(self):
        return 256

    def image_start_token(self):
        return 151665

    def image_end_token(self):
        return 151666

    def process_image(self, image):
        output = torch.tensor(image[0], dtype=torch.float32) / 255.0  # Convert to float and normalize in-place
        output = T.functional.resize(output.permute(2, 0, 1), (448, 448), interpolation=InterpolationMode.BICUBIC)
        output = T.functional.normalize(output, mean=SIGLIP_MEAN, std=SIGLIP_STD)  # Normalize using PyTorch
        return output

    def image_embeds(self, pixel_values):
        image_features = self.extract_feature(pixel_values)
        return image_features

    def system_prompt(self):
        return '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
    
class Eagle2_2BVLA(Eagle2_1BVLA):
    config_class = Eagle2_2BVLAConfig

AutoConfig.register("Eagle2_1BVLA", Eagle2_1BVLAConfig)
AutoModel.register(Eagle2_1BVLAConfig, Eagle2_1BVLA)

AutoConfig.register("Eagle2_2BVLA", Eagle2_2BVLAConfig)
AutoModel.register(Eagle2_2BVLAConfig, Eagle2_2BVLA)
