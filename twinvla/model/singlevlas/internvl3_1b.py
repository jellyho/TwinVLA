import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from transformers import AutoProcessor, AutoConfig, AutoModel, AutoTokenizer
from twinvla.model.base_models import SingleVLAMetaModel, SingleVLAConfig
from twinvla.model.modeling.InternVL3_1B.configuration_internvl_chat import InternVLChatConfig
from twinvla.model.modeling.InternVL3_1B.modeling_internvl_chat import InternVLChatModel
from PIL import Image

MEAN = np.array((0.485, 0.456, 0.406)).reshape(1, 1, 3)
STD = np.array((0.229, 0.224, 0.225)).reshape(1, 1, 3)

class InternVL3_1BVLAConfig(SingleVLAConfig, InternVLChatConfig):
    model_type = "InternVL3_1BVLA"
    pretrained_path = "OpenGVLab/InternVL3-1B"
    def __init__(self, **kwargs):   
        super().__init__(**kwargs)

class InternVL3_1BVLA(InternVLChatModel, SingleVLAMetaModel):
    config_class = InternVL3_1BVLAConfig
    
    def __init__(self, config, **kwargs):
        super(InternVL3_1BVLA, self).__init__(config, **kwargs)
        self.init_model(config) 
        if config.modeling != 'tokenization':
            self.language_model.lm_head = nn.Identity()

    def init_processor_tokenizer(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_path, trust_remote_code=True, use_fast=False)

    def text_backbone(self):
        return self

    def hidden_dim(self):
        return self.config.llm_config.hidden_size

    def vision_backbone(self):
        return self.vision_model

    def image_seq_len(self):
        return self.num_image_token

    def image_start_token(self):
        return 151665

    def image_end_token(self):
        return 151666

    def process_image(self, image):
        output = Image.fromarray(image[0].astype(np.uint8)).resize((448, 448))  # Efficient resize
        output = np.asarray(output, dtype=np.float32) / 255.0  # Avoid extra np.array()
        output = (output - MEAN) / STD  # Normalize directly
        output = torch.tensor(output, dtype=torch.float32).permute(2, 0, 1)
        return output

    def image_embeds(self, pixel_values):
        image_embeds = self.extract_feature(pixel_values)
        return image_embeds

    def system_prompt(self):
        return '<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>\n<|im_start|>user\n'

AutoConfig.register("InternVL3_1BVLA", InternVL3_1BVLAConfig)
AutoModel.register(InternVL3_1BVLAConfig, InternVL3_1BVLA)
