import torch
import torch.nn as nn
import numpy as np
from transformers import AutoProcessor, AutoConfig, AutoModel
from twinvla.model.base_models import SingleVLAMetaModel, SingleVLAConfig
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
class Qwen2VLVLAConfig(SingleVLAConfig, Qwen2VLConfig):
    model_type = "Qwen2VLVLA"
    pretrained_path = "Qwen/Qwen2-VL-2B-Instruct"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Qwen2VLVLA(Qwen2VLForConditionalGeneration, SingleVLAMetaModel):
    config_class = Qwen2VLVLAConfig
    # _tied_weights_keys = []
    def __init__(self, config):
        super(Qwen2VLVLA, self).__init__(config)
        self.init_model(config)
        self.tie_weights = lambda: None  # tie_weights()를 비활성화
        self.lm_head.weight = nn.Parameter(self.lm_head.weight.clone())
        # for DDP training, we need to detach unused params
        if config.modeling != 'tokenization':
            self.lm_head = nn.Identity()

    def init_processor_tokenizer(self, config):
        max_token = 144
        min_pixels = max_token * 28 * 28
        max_pixels = max_token * 28 * 28
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
        self.tokenizer = self.processor.tokenizer
        random_tensor = torch.rand(3, max_token, max_token).to(dtype=torch.float32)
        self.grid_thw = torch.tensor(self.processor.image_processor(random_tensor)['image_grid_thw'])

    def text_backbone(self):
        return self

    def hidden_dim(self):
        return self.model.config.hidden_size

    def vision_backbone(self):
        return self.visual

    def image_seq_len(self):
        return self.grid_thw.prod() // 4

    def image_start_token(self):
        return self.model.config.vision_start_token_id

    def image_end_token(self):
        return self.model.config.vision_end_token_id

    def process_image(self, image):
        return torch.tensor(self.processor.image_processor(image)['pixel_values'])

    def image_embeds(self, pixel_values):
        B = pixel_values.shape[0]
        return self.visual(pixel_values, grid_thw=self.grid_thw.expand(B, -1))
    
    # QWEN2 style prompt
    def system_prompt(self):
        return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"

AutoConfig.register("Qwen2VLVLA", Qwen2VLVLAConfig)
AutoModel.register(Qwen2VLVLAConfig, Qwen2VLVLA)
