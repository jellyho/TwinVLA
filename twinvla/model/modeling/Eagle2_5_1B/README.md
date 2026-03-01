---
license: cc-by-nc-4.0
pipeline_tag: image-text-to-text
library_name: transformers
base_model:
  - google/paligemma-3b-mix-448
  - Qwen/Qwen2.5-0.5B-Instruct
  - google/siglip-so400m-patch14-384
base_model_relation: merge
language:
  - multilingual
tags:
  - eagle
  - VLM
---


# Eagle-2


[\[📂 GitHub\]](https://github.com/NVlabs/EAGLE)   [\[📜 Eagle2 Tech Report\]](http://arxiv.org/abs/2501.14818)
[\[🤗 HF Demo\]](https://huggingface.co/spaces/nvidia/Eagle2-Demo)  

# News:
- We update the model arch to `eagle_2_5_vl` to support  `generate` feature.

## Introduction

We are thrilled to release our latest Eagle2 series Vision-Language Model. Open-source Vision-Language Models (VLMs) have made significant strides in narrowing the gap with proprietary models. However, critical details about data strategies and implementation are often missing, limiting reproducibility and innovation. In this project, we focus on VLM post-training from a data-centric perspective, sharing insights into building effective data strategies from scratch. By combining these strategies with robust training recipes and model design, we introduce Eagle2, a family of performant VLMs. Our work aims to empower the open-source community to develop competitive VLMs with transparent processes.


In this repo, we are open-sourcing Eagle2-1B, a compact and efficient model designed for scenarios requiring fast inference and minimal computational resources, without compromising essential performance








## Model Zoo
We provide the following models:

| model name         | LLM  | Vision  | Max Length| HF Link|
| ----------- | ------- |---------|-|-|
| Eagle2-1B | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |  Siglip    | 16K| [🤗 link](https://huggingface.co/NVIDIA/Eagle2-1B)|
| Eagle2-2B | [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |  Siglip    | 16K| [🤗 link](https://huggingface.co/NVIDIA/Eagle2-2B)|
| Eagle2-9B | [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |  Siglip+ConvNext    | 16K| [🤗 link](https://huggingface.co/NVIDIA/Eagle2-9B)|

## Benchmark Results
|          Benchmark           | LLaVa-One-Vision-0.5B | InternVL2-1B | InternVL2.5-1B |Qwen2-VL-2B| Eagle2-1B|
| :--------------------------: | :------------------: | :----------------: | :----------: |:----------: |:----------: |  
|    DocVQA<sub>test</sub>     |         70.0         |        81.7        |     84.8     |90.1|81.8|
|    ChartQA<sub>test</sub>    |          61.4         |        72.9        |     75.9     |73.0|77.0|
|    InfoVQA<sub>test</sub>    |          41.8           |        50.9        |     56.0     |65.5|54.8|
|    TextVQA<sub>val</sub>     |         -         |        70.0       |     72.0     |79.7|76.6|
|           OCRBench           |         565          |        754         |     785      |809|767|
|      MME<sub>sum</sub>       |        1438.0     |       1794.4      |    1950.5   |  1872.0| 1790.2|
|         RealWorldQA          |        55.6     |        50.3       |    57.5     |62.6|55.4|
|     AI2D<sub>test</sub>      |         57.1         |        64.1        |     69.3    | 74.7 |70.9|
|      MMMU<sub>val</sub>      |          31.4       |    36.7     | 40.9  |41.1|38.8|
| MMVet<sub>GPT-4-Turbo</sub>  |         32.2       |        32.7       |    48.8    | 49.5|40.9|             HallBench<sub>avg</sub>    |         27.9      |        34.0       |     39.0     |**41.7**|35.3
| MathVista<sub>testmini</sub> |         33.8         |        37.7        |     43.2     |43.0|45.3|
| MMstar |             37.7    |       45.7      |     50.1|48.0|48.5|



## Quick Start



We provide a [inference script](./demo.py) to help you quickly start using the model. We support different input types: 
- pure text input
- single image input
- multiple image input
- video input

###  Install the dependencies

```bash
pip install transformers
pip install flash-attn
```


### single image

```python
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
model = AutoModel.from_pretrained("nvidia/Eagle2-1B",trust_remote_code=True, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("nvidia/Eagle2-1B", trust_remote_code=True, use_fast=True)
processor.tokenizer.padding_side = "left"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://www.ilankelman.org/stopsigns/australia.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text_list = [processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)]
image_inputs, video_inputs = processor.process_vision_info(messages)
inputs = processor(text = text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
inputs = inputs.to("cuda")
model = model.to("cuda")
generated_ids = model.generate(**inputs, max_new_tokens=1024)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

### stream generation

```python
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import torch

from transformers import TextIteratorStreamer
import threading


model = AutoModel.from_pretrained("nvidia/Eagle2-1B",trust_remote_code=True, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("nvidia/Eagle2-1B", trust_remote_code=True, use_fast=True)
processor = AutoProcessor.from_pretrained("nvidia/Eagle2-1B", trust_remote_code=True, use_fast=True)
processor.tokenizer.padding_side = "left"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://www.ilankelman.org/stopsigns/australia.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text_list = [processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)]
image_inputs, video_inputs = processor.process_vision_info(messages)
inputs = processor(text = text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
inputs = inputs.to("cuda")
model = model.to("cuda")

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

generation_kwargs = dict(
    **inputs,
    streamer=streamer,
    max_new_tokens=1024,
    do_sample=True,
    top_p=0.95,
    temperature=0.8
)
thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()


for new_text in streamer:
    print(new_text, end="", flush=True)
```

### multiple-images 

```python
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
model = AutoModel.from_pretrained("nvidia/Eagle2-1B",trust_remote_code=True, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("nvidia/Eagle2-1B", trust_remote_code=True, use_fast=True)
processor.tokenizer.padding_side = "left"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://www.ilankelman.org/stopsigns/australia.jpg",
            },
            {
                "type": "image",
                "image": "https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-vert-500x200-2c50-d@2x.png",
            },
            {"type": "text", "text": "Describe these two images."},
        ],
    }
]

text_list = [processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)]
image_inputs, video_inputs = processor.process_vision_info(messages)
inputs = processor(text = text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
inputs = inputs.to("cuda")
model = model.to("cuda")
generated_ids = model.generate(**inputs, max_new_tokens=1024)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

### single video 

```python

from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
model = AutoModel.from_pretrained("nvidia/Eagle2-1B",trust_remote_code=True, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("nvidia/Eagle2-1B", trust_remote_code=True, use_fast=True)
processor.tokenizer.padding_side = "left"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "../Eagle2-8B/space_woaudio.mp4",
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

text_list = [processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)]
image_inputs, video_inputs, video_kwargs = processor.process_vision_info(messages, return_video_kwargs=True)

inputs = processor(text = text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True, videos_kwargs=video_kwargs)
inputs = inputs.to("cuda")
model = model.to("cuda")
generated_ids = model.generate(**inputs, max_new_tokens=1024)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

```

### multieple videos

```python
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
model = AutoModel.from_pretrained("nvidia/Eagle2-1B",trust_remote_code=True, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("nvidia/Eagle2-1B", trust_remote_code=True, use_fast=True)
processor.tokenizer.padding_side = "left"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "../Eagle2-8B/space_woaudio.mp4",
                "nframes": 10,
            },
            {
                "type": "video",
                "video": "../Eagle2-8B/video_ocr.mp4",
                "nframes": 10,
            },
            {"type": "text", "text": "Describe these two videos respectively."},
        ],
    }
]

text_list = [processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)]
image_inputs, video_inputs, video_kwargs = processor.process_vision_info(messages, return_video_kwargs=True)
inputs = processor(text = text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True, videos_kwargs=video_kwargs)
inputs = inputs.to("cuda")
model = model.to("cuda")
generated_ids = model.generate(**inputs, max_new_tokens=1024)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

### batch inference

```python
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
model = AutoModel.from_pretrained("nvidia/Eagle2-1B",trust_remote_code=True, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained("nvidia/Eagle2-1B", trust_remote_code=True, use_fast=True)
processor.tokenizer.padding_side = "left"

messages1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://www.ilankelman.org/stopsigns/australia.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

messages2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-vert-500x200-2c50-d@2x.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text_list = [processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
) for messages in [messages1, messages2]]
image_inputs, video_inputs = processor.process_vision_info([messages1, messages2])
inputs = processor(text = text_list, images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True)
inputs = inputs.to("cuda")
model = model.to("cuda")
generated_ids = model.generate(**inputs, max_new_tokens=1024)
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```


## TODO
- [ ] Support vLLM Inference
- [ ] Provide AWQ Quantization Weights
- [ ] Provide fine-tuning scripts


## License/Terms of Use
- The code is released under the Apache 2.0 license as found in the [LICENSE](https://huggingface.co/NVEagle/Eagle-X5-13B-Chat/blob/main/LICENSE) file.
- The pretrained model weights are released under the [Creative Commons Attribution: Non-Commercial 4.0 International](https://spdx.org/licenses/CC-BY-NC-4.0) <br>
- The service is a research preview intended for non-commercial use only, and is subject to the following licenses and terms:
  - Model License of Qwen2.5-0.5B-Instruct: [Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE)
  - Model License of PaliGemma: [Gemma license](https://ai.google.dev/gemma/terms)



## Citation

## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.    

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
 