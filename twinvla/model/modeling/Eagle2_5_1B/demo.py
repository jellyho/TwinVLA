from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import torch

from transformers import TextIteratorStreamer
import threading


model = AutoModel.from_pretrained("/home/zhidingy/workspace/eagle-next/internvl_chat/work_dirs/release/test/Eagle2-1B",trust_remote_code=True, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("/home/zhidingy/workspace/eagle-next/internvl_chat/work_dirs/release/test/Eagle2-1B", trust_remote_code=True, use_fast=True)
processor = AutoProcessor.from_pretrained("/home/zhidingy/workspace/eagle-next/internvl_chat/work_dirs/release/test/Eagle2-1B", trust_remote_code=True, use_fast=True)
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
# generated_ids = model.generate(**inputs, max_new_tokens=1024)
# output_text = processor.batch_decode(
#     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)

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
