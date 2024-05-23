import os
import torch
import folder_paths
import json

from transformers import AutoModelForCausalLM, AutoProcessor

from .lib.ximg import *
from .lib.xmodel import *

class Phi_3_vision:
    def __init__(self):
        self.model_checkpoint = None
        self.model = None
        self.processor = None
        self.model_cache = None
        self.tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING",{"default": 'Describe the image below', "multiline": True}),
                "model": (["Phi-3-vision-128k-instruct","phi-3-vision-128k-instruct-quantized"],),
                "temperature": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 500, "min": 100, "max": 2000, "step": 500}),
                "cache": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "CXH/LLM"

    def inference(self,image,prompt,model,temperature,max_new_tokens,cache):

        self.cache = cache

        #下载本地
        if model == "Phi-3-vision-128k-instruct":
            model_id = f"microsoft/{model}"
            model_checkpoint = download_hg_model(model_id,"microsoft")
        else:
            model_id = f"dwb2023/{model}"
            model_checkpoint = download_hg_model(model_id,"microsoft")    
        
        torch.random.manual_seed(0)
        
        #加载模型
        if self.model == None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_checkpoint,
                device_map="cuda",
                trust_remote_code=True, 
                torch_dtype="auto"
            )

        if self.processor ==None:
            self.processor = AutoProcessor.from_pretrained(model_checkpoint, trust_remote_code=True) 

        # 图片转换
        image = tensor2pil(image)

        messages = [ 
            {"role": "user", "content": "<|image_1|>\n"+prompt}
        ] 

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda:0") 

        generation_args = { 
            "max_new_tokens": max_new_tokens, 
            "temperature": temperature, 
            "do_sample": False, 
        }

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        if cache ==False:
            self.model = None
            self.processor = None

        return (response,)
        

# NODE_CLASS_MAPPINGS = {
#     "Phi_3_vision": Phi_3_vision,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Phi_3_vision": "Phi_3_vision",
# }