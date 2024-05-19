
import os
import torch
from PIL import Image
import folder_paths
from ..lib.ximg import *

class node_example:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}), #image_upload显示
                "images": ("IMAGE",),
                "prompt":   ("STRING", {"multiline": True, "default": "", "forceInput": True},), # forceInput 让节点直接显示在连接处
                "model": (["Phi-3-mini-4k-instruct", "Phi-3-mini-128k-instruct"],),
                "temperature": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask": ("MASK",),
                "mask_threshold":("INT", {"default": 250, "min": 0, "max": 255, "step": 1, "display": "slider"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "seed":("INT"),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("out",)
    FUNCTION = "gen"
    OUTPUT_NODE = False
    CATEGORY = "CXH/example"

    def gen(self, images):
        results = []

        for image in images:
            image2 = img_from_url(image)
            image2 = pil2tensor(image2)
            results.append(image2)
            

        # 将结果列表中的张量连接在一起
        return (torch.cat(results, dim=0),)



NODE_CLASS_MAPPINGS = {
    "node_example":"node_example"
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "node_example":"node_example"
}