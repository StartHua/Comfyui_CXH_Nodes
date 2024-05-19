
import os
import torch
from PIL import Image
import folder_paths
import latent_preview
import node_helpers
import numpy as np
import safetensors.torch

from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo

from .lib.ximg import *

class Net_image:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "IsNet": ("BOOLEAN", {"default": False}),
                "url":   ("STRING", {"multiline": True, "default": ""},),
                "image":(sorted(files), {"image_upload": True}), #image_upload显示 
            }
        }

    CATEGORY = "CXH/images"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, IsNet,url,image):
        output_images = []
        output_masks = []
        
        #网络图
        if IsNet ==True:
           img =  img_from_url(url)
        else:
            image_path = folder_paths.get_annotated_filepath(image)
            img = node_helpers.pillow(Image.open, image_path)

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)


NODE_CLASS_MAPPINGS = {
    "Net_image":Net_image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Net_image":"Net_image"
}