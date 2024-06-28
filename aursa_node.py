
import os
import torch
from PIL import Image
import folder_paths
from .lib.ximg import *
from .lib.xmodel import *

import os
import json

from torchvision import transforms
from aura_sr import UnetUpsampler,tile_image,merge_tiles

class AursaCXH:

    def __init__(self):
        self.upsampler = None
        self.input_image_size = None
        self.checkpoint = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("out",)
    FUNCTION = "gen"
    OUTPUT_NODE = False
    CATEGORY = "CXH/Scale"

    @torch.no_grad()
    def upscale_4x(self, image: Image.Image, max_batch_size=8) -> Image.Image:
        tensor_transform = transforms.ToTensor()
        device = self.upsampler.device

        image_tensor = tensor_transform(image).unsqueeze(0)
        _, _, h, w = image_tensor.shape
        pad_h = (self.input_image_size - h % self.input_image_size) % self.input_image_size
        pad_w = (self.input_image_size - w % self.input_image_size) % self.input_image_size

        # Pad the image
        image_tensor = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
        tiles, h_chunks, w_chunks = tile_image(image_tensor, self.input_image_size)

        # Batch processing of tiles
        num_tiles = len(tiles)
        batches = [tiles[i:i + max_batch_size] for i in range(0, num_tiles, max_batch_size)]
        reconstructed_tiles = []

        for batch in batches:
            model_input = torch.stack(batch).to(device)
            generator_output = self.upsampler(
                lowres_image=model_input,
                noise=torch.randn(model_input.shape[0], 128, device=device)
            )
            reconstructed_tiles.extend(list(generator_output.clamp_(0, 1).detach().cpu()))

        merged_tensor = merge_tiles(reconstructed_tiles, h_chunks, w_chunks, self.input_image_size * 4)
        unpadded = merged_tensor[:, :h * 4, :w * 4]

        to_pil = transforms.ToPILImage()
        return to_pil(unpadded)

    def gen(self, images):
        results = []

        from safetensors.torch import load_file

        # 检查下载模型
        model_id = "fal/AuraSR"
        model_checkpoint = os.path.join(folder_paths.models_dir, "fal", os.path.basename(model_id))
        config_path = os.path.join(model_checkpoint,"config.json")
        mode_safetensors =  os.path.join(model_checkpoint,"model.safetensors")
        if not os.path.exists(config_path) or not os.path.exists(mode_safetensors):
            download_hg_model("fal/AuraSR","fal")

        with open(config_path, 'r') as file:
            config = json.load(file)

        if self.input_image_size ==None:
            self.input_image_size = config["input_image_size"]

        if self.upsampler == None:
            self.upsampler = UnetUpsampler(**config).to("cuda")

        if self.checkpoint == None:    
            checkpoint = load_file(mode_safetensors)
            self.upsampler.load_state_dict(checkpoint, strict=True)

        for IMAGE in images:
            image = tensor2pil(IMAGE)
            upscaled_image = self.upscale_4x(image)
            image2 = pil2tensor(upscaled_image)
            results.append(image2)
            

        # 将结果列表中的张量连接在一起
        return (torch.cat(results, dim=0),)

