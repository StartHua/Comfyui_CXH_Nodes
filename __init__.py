# from .image_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
# from .LLM_nodes import NODE_CLASS_MAPPINGS,NODE_DISPLAY_NAME_MAPPINGS

from .image_nodes import Net_image
from .image_nodes import SubtractMask
from .LLM_nodes import Phi_3_vision

NODE_CLASS_MAPPINGS = {
    "Net_image":Net_image,
    "SubtractMask":SubtractMask,
    "Phi_3_vision":Phi_3_vision
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Net_image":"Net_image",
    "SubtractMask":"SubtractMask",
    "Phi_3_vision":"Phi_3_vision"
}