import os
import numpy as np
from PIL import Image

from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.utils import create_mask_from_layer, save_temp_image, handle_seed
from core.workflow_utils import get_filename_prefix

WORKFLOW_RECIPE_PATH = "qwen_inpaint_recipe.yaml"
PREFIX = "qwen_inpaint"

def process_inputs(ui_values, seed_override=None):
    vals = {k.replace(f'{PREFIX}_', ''): v for k, v in ui_values.items() if isinstance(k, str) and k.startswith(PREFIX)}
    
    img_dict = vals.get('input_image_dict')
    mask_img = create_mask_from_layer(img_dict)
    if not img_dict or img_dict.get('background') is None or mask_img is None:
        raise ValueError("Input image and a drawn mask are required.")
    
    background_img = img_dict['background'].convert("RGBA")
    
    mask_alpha = mask_img.split()[-1]
    inverted_alpha = Image.fromarray(255 - np.array(mask_alpha), mode='L')
    
    r, g, b, _ = background_img.split()
    composite_image = Image.merge('RGBA', [r, g, b, inverted_alpha])

    vals['input_image'] = save_temp_image(composite_image)

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    
    seed = seed_override if seed_override is not None else vals.get('seed', -1)
    vals['seed'] = handle_seed(seed)
    vals['filename_prefix'] = get_filename_prefix()

    workflow = assembler.assemble(vals)
    return workflow, {"extra_pnginfo": {"workflow": ""}}