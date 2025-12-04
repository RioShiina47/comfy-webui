import os
import numpy as np
from PIL import Image

from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.utils import create_mask_from_layer, save_temp_image, handle_seed
from core.workflow_utils import get_filename_prefix

WORKFLOW_RECIPE_PATH = "flux_dev_onereward_inpaint_recipe.yaml"

def process_inputs(ui_values, seed_override=None):
    vals = {k.replace(f'{PREFIX}_', ''): v for k, v in ui_values.items() if isinstance(k, str) and k.startswith(PREFIX)}
    
    img_dict = vals.get('input_image_dict')
    mask_img = create_mask_from_layer(img_dict)
    if not img_dict or img_dict.get('background') is None or mask_img is None:
        raise ValueError("Input image and a drawn mask are required.")
    
    background_img = img_dict['background'].convert("RGBA")
    _, _, _, mask_alpha = mask_img.split()
    background_alpha_np = np.array(background_img.split()[-1])
    mask_alpha_np = np.array(mask_alpha)
    new_alpha_np = np.minimum(background_alpha_np, 255 - mask_alpha_np)
    new_alpha_pil = Image.fromarray(new_alpha_np, mode='L')
    
    r, g, b, _ = background_img.split()
    composite_image = Image.merge('RGBA', [r, g, b, new_alpha_pil])

    vals['input_image'] = save_temp_image(composite_image)

    if not vals.get('positive_prompt') or not vals.get('positive_prompt').strip():
        vals['positive_prompt'] = ' '
        
    if not vals.get('negative_prompt') or not vals.get('negative_prompt').strip():
        vals['negative_prompt'] = ' '

    if vals.get('remove_object_mode', False):
        vals['lora_chain'] = [{
            "lora_name": "removal_timestep_alpha-2-1740.safetensors",
            "strength_model": 1.0
        }]
    else:
        vals['lora_chain'] = []

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    
    seed = seed_override if seed_override is not None else vals.get('seed', -1)
    vals['seed'] = handle_seed(seed)
    vals['filename_prefix'] = get_filename_prefix()

    workflow = assembler.assemble(vals)
    return workflow, {"extra_pnginfo": {"workflow": ""}}