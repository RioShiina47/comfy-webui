import os
import math
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "flux2_dev_recipe.yaml"
MAX_REF_IMAGES = 10

ASPECT_RATIO_PRESETS = {
    "1:1 (Square)": (1024, 1024), 
    "3:2 (Photography)": (1248, 832),
    "2:3 (Photography Portrait)": (832, 1248),
    "16:9 (Landscape)": (1344, 768), 
    "9:16 (Portrait)": (768, 1344),
    "4:3 (Classic)": (1152, 896), 
    "3:4 (Classic Portrait)": (896, 1152),
}

def update_resolution(ratio_key, megapixels_str):
    base_w, base_h = ASPECT_RATIO_PRESETS.get(ratio_key, (1024, 1024))
    
    mp_map = {"1MP": 1.0, "2MP": 2.0, "4MP": 4.0}
    mp_multiplier = mp_map.get(megapixels_str, 1.0)
    
    target_pixels = mp_multiplier * 1024 * 1024
    
    current_pixels = base_w * base_h
    scale_factor = math.sqrt(target_pixels / current_pixels)
    
    new_width = int(round((base_w * scale_factor) / 8) * 8)
    new_height = int(round((base_h * scale_factor) / 8) * 8)
    
    return new_width, new_height

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    if not local_ui_values.get('positive_prompt'):
        raise ValueError("Prompt is required.")

    selected_ratio = local_ui_values.get('aspect_ratio', "1:1 (Square)")
    megapixels_str = local_ui_values.get('megapixels', '1MP')
    width, height = update_resolution(selected_ratio, megapixels_str)
    local_ui_values['width'] = width
    local_ui_values['height'] = height

    all_images = []
    if local_ui_values.get('ref_image_inputs') and isinstance(local_ui_values['ref_image_inputs'], list):
        for img in local_ui_values['ref_image_inputs']:
            if img is not None:
                all_images.append(img)
    
    if all_images:
        image_filenames = [save_temp_image(img) for i, img in enumerate(all_images)]
        local_ui_values['reference_chain'] = image_filenames
    else:
        local_ui_values['reference_chain'] = []

    seed = seed_override if seed_override is not None else local_ui_values.get('seed', -1)
    local_ui_values['seed'] = handle_seed(seed)
        
    local_ui_values['filename_prefix'] = get_filename_prefix()

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None