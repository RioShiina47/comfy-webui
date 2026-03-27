import os
from PIL import Image
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "ltx2_3_flf2v_recipe.yaml"
WORKFLOW_RECIPE_2X = "ltx2_3_flf2v_2x_recipe.yaml"
WORKFLOW_RECIPE_3X = "ltx2_3_flf2v_3x_recipe.yaml"

RESOLUTION_PRESETS = {
    "720p": {
        "16:9 (Widescreen)": (1024, 576),
        "9:16 (Vertical)": (576, 1024),
        "1:1 (Square)": (768, 768),
        "4:3 (Classic TV)": (896, 672),
        "3:4 (Classic Portrait)": (672, 896),
    },
    "480p": {
        "16:9 (Widescreen)": (832, 480),
        "9:16 (Vertical)": (480, 832),
        "1:1 (Square)": (640, 640),
        "4:3 (Classic TV)": (640, 480),
        "3:4 (Classic Portrait)": (480, 640),
    }
}

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    start_image_pil = local_ui_values.get('start_image')
    if start_image_pil is None:
        raise ValueError("Start image is required.")
    local_ui_values['start_image'] = save_temp_image(start_image_pil)
    
    end_image_pil = local_ui_values.get('end_image')
    if end_image_pil is None:
        raise ValueError("End image is required.")
    local_ui_values['end_image'] = save_temp_image(end_image_pil)
    
    use_spatial = local_ui_values.get('use_spatial_upscaler', False)
    use_temporal = local_ui_values.get('use_temporal_upscaler', False)

    if use_spatial and use_temporal:
        recipe_path = WORKFLOW_RECIPE_3X
    elif use_spatial:
        recipe_path = WORKFLOW_RECIPE_2X
        local_ui_values['upscaler_model_name'] = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    elif use_temporal:
        recipe_path = WORKFLOW_RECIPE_2X
        local_ui_values['upscaler_model_name'] = "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"
    else:
        recipe_path = WORKFLOW_RECIPE_PATH
    
    resolution = local_ui_values.get('resolution', '480p')
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Widescreen)") 
    width, height = RESOLUTION_PRESETS[resolution][selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['video_length'] = int(local_ui_values.get('video_length', 121))
    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}_ltx2.3_flf2v"
    
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, dynamic_values=local_ui_values, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None