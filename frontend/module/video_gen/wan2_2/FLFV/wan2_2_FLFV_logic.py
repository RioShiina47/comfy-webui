import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "wan2_2_FLFV_recipe.yaml"

ASPECT_RATIO_PRESETS = {
    "16:9 (Landscape)": (1280, 720),
    "9:16 (Portrait)": (720, 1280),
    "1:1 (Square)": (960, 960),
    "4:3 (Classic TV)": (1088, 816),
    "3:4 (Classic Portrait)": (816, 1088),
    "3:2 (Photography)": (1152, 768),
    "2:3 (Photography Portrait)": (768, 1152),
}

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    start_image_pil = local_ui_values.get('start_image')
    end_image_pil = local_ui_values.get('end_image')
    if start_image_pil is None: raise ValueError("Start image is required.")
    if end_image_pil is None: raise ValueError("End image is required.")
    
    selected_ratio = local_ui_values.get('aspect_ratio')
    width, height = ASPECT_RATIO_PRESETS[selected_ratio]

    local_ui_values['width'] = width
    local_ui_values['height'] = height
    local_ui_values['start_image'] = save_temp_image(start_image_pil)
    local_ui_values['end_image'] = save_temp_image(end_image_pil)
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)
    
    local_ui_values['video_length'] = int(local_ui_values.get('video_length', 81))
    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None