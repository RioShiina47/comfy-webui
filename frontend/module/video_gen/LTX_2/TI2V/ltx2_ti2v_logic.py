import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_image
from core.input_processors import process_lora_inputs

WORKFLOW_RECIPE_T2V = "ltx2_t2v_recipe.yaml"
WORKFLOW_RECIPE_I2V = "ltx2_i2v_recipe.yaml"

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

    if start_image_pil:
        recipe_path = WORKFLOW_RECIPE_I2V
        local_ui_values['start_image'] = save_temp_image(start_image_pil)
    else:
        recipe_path = WORKFLOW_RECIPE_T2V
    
    resolution = local_ui_values.get('resolution', '720p')
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Widescreen)") 
    width, height = RESOLUTION_PRESETS[resolution][selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['video_length'] = int(local_ui_values.get('video_length', 121))
    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}"

    if local_ui_values.get('use_easy_cache', False):
        local_ui_values['use_easy_cache'] = [{}]

    local_ui_values['loras_model_only'] = process_lora_inputs(ui_values, 'ti2v_lora')

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None