import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "longcat_image_edit_recipe.yaml"
TURBO_WORKFLOW_RECIPE_PATH = "longcat_image_edit_turbo_recipe.yaml"

ASPECT_RATIO_PRESETS = {
    "1:1 (Square)": (1024, 1024), 
    "16:9 (Landscape)": (1344, 768), 
    "9:16 (Portrait)": (768, 1344),
    "4:3 (Classic)": (1152, 896), 
    "3:4 (Classic Portrait)": (896, 1152),
    "3:2 (Photography)": (1216, 832),
    "2:3 (Photography Portrait)": (832, 1216)
}

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    main_img = local_ui_values.get('input_image')
    if main_img is None:
        raise ValueError("Please upload an image to edit.")

    local_ui_values['input_image'] = save_temp_image(main_img)

    selected_ratio = local_ui_values.get('aspect_ratio', "1:1 (Square)")
    width, height = ASPECT_RATIO_PRESETS.get(selected_ratio, (1024, 1024))
    local_ui_values['width'] = width
    local_ui_values['height'] = height

    if not local_ui_values.get('positive_prompt'):
        local_ui_values['positive_prompt'] = ""
    if not local_ui_values.get('negative_prompt'):
        local_ui_values['negative_prompt'] = ""

    seed = seed_override if seed_override is not None else local_ui_values.get('seed', -1)
    local_ui_values['seed'] = handle_seed(seed)
        
    local_ui_values['filename_prefix'] = get_filename_prefix()

    is_turbo = local_ui_values.get('use_turbo', False)
    recipe_path = TURBO_WORKFLOW_RECIPE_PATH if is_turbo else WORKFLOW_RECIPE_PATH

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None