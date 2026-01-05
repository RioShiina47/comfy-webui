import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "chrono_edit_recipe.yaml"

ASPECT_RATIO_PRESETS = {
    "16:9 (Landscape)": (1280, 720),
    "9:16 (Portrait)": (720, 1280),
    "1:1 (Square)": (960, 960),
    "4:3 (Classic TV)": (1088, 816),
    "3:4 (Classic Portrait)": (816, 1088),
    "3:2 (Photography)": (1152, 768),
    "2:3 (Photography Portrait)": (768, 1152),
}

def process_inputs(params: dict, seed_override=None):
    local_params = params.copy()
    
    main_img = local_params.get('start_image')
    if main_img is None:
        raise ValueError("Please upload an image to edit.")

    local_params['start_image'] = save_temp_image(main_img)

    selected_ratio = local_params.get('aspect_ratio')
    width, height = ASPECT_RATIO_PRESETS.get(selected_ratio, (960, 960))
    local_params['width'] = width
    local_params['height'] = height

    seed_val = seed_override if seed_override is not None else local_params.get('seed', -1)
    local_params['seed'] = handle_seed(seed_val)

    local_params['filename_prefix'] = get_filename_prefix()

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_params)
    
    return final_workflow, None