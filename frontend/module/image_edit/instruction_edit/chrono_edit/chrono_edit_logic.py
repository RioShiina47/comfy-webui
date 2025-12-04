import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "chrono_edit_recipe.yaml"

def process_inputs(params: dict, seed_override=None):
    local_params = params.copy()
    
    main_img = local_params.get('start_image')
    if main_img is None:
        raise ValueError("Please upload an image to edit.")

    local_params['start_image'] = save_temp_image(main_img)

    seed_val = seed_override if seed_override is not None else local_params.get('seed', -1)
    local_params['seed'] = handle_seed(seed_val)

    local_params['filename_prefix'] = get_filename_prefix()

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_params)
    
    return final_workflow, None