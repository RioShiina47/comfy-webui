import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_image

WORKFLOW_RECIPE_PATH = "Kandinsky_img2video_recipe.yaml"

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()

    start_image = local_ui_values.get('start_image')
    if start_image is None:
        raise ValueError("Start image is required.")
    local_ui_values['start_image'] = save_temp_image(start_image)

    if 'width' not in local_ui_values or 'height' not in local_ui_values:
        raise ValueError("Width and height must be provided by the UI layer.")
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}"

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None