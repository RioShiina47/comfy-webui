import os
import shutil
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "invsr_recipe.yaml"

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    input_img = local_ui_values.get('input_image')
    if input_img is None:
        raise ValueError("Please upload an input image.")
        
    local_ui_values['input_image_filename'] = save_temp_image(input_img)

    seed = int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)
    
    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None