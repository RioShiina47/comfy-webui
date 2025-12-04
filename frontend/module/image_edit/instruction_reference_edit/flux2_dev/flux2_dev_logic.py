import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "flux2_dev_recipe.yaml"
MAX_REF_IMAGES = 10

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    if not local_ui_values.get('positive_prompt'):
        raise ValueError("Prompt is required.")

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