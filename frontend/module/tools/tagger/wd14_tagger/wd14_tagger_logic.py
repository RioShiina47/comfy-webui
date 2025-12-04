import random
import os
import tempfile

from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image

WORKFLOW_RECIPE_PATH = "wd14_tagger_recipe.yaml"

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    input_img = local_ui_values.get('input_image')
    if input_img is None:
        raise ValueError("Please upload an input image.")
        
    local_ui_values['input_image'] = save_temp_image(input_img)
    
    temp_dir = tempfile.gettempdir()
    unique_filename = f"wd14_tags_{get_filename_prefix()}_{random.randint(1000, 9999)}.txt"
    expected_output_path = os.path.join(temp_dir, unique_filename)
    
    expected_output_path = expected_output_path.replace("\\", "/")
    
    local_ui_values['output_file_path'] = os.path.dirname(expected_output_path)
    local_ui_values['filename_prefix'] = os.path.basename(expected_output_path).replace('.txt', '')

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, {"expected_text_file_path": expected_output_path}