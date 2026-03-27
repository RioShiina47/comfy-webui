import random
import os
import tempfile

from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    input_img = local_ui_values.get('input_image')
    is_image = input_img is not None
    use_abliterated = local_ui_values.get('use_abliterated', False)
    
    if is_image:
        recipe_path = "ti2p_abliterated_recipe.yaml" if use_abliterated else "ti2p_recipe.yaml"
        local_ui_values['input_image'] = save_temp_image(input_img)
    else:
        recipe_path = "t2p_abliterated_recipe.yaml" if use_abliterated else "t2p_recipe.yaml"
    
    seed = int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)
    
    temp_dir = tempfile.gettempdir()
    unique_filename = f"ltx2_prompt_{get_filename_prefix()}_{random.randint(1000, 9999)}.txt"
    expected_output_path = os.path.join(temp_dir, unique_filename).replace("\\", "/")
    
    local_ui_values['output_file_path'] = os.path.dirname(expected_output_path)
    local_ui_values['filename_prefix'] = os.path.splitext(os.path.basename(expected_output_path))[0]

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, {"expected_text_file_path": expected_output_path}