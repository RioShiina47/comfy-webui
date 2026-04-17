import random
import os
import tempfile

from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix

WORKFLOW_RECIPE_PATH = "newbie_recipe.yaml"

def process_inputs(ui_values):
    """
    Processes UI inputs and assembles the ComfyUI workflow to generate an XML prompt.
    """
    local_ui_values = ui_values.copy()
    
    character_chain_items = []
    real_char_number = 1
    for i in range(1, 5):
        is_enabled = ui_values.get(f'char{i}_enable', False)
        if is_enabled:
            char_data = {
                "character_number": real_char_number,
                "name": ui_values.get(f'char{i}_name', ""),
                "gender": ui_values.get(f'char{i}_gender', "1girl"),
                "appearance": ui_values.get(f'char{i}_appearance', ""),
                "clothing": ui_values.get(f'char{i}_clothing', ""),
                "expression": ui_values.get(f'char{i}_expression', ""),
                "action": ui_values.get(f'char{i}_action', ""),
                "interaction": ui_values.get(f'char{i}_interaction', ""),
                "position": ui_values.get(f'char{i}_position', "")
            }
            character_chain_items.append(char_data)
            real_char_number += 1
        
    local_ui_values['character_chain'] = character_chain_items

    temp_dir = tempfile.gettempdir()
    unique_filename = f"newbie_xml_{get_filename_prefix()}_{random.randint(1000, 9999)}.txt"
    expected_output_path = os.path.join(temp_dir, unique_filename).replace("\\", "/")
    
    local_ui_values['output_file_path'] = os.path.dirname(expected_output_path)
    local_ui_values['filename_prefix'] = os.path.splitext(os.path.basename(expected_output_path))[0]

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, {"expected_text_file_path": expected_output_path}