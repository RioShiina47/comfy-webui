import random
import os
import tempfile

from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed

WORKFLOW_RECIPE_PATH = "ernie_image_prompt_enhancer_recipe.yaml"

ASPECT_RATIO_MAP = {
    "1:1 (Square)": [1024, 1024],
    "16:9 (Landscape)": [1344, 768],
    "9:16 (Portrait)": [768, 1344],
    "4:3 (Classic)": [1152, 896],
    "3:4 (Classic Portrait)": [896, 1152],
    "3:2 (Photography)": [1216, 832],
    "2:3 (Photography Portrait)": [832, 1216]
}

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    prompt = local_ui_values.get('prompt', '').replace('"', '\\"')
    
    aspect_ratio_choice = local_ui_values.get('aspect_ratio', "1:1 (Square)")
    width, height = ASPECT_RATIO_MAP.get(aspect_ratio_choice, [1024, 1024])
    
    system_prompt = local_ui_values.get('system_prompt', "")
    
    formatted_prompt = f'{system_prompt}[INST]{{"prompt": "{prompt}", "width": {width}, "height": {height}}}[/INST]'
    local_ui_values['formatted_prompt'] = formatted_prompt
    
    seed = int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)
    
    temp_dir = tempfile.gettempdir()
    unique_filename = f"ernie_pe_{get_filename_prefix()}_{random.randint(1000, 9999)}.txt"
    expected_output_path = os.path.join(temp_dir, unique_filename).replace("\\", "/")
    
    local_ui_values['output_file_path'] = os.path.dirname(expected_output_path)
    local_ui_values['filename_prefix'] = os.path.splitext(os.path.basename(expected_output_path))[0]

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, {"expected_text_file_path": expected_output_path}