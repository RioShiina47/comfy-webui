import random
import os
import tempfile
from PIL import Image

from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix

WORKFLOW_RECIPE_PATH = "qwen_vl_recipe.yaml"

def save_temp_image(img: Image.Image) -> str:
    if not isinstance(img, Image.Image):
        return None
    filename = f"temp_qwen_vl_input_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    print(f"Saved temporary image to {filepath}")
    return os.path.basename(filepath)

def process_inputs(params):
    local_params = params.copy()
    
    input_img = local_params.get('input_image')
    if input_img is None:
        raise ValueError("Input image is required.")
        
    local_params['input_image'] = save_temp_image(input_img)
    
    model_mode = local_params.get('model_mode', 'Instruct')
    if model_mode == "Thinking":
        local_params['model_name'] = "Qwen3-VL-4B-Thinking-FP8"
    else:
        local_params['model_name'] = "Qwen3-VL-4B-Instruct-FP8"
    
    seed = int(local_params.get('seed', -1))
    if seed == -1:
        local_params['seed'] = random.randint(0, 2**32 - 1)
    
    temp_dir = tempfile.gettempdir()
    unique_filename = f"qwen_vl_desc_{get_filename_prefix()}_{random.randint(1000, 9999)}.txt"
    expected_output_path = os.path.join(temp_dir, unique_filename).replace("\\", "/")
    
    local_params['output_file_path'] = os.path.dirname(expected_output_path)
    local_params['filename_prefix'] = os.path.splitext(os.path.basename(expected_output_path))[0]

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    workflow = assembler.assemble(local_params)
    
    return workflow, {"expected_text_file_path": expected_output_path}