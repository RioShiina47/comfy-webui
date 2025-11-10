import random
import os
import shutil
from PIL import Image
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix

WORKFLOW_RECIPE_PATH = "wan2_2_img2video_recipe.yaml"

def save_temp_image(image_pil: Image.Image) -> str:
    if image_pil is None: return None
    comfyui_input_dir = COMFYUI_INPUT_PATH
    temp_filename = f"temp_i2v_input_{random.randint(1000, 9999)}.png"
    save_path = os.path.join(comfyui_input_dir, temp_filename)
    image_pil.save(save_path, "PNG")
    print(f"Saved temporary input image to: {save_path}")
    return temp_filename

def process_inputs(params, seed_override=None):
    local_params = params.copy()
    start_image_pil = local_params.get('start_image')
    if start_image_pil is None:
        raise ValueError("Input image is required for Img2Video generation.")

    local_params['start_image'] = save_temp_image(start_image_pil.copy())
    
    if seed_override is not None:
        local_params['seed'] = seed_override
    else:
        seed = int(local_params.get('seed', -1))
        if seed == -1:
            seed = random.randint(0, 999999999999999)
        local_params['seed'] = seed
        
    local_params['video_length'] = int(local_params.get('video_length', 81))
    local_params['filename_prefix'] = get_filename_prefix()
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_params)
    
    return final_workflow, None