import random
import os
from PIL import Image
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH, COMFYUI_OUTPUT_PATH
from core.workflow_utils import get_filename_prefix

WORKFLOW_RECIPE_PATH = "hunyuan3d2_img23d_recipe.yaml"

def save_temp_image(img: Image.Image) -> str:
    if not isinstance(img, Image.Image):
        return None
    filename = f"temp_hunyuan3d_input_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    print(f"Saved temporary image to {filepath}")
    return filename

def process_inputs(params, seed_override=None):
    local_params = params.copy()
    
    input_img = local_params.get('input_image')
    if input_img is None:
        raise ValueError("Input image is required.")

    local_params['input_image'] = save_temp_image(input_img)

    if seed_override is not None:
        local_params['seed'] = seed_override
    else:
        seed = int(local_params.get('seed', -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        local_params['seed'] = seed
    
    unique_prefix = get_filename_prefix()
    
    shape_relative_path = os.path.join("Hunyuan3D-2", f"{unique_prefix}_shape.glb").replace("\\", "/")
    textured_relative_path = os.path.join("Hunyuan3D-2", f"{unique_prefix}_textured.glb").replace("\\", "/")
    
    local_params['shape_save_path'] = shape_relative_path
    local_params['textured_save_path'] = textured_relative_path
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_params)
    
    expected_files = {
        "shape": os.path.join(COMFYUI_OUTPUT_PATH, shape_relative_path.replace('/', os.sep)),
        "textured": os.path.join(COMFYUI_OUTPUT_PATH, textured_relative_path.replace('/', os.sep))
    }
    
    return workflow, {"expected_files": expected_files}