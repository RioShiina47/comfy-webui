import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH, COMFYUI_OUTPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "hunyuan3d2_mv23d_recipe.yaml"

def process_inputs(params, seed_override=None):
    local_params = params.copy()
    
    front_img = local_params.get('input_front_image')
    back_img = local_params.get('input_back_image')
    left_img = local_params.get('input_left_image')
    
    if front_img is None or back_img is None or left_img is None:
        raise ValueError("All three view images (front, back, left) are required.")

    local_params['input_front_image'] = save_temp_image(front_img)
    local_params['input_back_image'] = save_temp_image(back_img)
    local_params['input_left_image'] = save_temp_image(left_img)

    seed = seed_override if seed_override is not None else int(local_params.get('seed', -1))
    local_params['seed'] = handle_seed(seed)
    
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