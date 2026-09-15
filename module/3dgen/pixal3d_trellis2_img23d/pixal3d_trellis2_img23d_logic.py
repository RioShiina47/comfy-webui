import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_OUTPUT_PATH
from core.utils import get_filename_prefix, save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "pixal3d_trellis2_img23d_recipe.yaml"


def process_inputs(params, seed_override=None):
    local_params = params.copy()
    
    input_img = local_params.get('input_image')
    if input_img is None:
        raise ValueError("Input image is required.")

    local_params['input_image'] = save_temp_image(input_img)

    seed = seed_override if seed_override is not None else int(local_params.get('seed', -1))
    master_seed = handle_seed(seed)
    
    local_params['structure_seed'] = master_seed
    local_params['shape_seed'] = (master_seed + 1) % (2**32)
    local_params['upsample_seed'] = (master_seed + 2) % (2**32)
    local_params['texture_seed'] = (master_seed + 3) % (2**32)
    
    unique_prefix = get_filename_prefix()
    shape_prefix = f"3d/Trellis2_{unique_prefix}_shape"
    textured_prefix = f"3d/Trellis2_{unique_prefix}_textured"
    
    local_params['shape_filename_prefix'] = shape_prefix
    local_params['textured_filename_prefix'] = textured_prefix
    
    local_params['target_face_count'] = int(local_params.get('target_face_count', 700000))
    local_params['texture_resolution'] = int(local_params.get('texture_resolution', 4096))
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_params)
    
    expected_files = {
        "shape": os.path.join(COMFYUI_OUTPUT_PATH, f"{shape_prefix}_00001.glb".replace('/', os.sep)),
        "textured": os.path.join(COMFYUI_OUTPUT_PATH, f"{textured_prefix}_00001.glb".replace('/', os.sep)),
        "prefix_shape": shape_prefix,
        "prefix_textured": textured_prefix
    }
    
    return workflow, {"expected_files": expected_files}
