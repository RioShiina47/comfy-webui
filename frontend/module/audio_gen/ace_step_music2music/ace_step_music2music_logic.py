import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_audio

WORKFLOW_RECIPE_PATH = "ace_step_music2music_recipe.yaml"

def process_inputs(params, seed_override=None):
    local_params = params.copy()
    
    input_audio_path = local_params.get('input_audio')
    if not input_audio_path:
        raise ValueError("Input audio file is required.")
        
    local_params['input_audio'] = save_temp_audio(input_audio_path)
    
    similarity = local_params.get('similarity', 0.7)
    local_params['denoise'] = 1.0 - similarity
    
    seed = seed_override if seed_override is not None else local_params.get('seed', -1)
    local_params['seed'] = handle_seed(seed)
        
    local_params['filename_prefix'] = f"audio/{get_filename_prefix()}"
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_params)
    
    return workflow, None