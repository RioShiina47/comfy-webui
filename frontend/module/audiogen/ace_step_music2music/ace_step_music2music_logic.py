import random
import os
import shutil
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix

WORKFLOW_RECIPE_PATH = "ace_step_music2music_recipe.yaml"

def save_temp_audio_file(audio_path):
    if not audio_path:
        return None
    
    filename = f"temp_m2m_input_{random.randint(1000, 9999)}{os.path.splitext(audio_path)[1]}"
    save_path = os.path.join(COMFYUI_INPUT_PATH, filename)
    shutil.copy(audio_path, save_path)
    print(f"Saved temporary audio file to: {save_path}")
    return filename

def process_inputs(params, seed_override=None):
    local_params = params.copy()
    
    input_audio_path = local_params.get('input_audio')
    if not input_audio_path:
        raise ValueError("Input audio file is required.")
        
    local_params['input_audio'] = save_temp_audio_file(input_audio_path)
    
    similarity = local_params.get('similarity', 0.7)
    local_params['denoise'] = 1.0 - similarity
    
    if seed_override is not None:
        local_params['seed'] = seed_override
    else:
        seed = int(local_params.get('seed', -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        local_params['seed'] = seed
        
    local_params['filename_prefix'] = f"audio/{get_filename_prefix()}"
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_params)
    
    return workflow, None