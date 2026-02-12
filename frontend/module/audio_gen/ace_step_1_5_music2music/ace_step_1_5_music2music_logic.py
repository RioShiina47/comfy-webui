import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_audio
from core.media_utils import get_media_metadata

WORKFLOW_RECIPE_PATH = "ace_step_1_5_music2music_recipe.yaml"

def process_inputs(params, seed_override=None):
    local_params = params.copy()
    
    input_audio_path = local_params.get('input_audio')
    if not input_audio_path:
        raise ValueError("Input audio file is required.")
        
    local_params['input_audio'] = save_temp_audio(input_audio_path)
    
    try:
        meta = get_media_metadata(local_params['input_audio'])
        duration = meta.get('duration', 0)
        if duration <= 0:
            print("Warning: Could not detect audio duration, defaulting to 120s.")
            duration = 120
    except Exception as e:
        print(f"Error calculating audio duration: {e}, defaulting to 120s.")
        duration = 120
        
    local_params['duration'] = int(duration)
    
    similarity = local_params.get('similarity', 0.7)
    local_params['denoise'] = 1.0 - similarity

    seed = seed_override if seed_override is not None else local_params.get('seed', -1)
    local_params['seed'] = handle_seed(seed)
    
    local_params['filename_prefix'] = f"audio/{get_filename_prefix()}"

    clips_selection = local_params.get('clips', '0.6b+1.7b')
    if clips_selection == '0.6b+4b':
        local_params['clip_name1'] = "qwen_0.6b_ace15.safetensors"
        local_params['clip_name2'] = "qwen_4b_ace15.safetensors"
    else:
        local_params['clip_name1'] = "qwen_0.6b_ace15.safetensors"
        local_params['clip_name2'] = "qwen_1.7b_ace15.safetensors"

    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_params)
    
    return workflow, None