import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed
from core.input_processors import process_lora_inputs

WORKFLOW_RECIPE_PATH = "wan2_2_txt2video_recipe.yaml"

RESOLUTION_PRESETS = {
    "720p": {
        "16:9 (Landscape)": (1280, 720),
        "9:16 (Portrait)": (720, 1280),
        "1:1 (Square)": (960, 960),
        "4:3 (Classic TV)": (1088, 816),
        "3:4 (Classic Portrait)": (816, 1088),
        "3:2 (Photography)": (1152, 768),
        "2:3 (Photography Portrait)": (768, 1152),
    },
    "480p": {
        "16:9 (Landscape)": (848, 480),
        "9:16 (Portrait)": (480, 848),
        "1:1 (Square)": (640, 640),
        "4:3 (Classic TV)": (640, 480),
        "3:4 (Classic Portrait)": (480, 640),
        "3:2 (Photography)": (720, 480),
        "2:3 (Photography Portrait)": (480, 720),
    }
}

def process_inputs(params, seed_override=None):
    local_params = params.copy()
    
    resolution = local_params.get('resolution', '720p')
    selected_ratio = local_params.get('aspect_ratio', "16:9 (Landscape)") 
    width, height = RESOLUTION_PRESETS[resolution][selected_ratio]
    local_params['width'] = width
    local_params['height'] = height
    
    seed = seed_override if seed_override is not None else int(local_params.get('seed', -1))
    local_params['seed'] = handle_seed(seed)

    local_params['video_length'] = int(local_params.get('video_length', 81))
    local_params['filename_prefix'] = get_filename_prefix()
    
    local_params['high_noise_loras_model_only'] = process_lora_inputs(params, 'high_noise')
    local_params['low_noise_loras_model_only'] = process_lora_inputs(params, 'low_noise')
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_params)
    
    return final_workflow, None