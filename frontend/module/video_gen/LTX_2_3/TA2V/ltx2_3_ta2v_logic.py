import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_audio

WORKFLOW_RECIPE_PATH = "ltx2_3_ta2v_recipe.yaml"
WORKFLOW_RECIPE_2X = "ltx2_3_ta2v_2x_recipe.yaml"
WORKFLOW_RECIPE_3X = "ltx2_3_ta2v_3x_recipe.yaml"

RESOLUTION_PRESETS = {
    "720p": {
        "16:9 (Widescreen)": (1024, 576),
        "9:16 (Vertical)": (576, 1024),
        "1:1 (Square)": (768, 768),
        "4:3 (Classic TV)": (896, 672),
        "3:4 (Classic Portrait)": (672, 896),
    },
    "480p": {
        "16:9 (Widescreen)": (832, 480),
        "9:16 (Vertical)": (480, 832),
        "1:1 (Square)": (640, 640),
        "4:3 (Classic TV)": (640, 480),
        "3:4 (Classic Portrait)": (480, 640),
    }
}

FPS = 24

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    audio_path = local_ui_values.get('audio_file')
    if audio_path is None:
        raise ValueError("Audio file is required.")

    try:
        metadata = get_media_metadata(audio_path, is_video=True)
        duration_seconds = metadata.get('duration', 0)
        if duration_seconds <= 0:
            raise ValueError("Could not determine audio duration or audio is empty.")
    except Exception as e:
        raise ValueError(f"Error getting audio metadata: {e}")

    local_ui_values['duration_in_seconds'] = duration_seconds
    local_ui_values['video_length'] = int(duration_seconds * FPS)
    
    local_ui_values['audio_file'] = save_temp_audio(audio_path)
    
    use_spatial = local_ui_values.get('use_spatial_upscaler', False)
    use_temporal = local_ui_values.get('use_temporal_upscaler', False)

    if use_spatial and use_temporal:
        recipe_path = WORKFLOW_RECIPE_3X
    elif use_spatial:
        recipe_path = WORKFLOW_RECIPE_2X
        local_ui_values['upscaler_model_name'] = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    elif use_temporal:
        recipe_path = WORKFLOW_RECIPE_2X
        local_ui_values['upscaler_model_name'] = "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"
    else:
        recipe_path = WORKFLOW_RECIPE_PATH
        
    resolution = local_ui_values.get('resolution', '480p')
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Widescreen)") 
    width, height = RESOLUTION_PRESETS[resolution][selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}_ltx2.3_ta2v"
    
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, dynamic_values=local_ui_values, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None