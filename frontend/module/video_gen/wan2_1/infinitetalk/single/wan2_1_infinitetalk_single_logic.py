import os
import math
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_image, save_temp_audio
from core.media_utils import get_media_metadata
from core.input_processors import process_lora_inputs

WORKFLOW_RECIPE_PATH = "wan2_1_infinitetalk_recipe.yaml"

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

MODEL_MAPPING = {
    "480p": "wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors",
    "720p": "wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors"
}

CHUNK_LENGTH = 81
FPS = 25

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    ref_image_pil = local_ui_values.get('ref_image')
    if ref_image_pil is None:
        raise ValueError("Reference image is required.")
    local_ui_values['ref_image'] = save_temp_image(ref_image_pil)
    
    audio_file_path = local_ui_values.get('audio_file')
    if audio_file_path is None:
        raise ValueError("Audio file is required.")
    local_ui_values['audio_file'] = save_temp_audio(audio_file_path)

    resolution = local_ui_values.get('resolution', '480p')
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Landscape)") 
    width, height = RESOLUTION_PRESETS[resolution].get(selected_ratio, (848, 480))
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    local_ui_values['unet_name'] = MODEL_MAPPING.get(resolution, MODEL_MAPPING["480p"])

    current_seed = handle_seed(seed_override if seed_override is not None else int(local_ui_values.get('seed', -1)))
    local_ui_values['seed'] = current_seed
    
    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}"

    local_ui_values['loras_model_only'] = process_lora_inputs(ui_values, 'wan2_1_infinitetalk_single_lora')

    mode = local_ui_values.get('mode', f'Preview ({CHUNK_LENGTH} frames)')
    if mode == 'Generate Full Video':
        audio_meta = get_media_metadata(audio_file_path, is_video=True)
        duration = audio_meta.get('duration', 0)
        if duration <= 0:
            raise ValueError("Could not determine audio duration or audio is empty.")
        
        total_frames = int(duration * FPS)
        num_chunks = math.ceil(total_frames / CHUNK_LENGTH) if total_frames > 0 else 1

        local_ui_values['infinitetalk_chain'] = {
            "num_chunks": num_chunks,
            "seed": current_seed,
        }

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None