import os
import math
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_image, save_temp_audio

WORKFLOW_RECIPE_PATH = "wan2_2_sound2video_recipe.yaml"

ASPECT_RATIO_PRESETS = {
    "16:9 (Landscape)": (1280, 720), "9:16 (Portrait)": (720, 1280),
    "1:1 (Square)": (960, 960), "4:3 (Classic TV)": (1088, 816),
    "3:4 (Classic Portrait)": (816, 1088),
    "3:2 (Photography)": (1152, 768),
    "2:3 (Photography Portrait)": (768, 1152),
}
CHUNK_LENGTH = 77
FPS = 16

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    ref_image_pil = local_ui_values.get('ref_image')
    original_audio_path = local_ui_values.get('audio_file')
    if ref_image_pil is None: raise ValueError("Reference image is required.")
    if original_audio_path is None: raise ValueError("Audio file is required.")

    local_ui_values['ref_image'] = save_temp_image(ref_image_pil)
    local_ui_values['audio_file'] = save_temp_audio(original_audio_path)
    
    selected_ratio = local_ui_values.get('aspect_ratio')
    width, height = ASPECT_RATIO_PRESETS[selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    current_seed = handle_seed(seed_override if seed_override is not None else local_ui_values.get('seed', -1))
    local_ui_values['seed'] = current_seed
    
    filename_prefix = get_filename_prefix()
    local_ui_values['filename_prefix'] = filename_prefix

    mode = local_ui_values.get('mode', 'Preview (77 frames)')
    if mode == 'Generate Full Video':
        audio_meta = get_media_metadata(original_audio_path, is_video=True)
        duration = audio_meta.get('duration', 0)
        if duration <= 0:
            raise ValueError("Could not determine audio duration or audio is empty.")
        
        total_frames = int(duration * FPS)
        num_chunks = math.ceil(total_frames / CHUNK_LENGTH) if total_frames > 0 else 1

        local_ui_values['s2v_chain'] = {
            "num_chunks": num_chunks,
            "seed": current_seed,
        }
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None