import os
from PIL import Image
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_image, save_temp_video

WORKFLOW_RECIPE_PATH = "wan2_2_fun_control_recipe.yaml"

ASPECT_RATIO_PRESETS = {
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

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    ref_image_pil = local_ui_values.get('ref_image')
    control_video_path = local_ui_values.get('control_video')

    if ref_image_pil is None: raise ValueError("Reference image is required.")
    if control_video_path is None: raise ValueError("Control video is required.")

    resolution = local_ui_values.get('resolution', '720p')
    selected_ratio = local_ui_values.get('aspect_ratio')
    
    metadata = get_media_metadata(control_video_path, is_video=True)
    width, height = metadata['width'], metadata['height']
    if width == 0 or height == 0:
        width, height = ASPECT_RATIO_PRESETS[resolution][selected_ratio]
        print(f"Warning: Could not auto-detect video dimensions. Falling back to selected preset: {resolution} {selected_ratio} ({width}x{height}).")

    local_ui_values['width'] = width
    local_ui_values['height'] = height
    local_ui_values['ref_image'] = save_temp_image(ref_image_pil.copy())
    local_ui_values['control_video'] = save_temp_video(control_video_path)
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['video_length'] = int(local_ui_values.get('video_length', 81))
    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None