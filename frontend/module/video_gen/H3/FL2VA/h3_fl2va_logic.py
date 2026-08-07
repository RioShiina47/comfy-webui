import os
import math
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_image
from core.input_processors import process_lora_inputs

WORKFLOW_RECIPE_PATH = "h3_fl2va_recipe.yaml"

RESOLUTION_PRESETS = {
    "1080p": {
        "16:9 (Landscape)": (1920, 1088),
        "9:16 (Portrait)": (1088, 1920),
        "1:1 (Square)": (1280, 1280),
        "4:3 (Classic TV)": (1440, 1088),
        "3:4 (Classic Portrait)": (1088, 1440),
    },
    "720p": {
        "16:9 (Landscape)": (1344, 768),
        "9:16 (Portrait)": (768, 1344),
        "1:1 (Square)": (960, 960),
        "4:3 (Classic TV)": (1024, 768),
        "3:4 (Classic Portrait)": (768, 1024),
    },
    "480p": {
        "16:9 (Landscape)": (864, 480),
        "9:16 (Portrait)": (480, 864),
        "1:1 (Square)": (640, 640),
        "4:3 (Classic TV)": (640, 480),
        "3:4 (Classic Portrait)": (480, 640),
    }
}

ASPECT_RATIO_PRESETS = RESOLUTION_PRESETS["720p"]

def calculate_h3_frame_length(duration_seconds: float) -> int:
    """
    Converts duration (seconds) at 24fps to a valid frame length
    snapped up to the model's 17-frame-per-block (17k+5) grid.
    Grid sequence: 5, 22, 39, 56, 73, 90, 107, 124, 141...
    """
    raw_frames = int(round(duration_seconds * 24))
    if raw_frames <= 5:
        return 5
    return 5 + 17 * math.ceil((raw_frames - 5) / 17)

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    width = int(local_ui_values.get('width') or 0)
    height = int(local_ui_values.get('height') or 0)
    
    if width <= 0 or height <= 0:
        resolution = local_ui_values.get('resolution', '720p')
        selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Landscape)")
        width, height = RESOLUTION_PRESETS.get(resolution, {}).get(selected_ratio, (1344, 768))
        
    local_ui_values['width'] = width
    local_ui_values['height'] = height

    first_frame_img = local_ui_values.get('first_frame')
    if first_frame_img is not None:
        local_ui_values['first_frame_loader_class'] = "LoadImage"
        local_ui_values['first_frame_scale_class'] = "ImageScale"
        local_ui_values['first_frame_image'] = save_temp_image(first_frame_img)
    else:
        local_ui_values['first_frame_loader_class'] = None
        local_ui_values['first_frame_scale_class'] = None

    last_frame_img = local_ui_values.get('last_frame')
    if last_frame_img is not None:
        local_ui_values['last_frame_loader_class'] = "LoadImage"
        local_ui_values['last_frame_scale_class'] = "ImageScale"
        local_ui_values['last_frame_image'] = save_temp_image(last_frame_img)
    else:
        local_ui_values['last_frame_loader_class'] = None
        local_ui_values['last_frame_scale_class'] = None

    duration = float(local_ui_values.get('duration', 3.0))
    local_ui_values['length'] = calculate_h3_frame_length(duration)
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)
    
    filename_prefix = get_filename_prefix()
    local_ui_values['filename_prefix'] = f"video/{filename_prefix}"

    local_ui_values['loras'] = process_lora_inputs(ui_values, 'h3_fl2va_lora')

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None
