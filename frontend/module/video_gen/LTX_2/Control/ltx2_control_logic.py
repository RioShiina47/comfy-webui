import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_image, save_temp_video
from core.input_processors import process_lora_inputs

WORKFLOW_RECIPE_PATH = "ltx2_control_recipe.yaml"

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

CONTROL_LORA_MAPPING = {
    "Canny": "ltx-2-19b-ic-lora-canny-control.safetensors",
    "Depth": "ltx-2-19b-ic-lora-depth-control.safetensors",
    "Pose": "ltx-2-19b-ic-lora-pose-control.safetensors"
}

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()

    control_video = local_ui_values.get('control_video')
    if control_video is None:
        raise ValueError("Control Video is required.")
    local_ui_values['control_video'] = save_temp_video(control_video)

    start_image_pil = local_ui_values.get('start_image')
    if start_image_pil is None:
        raise ValueError("Start Image is required.")
    local_ui_values['start_image'] = save_temp_image(start_image_pil)

    resolution = local_ui_values.get('resolution', '480p')
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Widescreen)") 
    width, height = RESOLUTION_PRESETS[resolution].get(selected_ratio, (832, 480))
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['video_length'] = int(local_ui_values.get('video_length', 121))
    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}"

    control_type = local_ui_values.get('control_type', 'Canny')
    local_ui_values['control_lora_name'] = CONTROL_LORA_MAPPING.get(control_type, CONTROL_LORA_MAPPING['Canny'])

    local_ui_values['loras_model_only'] = process_lora_inputs(ui_values, 'ltx2_control_lora')

    module_path = os.path.dirname(os.path.abspath(__file__))
    
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, dynamic_values=local_ui_values, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None