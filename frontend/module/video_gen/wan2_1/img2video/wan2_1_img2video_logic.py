import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_image
from core.input_processors import process_lora_inputs

WORKFLOW_RECIPE_PATH = "wan2_1_img2video_recipe.yaml"

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
    "720p": "wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors",
    "480p": "wan2.1_i2v_480p_scaled_fp8_e4m3_lightx2v_4step_comfyui.safetensors"
}

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    start_image_pil = local_ui_values.get('start_image')
    if start_image_pil is None:
        raise ValueError("Start image is required.")
    local_ui_values['start_image'] = save_temp_image(start_image_pil)
    
    resolution = local_ui_values.get('resolution', '720p')
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Landscape)")
    
    width, height = RESOLUTION_PRESETS[resolution][selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    local_ui_values['unet_name'] = MODEL_MAPPING[resolution]
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['video_length'] = int(local_ui_values.get('video_length', 81))
    
    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}"

    local_ui_values['loras_model_only'] = process_lora_inputs(ui_values, 'wan2_1_img2video_lora')

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None