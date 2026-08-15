import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed, save_temp_image
from core.input_processors import process_lora_inputs

WORKFLOW_RECIPE_I2V = "ltx2_5_i2v_recipe.yaml"
WORKFLOW_RECIPE_I2V_2X = "ltx2_5_i2v_2x_recipe.yaml"
WORKFLOW_RECIPE_I2V_3X = "ltx2_5_i2v_3x_recipe.yaml"

RESOLUTION_PRESETS = {
    "1080P": {
        "16:9 (Widescreen)": (1920, 1088),
        "9:16 (Vertical)": (1088, 1920),
        "1:1 (Square)": (1440, 1440),
        "4:3 (Classic TV)": (1440, 1088),
        "3:4 (Classic Portrait)": (1088, 1440),
        "3:2 (Photography)": (1632, 1088),
        "2:3 (Photography Portrait)": (1088, 1632),
    },
    "768P": {
        "16:9 (Widescreen)": (1344, 768),
        "9:16 (Vertical)": (768, 1344),
        "1:1 (Square)": (1024, 1024),
        "4:3 (Classic TV)": (1152, 864),
        "3:4 (Classic Portrait)": (864, 1152),
        "3:2 (Photography)": (1152, 768),
        "2:3 (Photography Portrait)": (768, 1152),
    },
    "480P": {
        "16:9 (Widescreen)": (864, 480),
        "9:16 (Vertical)": (480, 864),
        "1:1 (Square)": (640, 640),
        "4:3 (Classic TV)": (640, 480),
        "3:4 (Classic Portrait)": (480, 640),
        "3:2 (Photography)": (736, 480),
        "2:3 (Photography Portrait)": (480, 736),
    }
}

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    start_image_pil = local_ui_values.get('start_image')
    
    if start_image_pil is None:
        raise ValueError("Start Image is required for Image-to-Video generation.")

    use_spatial = local_ui_values.get('use_spatial_upscaler', False)
    use_temporal = local_ui_values.get('use_temporal_upscaler', False)
    
    if use_spatial and use_temporal:
        recipe_path = WORKFLOW_RECIPE_I2V_3X
    elif use_spatial:
        recipe_path = WORKFLOW_RECIPE_I2V_2X
        local_ui_values['upscaler_model_name'] = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    elif use_temporal:
        recipe_path = WORKFLOW_RECIPE_I2V_2X
        local_ui_values['upscaler_model_name'] = "ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
    else:
        recipe_path = WORKFLOW_RECIPE_I2V

    local_ui_values['start_image'] = save_temp_image(start_image_pil)

    resolution = local_ui_values.get('resolution', '768P')
    preset_dict = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS.get(resolution.upper(), RESOLUTION_PRESETS["768P"]))
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Widescreen)") 
    width, height = preset_dict.get(selected_ratio, (1344, 768))
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    fps_raw = local_ui_values.get('fps', "24fps")
    if isinstance(fps_raw, str):
        fps = int(fps_raw.replace('fps', '').replace('FPS', '').strip())
    else:
        fps = int(fps_raw)

    duration = float(local_ui_values.get('duration', 5.0))
    video_length = int(round(duration * fps)) + 1
    local_ui_values['video_length'] = video_length
    local_ui_values['frame_rate'] = fps
    local_ui_values['strength'] = 0.7
    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}"

    local_ui_values['loras'] = process_lora_inputs(ui_values, 'ltx2_5_i2v_lora')

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, dynamic_values=local_ui_values, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None
