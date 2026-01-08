import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed
from core.input_processors import process_lora_inputs

RECIPE_PREVIEW = "wan2_1_wanmove_preview_recipe.yaml"
RECIPE_GENERATE = "wan2_1_wanmove_recipe.yaml"

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

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    mode = local_ui_values.get('mode', 'Preview Tracks')
    recipe_path = RECIPE_PREVIEW if mode == 'Preview Tracks' else RECIPE_GENERATE
    
    input_image = local_ui_values.get('input_image')
    if input_image is None:
        raise ValueError("Input image is required.")
    local_ui_values['input_image'] = save_temp_image(input_image)
    
    resolution = local_ui_values.get('resolution', '720p')
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Landscape)") 
    width, height = RESOLUTION_PRESETS.get(resolution, {}).get(selected_ratio, (1280, 720))
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}_WanMove"

    segment_count = local_ui_values.get('track_segment_count', 1)
    
    param_keys = ['start_x', 'start_y', 'end_x', 'end_y', 'bezier', 'mid_x', 'mid_y', 'interpolation', 'num_tracks', 'track_spread']
    
    list_keys_map = {
        'start_x': 'all_start_xs',
        'start_y': 'all_start_ys',
        'end_x': 'all_end_xs',
        'end_y': 'all_end_ys',
        'bezier': 'all_beziers',
        'mid_x': 'all_mid_xs',
        'mid_y': 'all_mid_ys',
        'interpolation': 'all_interpolations',
        'num_tracks': 'all_num_tracks',
        'track_spread': 'all_track_spreads',
    }
    all_params = {key: local_ui_values.get(list_key, []) for key, list_key in list_keys_map.items()}

    if segment_count > 0:
        for key in param_keys:
            if all_params[key]:
                local_ui_values[key] = all_params[key][0]

    wan_move_chain = []
    if segment_count > 1:
        for i in range(1, segment_count):
            segment_data = {}
            for key in param_keys:
                if i < len(all_params[key]):
                    segment_data[key] = all_params[key][i]
            if segment_data:
                wan_move_chain.append(segment_data)

    local_ui_values['wan_move_chain'] = wan_move_chain

    if mode == 'Generate Video':
        num_frames_per_segment = local_ui_values.get('num_frames', 81)
        total_frames = num_frames_per_segment
        local_ui_values['video_length'] = total_frames
        
        local_ui_values['positive_prompt'] = local_ui_values.get('positive_prompt', '')
        local_ui_values['negative_prompt'] = local_ui_values.get('negative_prompt', '')
        local_ui_values['strength'] = 1.0
        
        resolution = local_ui_values.get('resolution', '720p')
        if resolution == '480p':
            local_ui_values['lora_name'] = "Wan21_I2V_14B_480P_lightx2v_cfg_step_distill_lora_rank64.safetensors"
        else:
            local_ui_values['lora_name'] = "Wan21_I2V_14B_720P_lightx2v_cfg_step_distill_lora_rank64.safetensors"

        seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
        local_ui_values['seed'] = handle_seed(seed)
        
        local_ui_values['loras_model_only'] = process_lora_inputs(ui_values, 'wan2_1_wanmove_lora')
    else:
        if 'seed' in local_ui_values:
            del local_ui_values['seed']
        local_ui_values['use_easy_cache'] = False


    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None