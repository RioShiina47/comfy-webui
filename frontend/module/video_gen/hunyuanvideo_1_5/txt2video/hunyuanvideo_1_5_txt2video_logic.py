import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed

RESOLUTION_PRESETS = {
    "480p": {
        "16:9 (Landscape)": (848, 480), "9:16 (Portrait)": (480, 848), "1:1 (Square)": (672, 672),
        "4:3 (Classic TV)": (768, 576), "3:4 (Classic Portrait)": (576, 768),
        "3:2 (Photography)": (816, 544), "2:3 (Photography Portrait)": (544, 816),
    },
    "720p": {
        "16:9 (Landscape)": (1280, 720), "9:16 (Portrait)": (720, 1280), "1:1 (Square)": (960, 960),
        "4:3 (Classic TV)": (1088, 816), "3:4 (Classic Portrait)": (816, 1088),
        "3:2 (Photography)": (1152, 768), "2:3 (Photography Portrait)": (768, 1152),
    },
    "1080p": {
        "16:9 (Landscape)": (1920, 1080), "9:16 (Portrait)": (1080, 1920), "1:1 (Square)": (1440, 1440),
        "4:3 (Classic TV)": (1632, 1224), "3:4 (Classic Portrait)": (1224, 1632),
        "3:2 (Photography)": (1728, 1152), "2:3 (Photography Portrait)": (1152, 1728),
    }
}

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    resolution_key = local_ui_values.get('resolution', '480p')
    aspect_ratio_key = local_ui_values.get('aspect_ratio', "16:9 (Landscape)")

    if resolution_key == '480p':
        recipe_path = "hunyuanvideo_1_5_txt2video_480p_recipe.yaml"
    elif resolution_key == '720p':
        recipe_path = "hunyuanvideo_1_5_txt2video_720p_recipe.yaml"
    else:
        recipe_path = "hunyuanvideo_1_5_txt2video_1080p_recipe.yaml"

    base_width, base_height = RESOLUTION_PRESETS["480p"][aspect_ratio_key]
    local_ui_values['width'] = base_width
    local_ui_values['height'] = base_height

    if resolution_key in ["720p", "1080p"]:
        width_720, height_720 = RESOLUTION_PRESETS["720p"][aspect_ratio_key]
        local_ui_values['width_720p'] = width_720
        local_ui_values['height_720p'] = height_720

    if resolution_key == "1080p":
        width_1080, height_1080 = RESOLUTION_PRESETS["1080p"][aspect_ratio_key]
        local_ui_values['width_1080p'] = width_1080
        local_ui_values['height_1080p'] = height_1080
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['video_length'] = int(local_ui_values.get('video_length', 121))
    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}"

    if local_ui_values.get('use_easy_cache', False):
        local_ui_values['use_easy_cache'] = [{}]

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None