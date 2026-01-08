import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed
from core.input_processors import process_lora_inputs

WORKFLOW_RECIPE_PATH = "wan2_1_txt2video_recipe.yaml"
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
    
    resolution = local_ui_values.get('resolution', '720p')
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Landscape)") 
    width, height = RESOLUTION_PRESETS.get(resolution, {}).get(selected_ratio, (1280, 720))
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['video_length'] = int(local_ui_values.get('video_length', 81))
    
    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}"

    local_ui_values['loras_model_only'] = process_lora_inputs(ui_values, 'wan2_1_txt2video_lora')

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None