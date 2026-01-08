import os
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import handle_seed

WORKFLOW_RECIPE_PATH = "Kandinsky_txt2video_recipe.yaml"

MODEL_CONFIG = {
    "5s": {
        "unet_name": "kandinsky5lite_t2v_sft_5s.safetensors",
        "length": 121
    },
    "10s": {
        "unet_name": "kandinsky5lite_t2v_sft_10s.safetensors",
        "length": 241
    }
}

ASPECT_RATIO_PRESETS = {
    "16:9 (Widescreen)": (840, 472),
    "3:2 (Landscape)": (768, 512),
    "4:3 (Classic TV)": (736, 552),
    "1:1 (Square)": (624, 624),
    "9:16 (Vertical)": (472, 840),
    "2:3 (Portrait)": (512, 768),
    "3:4 (Classic Portrait)": (552, 736),
}

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()

    duration = local_ui_values.get('duration', '5s')
    config = MODEL_CONFIG[duration]
    
    local_ui_values['unet_name'] = config['unet_name']
    local_ui_values['video_length'] = config['length']
    
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Widescreen)") 
    width, height = ASPECT_RATIO_PRESETS.get(selected_ratio, (840, 472))
    local_ui_values['width'] = width
    local_ui_values['height'] = height

    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}"

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None