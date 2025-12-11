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

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()

    duration = local_ui_values.get('duration', '5s')
    config = MODEL_CONFIG[duration]
    
    local_ui_values['unet_name'] = config['unet_name']
    local_ui_values['video_length'] = config['length']
    
    if 'width' not in local_ui_values or 'height' not in local_ui_values:
        raise ValueError("Width and height must be provided by the UI layer.")

    seed = seed_override if seed_override is not None else int(local_ui_values.get('seed', -1))
    local_ui_values['seed'] = handle_seed(seed)

    local_ui_values['filename_prefix'] = f"video/{get_filename_prefix()}"

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None