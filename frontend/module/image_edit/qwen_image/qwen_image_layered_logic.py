import os

from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "qwen_image_layered_recipe.yaml"
PREFIX = "qwen_layered"

def process_inputs(ui_values, seed_override=None):
    vals = {k.replace(f'{PREFIX}_', ''): v for k, v in ui_values.items() if isinstance(k, str) and k.startswith(PREFIX)}

    input_img = vals.get('input_image')
    if input_img is None:
        raise ValueError("Please upload an input image.")

    vals['input_image'] = save_temp_image(input_img)

    if not vals.get('positive_prompt', '').strip():
        vals['positive_prompt'] = ' '
    if not vals.get('negative_prompt', '').strip():
        vals['negative_prompt'] = ' '
    mode = vals.get('mode', 'Fast')
    if mode == "Fast":
        vals['steps'] = 20
        vals['cfg'] = 2.5
    else:
        vals['steps'] = 50
        vals['cfg'] = 4.0

    seed = seed_override if seed_override is not None else vals.get('seed', -1)
    vals['seed'] = handle_seed(seed)
    
    vals['filename_prefix'] = get_filename_prefix()

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    workflow = assembler.assemble(vals)
    
    return workflow, None