import os

from core.workflow_assembler import WorkflowAssembler
from core.utils import save_temp_image, handle_seed
from core.utils import get_filename_prefix
from core.input_processors import process_lora_inputs

WORKFLOW_RECIPE_PATH = "qwen_outpaint_recipe.yaml"
PREFIX = "qwen_outpaint"

def process_inputs(ui_values, seed_override=None):
    vals = {k.replace(f'{PREFIX}_', ''): v for k, v in ui_values.items() if isinstance(k, str) and k.startswith(PREFIX)}
    
    input_img = vals.get('input_image')
    if input_img is None:
        raise ValueError("Input image is required for Outpainting.")
    
    vals['input_image'] = save_temp_image(input_img)
    
    vals['lora_chain'] = process_lora_inputs(ui_values, prefix=PREFIX)

    model_version = vals.get('model_version', 'Qwen-Image-2512')
    if model_version == 'Qwen-Image-2512':
        vals['unet_name'] = "qwen_image_2512_fp8_e4m3fn.safetensors"
        vals['lora_name'] = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
    else:
        vals['unet_name'] = "qwen_image_fp8_e4m3fn.safetensors"
        vals['lora_name'] = "Qwen-Image-fp8-e4m3fn-Lightning-4steps-V1.0-bf16.safetensors"
    
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)

    seed = seed_override if seed_override is not None else int(vals.get('seed', -1))
    vals['seed'] = handle_seed(seed)
    vals['filename_prefix'] = get_filename_prefix()

    workflow = assembler.assemble(vals)
    return workflow, {"extra_pnginfo": {"workflow": ""}}