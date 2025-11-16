import random
import os
from PIL import Image
import gradio as gr
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix

WORKFLOW_RECIPE_PATH = "chrono_edit_recipe.yaml"

def save_temp_image(img: Image.Image, name: str) -> str:
    if not isinstance(img, Image.Image):
        return None
    filename = f"temp_chrono_edit_{name}_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    print(f"Saved temporary image to {filepath}")
    return os.path.basename(filepath)

def process_inputs(params: dict, seed_override=None):
    local_params = params.copy()
    
    main_img = local_params.get('start_image')
    if main_img is None:
        raise gr.Error("Please upload an image to edit.")

    local_params['start_image'] = save_temp_image(main_img, "start")

    seed = int(local_params.get('seed', -1))
    if seed_override is not None:
        local_params['seed'] = seed_override
    elif seed == -1:
        local_params['seed'] = random.randint(0, 999999999999999)

    local_params['filename_prefix'] = get_filename_prefix()

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_params)
    
    return final_workflow, None