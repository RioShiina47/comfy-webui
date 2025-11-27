import random
import os
from PIL import Image
import gradio as gr
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix

WORKFLOW_RECIPE_PATH = "flux2_dev_recipe.yaml"
MAX_REF_IMAGES = 10

def save_temp_image(img: Image.Image, name: str) -> str:
    if not isinstance(img, Image.Image):
        return None
    filename = f"temp_flux2_dev_{name}_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    print(f"Saved temporary image to {filepath}")
    return filename

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    if not local_ui_values.get('positive_prompt'):
        raise gr.Error("Prompt is required.")

    all_images = []
    if local_ui_values.get('ref_image_inputs') and isinstance(local_ui_values['ref_image_inputs'], list):
        for img in local_ui_values['ref_image_inputs']:
            if img is not None:
                all_images.append(img)
    
    if all_images:
        image_filenames = [save_temp_image(img, f"ref_{i}") for i, img in enumerate(all_images)]
        local_ui_values['reference_chain'] = image_filenames
    else:
        local_ui_values['reference_chain'] = []


    seed = int(local_ui_values.get('seed', -1))
    if seed_override is not None:
        local_ui_values['seed'] = seed_override
    elif seed == -1:
        local_ui_values['seed'] = random.randint(0, 2**32 - 1)
        
    local_ui_values['filename_prefix'] = get_filename_prefix()

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None