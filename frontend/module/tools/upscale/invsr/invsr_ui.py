import gradio as gr
import random
import os
import shutil
import traceback
from PIL import Image

from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix

UI_INFO = {
    "workflow_recipe": "invsr_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "InvSR",
    "run_button_text": "🚀 Upscale with InvSR"
}

def save_temp_image_from_pil(img: Image.Image, prefix: str) -> str:
    if not isinstance(img, Image.Image):
        return None
    filename = f"temp_{prefix}_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    print(f"Saved temporary image to {filepath}")
    return os.path.basename(filepath)

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## InvSR Upscaler")
        gr.Markdown("💡 **Tip:** Upload an image to upscale using the InvSR method. This implementation uses a fixed `sd-turbo` model.")
        with gr.Row():
            with gr.Column(scale=1):
                components['input_image'] = gr.Image(type="pil", label="Input Image")
                components['num_steps'] = gr.Slider(label="Steps", minimum=1, maximum=5, step=1, value=1)
                components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
            with gr.Column(scale=1):
                 components['output_gallery'] = gr.Gallery(label="Result", show_label=False, object_fit="contain", height=462, interactive=False)
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
    return components

def get_main_output_components(components: dict):
    return [components['output_gallery'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass
    
def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    input_img = local_ui_values.get('input_image')
    if input_img is None:
        raise gr.Error("Please upload an input image.")
        
    local_ui_values['input_image_filename'] = save_temp_image_from_pil(input_img, "invsr_input")

    seed = int(local_ui_values.get('seed', -1))
    if seed == -1:
        local_ui_values['seed'] = random.randint(0, 2**32 - 1)
    
    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    recipe_path = UI_INFO["workflow_recipe"]
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None

def run_generation(ui_values):
    final_files = []
    try:
        yield ("Status: Preparing...", None)
        
        workflow, extra_data = process_inputs(ui_values)
        workflow_package = (workflow, extra_data)
        
        for status, output_files in run_workflow_and_get_output(workflow_package):
            if output_files and isinstance(output_files, list):
                final_files = output_files
            
            yield (status, final_files)

    except Exception as e:
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return

    yield ("Status: Loaded successfully!", final_files)