import gradio as gr
import random
import os
import shutil
import traceback
import tempfile
import time
from PIL import Image

from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix

UI_INFO = {
    "workflow_recipe": "clip_interrogator_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "CLIP Interrogator",
    "run_button_text": "🔍 Interrogate Image"
}

MODE_CHOICES = ["fast", "classic", "best", "negative"]

def save_temp_image(img: Image.Image) -> str:
    if not isinstance(img, Image.Image):
        return None
    filename = f"temp_interrogator_input_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    print(f"Saved temporary image to {filepath}")
    return os.path.basename(filepath)

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## CLIP Interrogator")
        gr.Markdown("💡 **Tip:** Upload an image to generate a descriptive prompt. Requires the `ComfyUI-easy-use` custom node pack.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_image'] = gr.Image(type="pil", label="Input Image", height=380)
                
                components['mode'] = gr.Radio(
                    label="Mode", choices=MODE_CHOICES, value="fast"
                )
                components['use_lowvram'] = gr.Checkbox(label="Use Low VRAM Mode", value=False)
                
                components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
            
            with gr.Column(scale=1):
                components['prompt_output'] = gr.Textbox(
                    label="Generated Prompt", lines=20, interactive=False, show_copy_button=True
                )

    return components

def get_main_output_components(components: dict):
    return [components['prompt_output'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    input_img = local_ui_values.get('input_image')
    if input_img is None:
        raise gr.Error("Please upload an input image.")
        
    local_ui_values['input_image'] = save_temp_image(input_img)
    
    temp_dir = tempfile.gettempdir()
    unique_filename = f"interrogator_prompt_{get_filename_prefix()}_{random.randint(1000, 9999)}.txt"
    expected_output_path = os.path.join(temp_dir, unique_filename)
    
    expected_output_path = expected_output_path.replace("\\", "/")
    
    local_ui_values['output_file_path'] = os.path.dirname(expected_output_path)
    local_ui_values['filename_prefix'] = os.path.basename(expected_output_path).replace('.txt', '')

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(UI_INFO["workflow_recipe"], base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, {"expected_text_file_path": expected_output_path}

def run_generation(ui_values):
    final_text_content = "Processing..."
    expected_text_file_path = None
    try:
        yield ("Status: Preparing...", "Processing...")
        
        workflow, extra_data = process_inputs(ui_values)
        expected_text_file_path = extra_data.get("expected_text_file_path")
        workflow_package = (workflow, extra_data)
        
        for status, _ in run_workflow_and_get_output(workflow_package):
            yield (status, "Processing...")
        
        yield ("Status: Reading generated prompt...", "Processing...")
        
        text_file_found = False
        for _ in range(10): 
            if expected_text_file_path and os.path.exists(expected_text_file_path):
                text_file_found = True
                break
            time.sleep(0.5)

        if text_file_found:
            with open(expected_text_file_path, 'r', encoding='utf-8') as file:
                final_text_content = file.read()
        else:
            final_text_content = "Error: Could not find the generated prompt file after processing."

    except Exception as e:
        traceback.print_exc()
        final_text_content = f"An error occurred: {e}"
        yield (f"Error: {e}", final_text_content)
        return
    finally:
        if expected_text_file_path and os.path.exists(expected_text_file_path):
            try:
                os.remove(expected_text_file_path)
                print(f"Cleaned up temporary prompt file: {expected_text_file_path}")
            except Exception as e:
                print(f"Error cleaning up temporary prompt file: {e}")

    yield ("Status: Loaded successfully!", final_text_content)