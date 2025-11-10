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
    "workflow_recipe": "wd14_tagger_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "WD14 Tagger",
    "run_button_text": "🏷️ Generate Tags"
}

MODEL_CHOICES = [
    "wd-eva02-large-tagger-v3",
    "wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3",
    "wd-convnext-tagger-v3",
    "wd-v1-4-moat-tagger-v2",
    "wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-convnext-tagger-v2",
    "wd-v1-4-convnext-tagger",
    "wd-v1-4-vit-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2",
    "wd-v1-4-vit-tagger",
]
DEFAULT_MODEL = "wd-convnext-tagger-v3"

def save_temp_image(img: Image.Image) -> str:
    if not isinstance(img, Image.Image):
        return None
    filename = f"temp_wd14_input_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    print(f"Saved temporary image to {filepath}")
    return os.path.basename(filepath)

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## WD14 Tagger")
        gr.Markdown("💡 **Tip:** Upload an image to automatically generate descriptive tags. Requires the `WD14 Tagger` custom node (`pysssss`).")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_image'] = gr.Image(type="pil", label="Input Image", height=380)
                
                components['model'] = gr.Dropdown(
                    label="Model", choices=MODEL_CHOICES, value=DEFAULT_MODEL
                )
                components['threshold'] = gr.Slider(
                    label="General Tag Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.35
                )
                components['character_threshold'] = gr.Slider(
                    label="Character Tag Threshold", minimum=0.0, maximum=1.0, step=0.05, value=0.85
                )
                components['exclude_tags'] = gr.Textbox(
                    label="Exclude Tags", placeholder="e.g., 1girl, solo"
                )
                with gr.Row():
                    components['replace_underscore'] = gr.Checkbox(label="Replace Underscores with Spaces", value=True)
                    components['trailing_comma'] = gr.Checkbox(label="Add Trailing Comma", value=False)
                
                components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
            
            with gr.Column(scale=1):
                components['tags_output'] = gr.Textbox(
                    label="Generated Tags", lines=20, interactive=False, show_copy_button=True
                )

    return components

def get_main_output_components(components: dict):
    return [components['tags_output'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    input_img = local_ui_values.get('input_image')
    if input_img is None:
        raise gr.Error("Please upload an input image.")
        
    local_ui_values['input_image'] = save_temp_image(input_img)
    
    temp_dir = tempfile.gettempdir()
    unique_filename = f"wd14_tags_{get_filename_prefix()}_{random.randint(1000, 9999)}.txt"
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
        
        yield ("Status: Reading generated tags...", "Processing...")
        
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
            final_text_content = "Error: Could not find the generated tags file after processing."

    except Exception as e:
        traceback.print_exc()
        final_text_content = f"An error occurred: {e}"
        yield (f"Error: {e}", final_text_content)
        return
    finally:
        if expected_text_file_path and os.path.exists(expected_text_file_path):
            try:
                os.remove(expected_text_file_path)
                print(f"Cleaned up temporary tags file: {expected_text_file_path}")
            except Exception as e:
                print(f"Error cleaning up temporary tags file: {e}")

    yield ("Status: Loaded successfully!", final_text_content)