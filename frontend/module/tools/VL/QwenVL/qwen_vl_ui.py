import gradio as gr
import random
import os
import shutil
import traceback
import tempfile
import time
from PIL import Image

from .qwen_vl_logic import process_inputs
from core.comfy_api import run_workflow_and_get_output
from core import node_info_manager

UI_INFO = {
    "workflow_recipe": "qwen_vl_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "QwenVL",
    "run_button_text": "💬 Describe Image"
}

def create_ui():
    components = {}
    
    try:
        preset_prompt_choices = node_info_manager.get_node_input_options("AILab_QwenVL", "preset_prompt")
        model_choices = node_info_manager.get_node_input_options("AILab_QwenVL", "model_name")
        quantization_choices = node_info_manager.get_node_input_options("AILab_QwenVL", "quantization")
        if not all([preset_prompt_choices, model_choices, quantization_choices]):
            raise ValueError("Could not fetch all required options from node info.")
    except Exception as e:
        print(f"[UI Build Error] Could not get options for AILab_QwenVL: {e}")
        preset_prompt_choices = ["Prompt Style - Detailed", "Prompt Style - Simple", "Prompt Style - Poetic", "Custom"]
        model_choices = ["Qwen3-VL-4B-Instruct-FP8", "Qwen2-VL-7B-Instruct-FP8"]
        quantization_choices = ["None (FP16)", "INT8", "INT4"]
        gr.Warning("Could not load QwenVL options. Is 'ComfyUI_AILab_Qwen' installed and the backend running?")

    with gr.Column():
        gr.Markdown("## QwenVL Image Describer")
        gr.Markdown("💡 **Tip:** Upload an image to generate a detailed description. Requires the `ComfyUI_AILab_Qwen` custom node.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_image'] = gr.Image(type="pil", label="Input Image", height=380)
                
                components['model_mode'] = gr.Radio(["Instruct", "Thinking"], label="Model Mode", value="Instruct")
                components['preset_prompt'] = gr.Dropdown(label="Prompt Style", choices=preset_prompt_choices, value=preset_prompt_choices[0] if preset_prompt_choices else "Prompt Style - Detailed")
                components['custom_prompt'] = gr.Textbox(
                    label="Custom Prompt", 
                    placeholder="If provided, this will override the preset prompt.",
                    lines=5,
                    visible=True
                )

                components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
            
            with gr.Column(scale=1):
                components['description_output'] = gr.Textbox(
                    label="Generated Description", lines=20, interactive=False, show_copy_button=True
                )

    components['quantization'] = gr.State("None (FP16)")
    components['max_tokens'] = gr.State(1024)
    components['keep_model_loaded'] = gr.State(False)

    return components

def get_main_output_components(components: dict):
    return [components['description_output'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

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
        
        yield ("Status: Reading generated description...", "Processing...")
        
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
            final_text_content = "Error: Could not find the generated description file after processing."

    except Exception as e:
        traceback.print_exc()
        final_text_content = f"An error occurred: {e}"
        yield (f"Error: {e}", final_text_content)
        return
    finally:
        if expected_text_file_path and os.path.exists(expected_text_file_path):
            try:
                os.remove(expected_text_file_path)
                print(f"Cleaned up temporary description file: {expected_text_file_path}")
            except Exception as e:
                print(f"Error cleaning up temporary description file: {e}")

    yield ("Status: Loaded successfully!", final_text_content)