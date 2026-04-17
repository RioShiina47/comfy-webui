import gradio as gr
import os
import traceback
import time

from .ltx2_prompt_generate_logic import process_inputs
from core.comfy_api import run_workflow_and_get_output

UI_INFO = {
    "main_tab": "Tools",
    "sub_tab": "LTX-2 Prompt Gen",
    "run_button_text": "📄 Generate Prompt"
}

def create_ui():
    components = {}
    
    with gr.Column():
        gr.Markdown("## LTX-2 Prompt Generator")
        gr.Markdown("💡 **Tip:** Generate a detailed prompt for LTX-2 using Gemma 3. You can provide only text, or text + image.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_image'] = gr.Image(type="pil", label="Input Image (Optional)", height=280, visible=True)
                
                components['prompt'] = gr.Textbox(
                    label="Input Prompt", 
                    lines=5, 
                    placeholder="Describe what you want..."
                )

                with gr.Row():
                    components['use_abliterated'] = gr.Checkbox(label="Use Gemma-3 Abliterated LoRA", value=False)
                
                with gr.Accordion("Advanced Settings", open=False):
                    components['max_length'] = gr.Slider(label="Max Length", minimum=16, maximum=1024, step=16, value=256)
                    components['temperature'] = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.05, value=0.7)
                    components['top_k'] = gr.Slider(label="Top K", minimum=0, maximum=1000, step=1, value=64)
                    components['top_p'] = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, step=0.01, value=0.95)
                    components['min_p'] = gr.Slider(label="Min P", minimum=0.0, maximum=1.0, step=0.01, value=0.05)
                    components['repetition_penalty'] = gr.Slider(label="Repetition Penalty", minimum=0.0, maximum=5.0, step=0.01, value=1.05)
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                
                components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
            
            with gr.Column(scale=1):
                components['prompt_output'] = gr.Textbox(
                    label="Generated Prompt", lines=25, interactive=False, show_copy_button=True
                )

    return components

def get_main_output_components(components: dict):
    return [components['prompt_output'], components['run_button']]

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