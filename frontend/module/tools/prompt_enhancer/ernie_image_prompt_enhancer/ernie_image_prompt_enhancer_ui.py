import gradio as gr
import os
import traceback
import time

from .ernie_image_prompt_enhancer_logic import process_inputs
from core.comfy_api import run_workflow_and_get_output

UI_INFO = {
    "main_tab": "Tools",
    "sub_tab": "ERNIE Image Prompt Enhancer",
    "run_button_text": "📄 Enhance Prompt"
}

DEFAULT_SYSTEM_PROMPT = "<s>[SYSTEM_PROMPT]你是一个专业的文生图 Prompt 增强助手。你将收到用户的简短图片描述及目标生成分辨率，请据此扩写为一段内容丰富、细节充分的视觉描述，以帮助文生图模型生成高质量的图片。仅输出增强后的描述，不要包含任何解释或前缀。[/SYSTEM_PROMPT]"

ASPECT_RATIO_CHOICES = [
    "1:1 (Square)",
    "16:9 (Landscape)",
    "9:16 (Portrait)",
    "4:3 (Classic)",
    "3:4 (Classic Portrait)",
    "3:2 (Photography)",
    "2:3 (Photography Portrait)"
]

def create_ui():
    components = {}
    
    with gr.Column():
        gr.Markdown("## ERNIE Image Prompt Enhancer")
        gr.Markdown("💡 **Tip:** Generate a detailed image prompt from a simple description and target aspect ratio using ERNIE Image Prompt Enhancer.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['prompt'] = gr.Textbox(
                    label="Input Prompt", 
                    lines=5, 
                    placeholder="A simple description, e.g., a cute anime girl with massive fennec ears..."
                )

                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio", 
                        choices=ASPECT_RATIO_CHOICES, 
                        value="1:1 (Square)"
                    )
                
                with gr.Accordion("Advanced Settings", open=False):
                    components['max_length'] = gr.Slider(label="Max Length", minimum=16, maximum=4096, step=16, value=2048)
                    components['temperature'] = gr.Slider(label="Temperature", minimum=0.0, maximum=2.0, step=0.05, value=0.6)
                    components['top_k'] = gr.Slider(label="Top K", minimum=0, maximum=1000, step=1, value=64)
                    components['top_p'] = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, step=0.01, value=0.8)
                    components['min_p'] = gr.Slider(label="Min P", minimum=0.0, maximum=1.0, step=0.01, value=0.05)
                    components['repetition_penalty'] = gr.Slider(label="Repetition Penalty", minimum=0.0, maximum=5.0, step=0.01, value=1.05)
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

                with gr.Accordion("System Prompt", open=False):
                    components['system_prompt'] = gr.Textbox(
                        label="SYSTEM_PROMPT", 
                        lines=5, 
                        value=DEFAULT_SYSTEM_PROMPT
                    )
                
                components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
            
            with gr.Column(scale=1):
                components['prompt_output'] = gr.Textbox(
                    label="Enhanced Prompt", lines=25, interactive=False, show_copy_button=True
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
        
        yield ("Status: Reading enhanced prompt...", "Processing...")
        
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
            final_text_content = "Error: Could not find the enhanced prompt file after processing."

    except Exception as e:
        import traceback
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