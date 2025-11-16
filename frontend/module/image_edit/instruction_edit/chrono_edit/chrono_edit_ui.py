import gradio as gr
import random
import os
import traceback
from PIL import Image

from .chrono_edit_logic import process_inputs
from core.comfy_api import run_workflow_and_get_output

UI_INFO = {
    "workflow_recipe": "chrono_edit_recipe.yaml",
    "main_tab": "ImageEdit",
    "sub_tab": "ChronoEdit",
    "run_button_text": "🎨 Edit with Chrono"
}

ASPECT_RATIO_PRESETS = {
    "16:9 (Landscape)": (1280, 720),
    "9:16 (Portrait)": (720, 1280),
    "1:1 (Square)": (960, 960),
    "4:3 (Classic TV)": (1088, 816),
    "3:4 (Classic Portrait)": (816, 1088),
    "3:2 (Photography)": (1152, 768),
    "2:3 (Photography Portrait)": (768, 1152),
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## ChronoEdit Image Editing")
        gr.Markdown("💡 **Tip:** Upload an image and provide a text instruction to edit it.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['start_image'] = gr.Image(type="pil", label="Input Image", height=294)
            
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Edit Instruction", lines=4, placeholder="e.g., Make it a rainy day.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=4)

        with gr.Row():
            with gr.Column(scale=1):
                components['aspect_ratio'] = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=list(ASPECT_RATIO_PRESETS.keys()),
                    value="9:16 (Portrait)",
                    interactive=True
                )
                components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)
            
            with gr.Column(scale=1):
                components['output_gallery'] = gr.Gallery(label="Result", show_label=False, object_fit="contain", height=378)
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_gallery'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def run_generation(ui_values):
    all_output_files = []
    
    try:
        batch_count = int(ui_values.get('batch_count', 1))
        original_seed = int(ui_values.get('seed', -1))

        for i in range(batch_count):
            current_seed = original_seed + i if original_seed != -1 else None
            batch_msg = f" (Batch {i + 1}/{batch_count})" if batch_count > 1 else ""
            
            yield (f"Status: Preparing{batch_msg}...", None)
            
            local_ui_values = ui_values.copy()
            selected_ratio = local_ui_values.get('aspect_ratio')
            width, height = ASPECT_RATIO_PRESETS[selected_ratio]
            local_ui_values['width'] = width
            local_ui_values['height'] = height
            
            workflow, extra_data = process_inputs(local_ui_values, seed_override=current_seed)
            workflow_package = (workflow, extra_data)
            
            for status, output_path in run_workflow_and_get_output(workflow_package):
                status_msg = f"Status: {status.replace('Status: ', '')}{batch_msg}"
                
                if output_path and isinstance(output_path, list):
                    all_output_files.extend(p for p in output_path if p not in all_output_files)

                yield (status_msg, all_output_files)

    except Exception as e:
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return

    yield ("Status: Loaded successfully!", all_output_files)