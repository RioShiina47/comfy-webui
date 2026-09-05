import gradio as gr
import random
import os
import traceback
from PIL import Image

from .chrono_edit_logic import process_inputs, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation

UI_INFO = {
    "workflow_recipe": "chrono_edit_recipe.yaml",
    "main_tab": "ImageEdit",
    "sub_tab": "ChronoEdit",
    "run_button_text": "🎨 Edit with Chrono"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## ChronoEdit Image Editing")
        gr.Markdown("💡 **Tip:** Upload an image and provide a text instruction to edit it.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['start_image'] = gr.Image(type="pil", label="Input Image", height=255)
            
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Edit Instruction", lines=3, placeholder="e.g., Make it a rainy day.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)

        with gr.Row():
            with gr.Column(scale=1):
                components['aspect_ratio'] = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=list(ASPECT_RATIO_PRESETS.keys()),
                    value="9:16 (Portrait)",
                    interactive=True
                )
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                with gr.Row():
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)
            
            with gr.Column(scale=1):
                components['output_gallery'] = gr.Gallery(label="Result", show_label=False, object_fit="contain", height=378)
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_gallery'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)