import gradio as gr
import traceback

from .z_image_outpaint_logic import process_inputs
from core.utils import create_simple_run_generation

UI_INFO = { 
    "main_tab": "ImageEdit", 
    "sub_tab": "Z-Image Outpaint",
    "run_button_text": "🎨 Outpaint with Z-Image" 
}
PREFIX = "z_image_outpaint"

def create_ui():
    components = {}
    key = lambda name: f"{PREFIX}_{name}"
    
    with gr.Column():
        gr.Markdown("## Z-Image Outpainting")
        gr.Markdown("💡 **Tip:** Upload an image, use the sliders to define the area to expand, and describe the content for the new area.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components[key('input_image')] = gr.Image(type="pil", label="Input Image", height=255)
            with gr.Column(scale=2):
                components[key('positive_prompt')] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the content for the expanded areas...", interactive=True)
                components[key('negative_prompt')] = gr.Textbox(label="Negative Prompt", lines=3, interactive=True)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components[key('left')] = gr.Slider(label="Pad Left", minimum=0, maximum=512, step=8, value=0)
                    components[key('top')] = gr.Slider(label="Pad Top", minimum=0, maximum=512, step=8, value=0)
                with gr.Row():
                    components[key('right')] = gr.Slider(label="Pad Right", minimum=0, maximum=512, step=8, value=256)
                    components[key('bottom')] = gr.Slider(label="Pad Bottom", minimum=0, maximum=512, step=8, value=0)
                
                components[key('feathering')] = gr.Slider(label="Feathering", minimum=0, maximum=100, step=1, value=40, interactive=True)
                
                with gr.Row():
                    components[key('seed')] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0, interactive=True)

            with gr.Column(scale=1):
                components[key('output_gallery')] = gr.Gallery(
                    label="Result", show_label=False, object_fit="contain", height=362
                )
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
        components[key('run_button')] = components['run_button']
    
    return components

def get_main_output_components(components: dict):
    return [components[f'{PREFIX}_output_gallery'], components[f'{PREFIX}_run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_simple_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)