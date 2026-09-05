import gradio as gr
import traceback
from .invsr_logic import process_inputs
from core.utils import create_simple_run_generation

UI_INFO = {
    "workflow_recipe": "invsr_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "InvSR",
    "run_button_text": "🚀 Upscale with InvSR"
}

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

run_generation = create_simple_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)