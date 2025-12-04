import gradio as gr
import traceback

from .humo_logic import process_inputs, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation

UI_INFO = {
    "workflow_recipe": "humo_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "HuMo",
    "run_button_text": "🕺 Generate HuMo Video"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## HuMo (Human Motion)")
        gr.Markdown("💡 **Tip:** Upload a reference image and an audio file. You can set the video length manually up to 97 frames.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['ref_image'] = gr.Image(type="pil", label="Reference Image", height=282)
            with gr.Column(scale=1):
                components['audio_file'] = gr.Audio(type="filepath", label="Audio File")

        with gr.Row():
            with gr.Column(scale=1):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the character and scene.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3, value="")
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )
                    components['video_length'] = gr.Slider(label="Video Length (frames)", minimum=8, maximum=97, step=1, value=97)
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Video(
                    label="Result", 
                    show_label=False, 
                    interactive=False, 
                    height=468
                )

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)