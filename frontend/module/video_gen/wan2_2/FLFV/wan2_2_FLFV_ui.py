import gradio as gr
import traceback
from .wan2_2_FLFV_logic import process_inputs, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation

UI_INFO = {
    "workflow_recipe": "wan2_2_FLFV_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "FLFV",
    "run_button_text": "🚀 Generate FLFV Video"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.2 FLFV (First-Last-Frame-Video) (14B) - Lightning")
        gr.Markdown("💡 **Tip:** Please select an aspect ratio for the output video.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['start_image'] = gr.Image(type="pil", label="Start Image")
            with gr.Column(scale=1):
                components['end_image'] = gr.Image(type="pil", label="End Image")

        with gr.Row():
            with gr.Column(scale=1):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the scene and the desired transformation.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )
                    components['video_length'] = gr.Slider(label="Video Length (frames)", minimum=8, maximum=81, step=1, value=81)
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Video(label="Result", show_label=False, interactive=False, height=472)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files[-1] if files else None)
)