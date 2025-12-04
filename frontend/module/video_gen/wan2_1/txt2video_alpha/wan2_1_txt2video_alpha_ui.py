import gradio as gr
import traceback

from .wan2_1_txt2video_alpha_logic import process_inputs, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation

UI_INFO = {
    "workflow_recipe": "wan2_1_txt2video_alpha_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "txt2video(alpha)",
    "run_button_text": "🎬 Generate Alpha Video"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.1 T2V Alpha (14B) - Lightning")
        gr.Markdown("💡 **Tip:** This model generates both a standard RGB video and an alpha channel (mask) video.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt for video generation...")
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
                components['output_gallery'] = gr.Gallery(
                    label="Result (RGB Video and Alpha Mask)",
                    show_label=True,
                    object_fit="contain",
                    height=468,
                    columns=2,
                    preview=True
                )

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