import gradio as gr
from .Kandinsky_txt2video_logic import process_inputs as process_inputs_logic, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation

UI_INFO = {
    "main_tab": "VideoGen",
    "sub_tab": "Kandinsky T2V",
    "run_button_text": "🎬 Generate Kandinsky Video"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Kandinsky Text-to-Video")
        gr.Markdown("💡 **Tip:** This model supports generating 5 or 10-second clips. The 10s model may require more VRAM.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt for video generation...")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3, value="")
                
                with gr.Row():
                    components['duration'] = gr.Radio(
                        label="Video Duration",
                        choices=["5s", "10s"],
                        value="5s"
                    )
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="16:9 (Widescreen)",
                        interactive=True
                    )
                
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Gallery(
                    label="Result", 
                    show_label=False,
                    interactive=False,
                    height=468,
                    object_fit="contain",
                    columns=2,
                    preview=True
                )
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def process_inputs(ui_values, seed_override=None):
    return process_inputs_logic(ui_values, seed_override)

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)