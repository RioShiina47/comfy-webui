import gradio as gr
from .Kandinsky_img2video_logic import process_inputs as process_inputs_logic, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation

UI_INFO = {
    "main_tab": "VideoGen",
    "sub_tab": "Kandinsky I2V",
    "run_button_text": "🎬 Generate Kandinsky Video from Image"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Kandinsky Image-to-Video")
        gr.Markdown("💡 **Tip:** Upload an image and describe the desired motion. This model only supports 5-second video generation.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['start_image'] = gr.Image(type="pil", label="Start Image", height=294)
                
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Prompt (Motion Description)", lines=4, placeholder="Describe the motion or change you want to see...")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=4, value="")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['duration'] = gr.Radio(
                        label="Video Duration",
                        choices=["5s"],
                        value="5s",
                        interactive=False
                    )
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="16:9 (Widescreen)",
                        interactive=True
                    )
                
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                with gr.Row():
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Gallery(
                    label="Result", 
                    show_label=False,
                    interactive=False,
                    height=390,
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