import gradio as gr
from .ltx2_control_logic import process_inputs, RESOLUTION_PRESETS
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "main_tab": "VideoGen",
    "sub_tab": "LTX-2 Control",
    "run_button_text": "🎬 Generate LTX-2 Control Video"
}

REQUIRED_LORA_DIRS = ["ltx-2"]

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## LTX-2 Control")
        gr.Markdown("💡 **Tip:** Provide a control video to guide the motion structure. The first frame of the control video will be used for guidance. You can optionally provide a start image to set the initial scene.")
        
        with gr.Row():
            components['start_image'] = gr.Image(type="pil", label="Start Image", scale=1, height=294)
            components['control_video'] = gr.Video(label="Control Video", scale=1, height=294)

        components['positive_prompt'] = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the desired content and style...")
        components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=4)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['resolution'] = gr.Radio(
                        label="Resolution",
                        choices=["480p", "720p"],
                        value="720p",
                        interactive=True
                    )
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(RESOLUTION_PRESETS['480p'].keys()),
                        value="16:9 (Widescreen)",
                        interactive=True
                    )
                with gr.Row():
                    components['control_type'] = gr.Dropdown(
                        label="Control",
                        choices=["Canny", "Depth", "Pose"],
                        value="Canny",
                        interactive=True
                    )
                    components['video_length'] = gr.Slider(label="Video Length (frames)", minimum=9, maximum=121, step=8, value=121)
                
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Gallery(
                    label="Result", 
                    show_label=False,
                    interactive=False,
                    object_fit="contain",
                    columns=2,
                    preview=True,
                    height=408
                )
        
        create_lora_ui(components, "ltx2_control_lora", required_lora_dirs=REQUIRED_LORA_DIRS)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "ltx2_control_lora")

    def update_aspect_ratio_choices(resolution):
        return gr.update(choices=list(RESOLUTION_PRESETS[resolution].keys()))

    components['resolution'].change(
        fn=update_aspect_ratio_choices,
        inputs=[components['resolution']],
        outputs=[components['aspect_ratio']],
        show_api=False
    )

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)