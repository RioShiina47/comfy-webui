import gradio as gr
from .ltx2_5_i2v_logic import process_inputs, RESOLUTION_PRESETS
from core.utils import create_batched_run_generation
from core import node_info_manager
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "main_tab": "VideoGen",
    "sub_tab": "LTX-2.5 I2V",
    "run_button_text": "🎬 Generate LTX-2.5 Video"
}

REQUIRED_LORA_DIRS = ["ltx-2.5", "ltx-2"]

def create_ui():
    components = {}

    with gr.Column():
        gr.Markdown("## LTX-2.5 Image-to-Video")
        gr.Markdown("💡 **Tip:** This model generates video and audio simultaneously from a starting image and a text prompt.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['start_image'] = gr.Image(type="pil", label="Start Image", height=294)
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the desired content and motion...")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=4)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['resolution'] = gr.Radio(
                        label="Resolution",
                        choices=["480P", "768P", "1080P"],
                        value="768P",
                        interactive=True
                    )
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(RESOLUTION_PRESETS['768P'].keys()),
                        value="16:9 (Widescreen)",
                        interactive=True
                    )
                with gr.Row():
                    components['duration'] = gr.Slider(
                        label="Duration (seconds)",
                        minimum=1.0,
                        maximum=20.0,
                        step=1.0,
                        value=5.0,
                        interactive=True
                    )
                    components['fps'] = gr.Dropdown(
                        label="FPS",
                        choices=["24fps", "25fps"],
                        value="24fps",
                        interactive=True
                    )
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

                with gr.Row():
                    components['use_spatial_upscaler'] = gr.Checkbox(label="Use 2x Spatial Upscaler", value=False, interactive=True)
                    components['use_temporal_upscaler'] = gr.Checkbox(label="Use 2x Temporal Upscaler", value=False, interactive=True)

            with gr.Column(scale=1):
                components['output_video'] = gr.Gallery(
                    label="Result", 
                    show_label=False,
                    interactive=False,
                    object_fit="contain",
                    columns=2,
                    preview=True,
                    height=460
                )
        
        create_lora_ui(components, "ltx2_5_i2v_lora", required_lora_dirs=REQUIRED_LORA_DIRS)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "ltx2_5_i2v_lora")

    def update_aspect_ratio_choices(resolution):
        preset_dict = RESOLUTION_PRESETS.get(resolution, RESOLUTION_PRESETS.get(resolution.upper(), RESOLUTION_PRESETS["768P"]))
        return gr.update(choices=list(preset_dict.keys()))

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
