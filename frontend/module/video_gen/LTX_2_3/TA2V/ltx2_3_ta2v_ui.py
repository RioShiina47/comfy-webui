import gradio as gr
from .ltx2_3_ta2v_logic import process_inputs, RESOLUTION_PRESETS, FPS
from core.utils import create_batched_run_generation

UI_INFO = {
    "main_tab": "VideoGen",
    "sub_tab": "LTX-2.3 TA2V",
    "run_button_text": "🎬 Generate TA2V Video"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## LTX-2.3 Text & Audio to Video")
        gr.Markdown("💡 **Tip:** Provide a text prompt and an audio clip. The model will generate a video matching the prompt and synchronizing with the audio.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['audio_file'] = gr.Audio(type="filepath", label="Audio File")
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=4, placeholder="Describe the scene and action...")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=4)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['resolution'] = gr.Radio(
                        label="Resolution",
                        choices=["480p", "720p"],
                        value="480p",
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
                    height=352
                )

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
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