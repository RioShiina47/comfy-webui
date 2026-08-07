import gradio as gr
from .h3_fl2va_logic import process_inputs, RESOLUTION_PRESETS
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "h3_fl2va_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "H3 FL2VA",
    "run_button_text": "🎬 Generate H3 Video"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## MiniMax H3 Video Generation")
        gr.Markdown("💡 **Tip:** No image for T2VA; upload First Frame for I2VA; upload First Frame & Last Frame for FL2VA.")
        
        with gr.Row():
            components['first_frame'] = gr.Image(type="pil", label="First Frame (Optional)", height=220)
            components['last_frame'] = gr.Image(type="pil", label="Last Frame (Optional)", height=220)

        components['prompt'] = gr.Textbox(label="Prompt", lines=5)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['resolution'] = gr.Radio(
                        label="Resolution",
                        choices=["480p", "720p", "1080p"],
                        value="720p",
                        interactive=True
                    )

                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(RESOLUTION_PRESETS["720p"].keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )

                with gr.Row():
                    components['width'] = gr.Number(label="Width", value=1344, precision=0)
                    components['height'] = gr.Number(label="Height", value=768, precision=0)
                with gr.Row():
                    components['duration'] = gr.Slider(
                        label="Duration (seconds)",
                        minimum=0.2,
                        maximum=15.0,
                        step=0.1,
                        value=3.0,
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
                    height=492,
                    object_fit="contain",
                    columns=2,
                    preview=True
                )

        create_lora_ui(components, "h3_fl2va_lora", accordion_label="LoRA Settings")

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "h3_fl2va_lora")

    def update_dimensions(resolution, aspect_ratio):
        w, h = RESOLUTION_PRESETS.get(resolution, {}).get(aspect_ratio, (1344, 768))
        return w, h

    components['resolution'].change(
        fn=update_dimensions,
        inputs=[components['resolution'], components['aspect_ratio']],
        outputs=[components['width'], components['height']],
        show_api=False
    )

    components['aspect_ratio'].change(
        fn=update_dimensions,
        inputs=[components['resolution'], components['aspect_ratio']],
        outputs=[components['width'], components['height']],
        show_api=False
    )

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)
