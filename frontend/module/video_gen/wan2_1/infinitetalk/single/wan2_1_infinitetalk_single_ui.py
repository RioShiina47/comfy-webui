import gradio as gr
from .wan2_1_infinitetalk_single_logic import process_inputs, RESOLUTION_PRESETS, CHUNK_LENGTH
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "wan2_1_infinitetalk_single_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "wan2.1 infinitetalk single",
    "run_button_text": "🗣️ Generate Talking Video"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.1 InfiniteTalk (Single-Speaker)")
        gr.Markdown(f"💡 **Tip:** Upload a reference image of a person and an audio file of speech. Use 'Preview' for a quick test of the first {CHUNK_LENGTH} frames, or 'Generate Full Video' for the final output.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['ref_image'] = gr.Image(type="pil", label="Reference Image", height=282)
            with gr.Column(scale=1):
                components['audio_file'] = gr.Audio(type="filepath", label="Audio File")

        components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the person and the background scene.")
        components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3, value="")

        with gr.Row():
            with gr.Column(scale=1):
                components['mode'] = gr.Radio(
                    choices=[f"Preview ({CHUNK_LENGTH} frames)", "Generate Full Video"],
                    value=f"Preview ({CHUNK_LENGTH} frames)",
                    label="Generation Mode"
                )
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
                        choices=list(RESOLUTION_PRESETS["480p"].keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)
            
            with gr.Column(scale=1):
                components['output_video'] = gr.Gallery(
                    label="Result", show_label=False, interactive=False, height=397,
                    object_fit="contain", columns=2, preview=True
                )
        
        create_lora_ui(components, "wan2_1_infinitetalk_single_lora", accordion_label="LoRA Settings")

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "wan2_1_infinitetalk_single_lora")

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