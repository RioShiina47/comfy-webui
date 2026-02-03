import gradio as gr
from .wan2_1_infinitetalk_multi_logic import process_inputs, RESOLUTION_PRESETS, CHUNK_LENGTH
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "wan2_1_infinitetalk_multi_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "wan2.1 infinitetalk multi",
    "run_button_text": "🗣️ Generate Multi-Speaker Video"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.1 InfiniteTalk (Multi-Speaker)")
        gr.Markdown(f"💡 **Tip:** Upload the same background image to both editors. Use the brush on the left for Speaker 1 and the brush on the right for Speaker 2. Upload a separate audio file for each. Use 'Preview' to check the first {CHUNK_LENGTH} frames, or 'Generate Full Video' for longer clips based on audio duration.")
        
        with gr.Row():
            components['ref_image_1'] = gr.ImageEditor(
                type="pil", 
                label="Speaker 1: Image & Mask",
                brush=gr.Brush(colors=["#FF0000"], color_mode="fixed"),
                height=400,
                scale=1
            )
            components['ref_image_2'] = gr.ImageEditor(
                type="pil", 
                label="Speaker 2: Image & Mask",
                brush=gr.Brush(colors=["#0000FF"], color_mode="fixed"),
                height=400,
                scale=1
            )

        with gr.Row():
             components['audio_1'] = gr.Audio(type="filepath", label="Audio Speaker 1", scale=1)
             components['audio_2'] = gr.Audio(type="filepath", label="Audio Speaker 2", scale=1)

        components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the scene and the people.")
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
        
        create_lora_ui(components, "wan2_1_infinitetalk_multi_lora", accordion_label="LoRA Settings")

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "wan2_1_infinitetalk_multi_lora")

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