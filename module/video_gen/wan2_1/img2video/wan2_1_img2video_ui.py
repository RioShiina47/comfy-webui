import gradio as gr
from .wan2_1_img2video_logic import process_inputs, RESOLUTION_PRESETS
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "wan2_1_img2video_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "wan2.1 img2video",
    "run_button_text": "🎬 Generate from Image"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.1 I2V (14B) - Lightning")
        gr.Markdown("💡 **Tip:** Upload an image, describe the desired motion, and select an aspect ratio. The image will be resized to fit.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['start_image'] = gr.Image(type="pil", label="Input Image", height=294)
            
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=4)
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
                        choices=list(RESOLUTION_PRESETS["720p"].keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )
                
                with gr.Row():
                    components['video_length'] = gr.Slider(label="Video Length (frames)", minimum=8, maximum=81, step=1, value=81)
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                
                with gr.Row():
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)
                    components['use_easy_cache'] = gr.Checkbox(label="Use EasyCache", value=True)

            with gr.Column(scale=1):
                components['output_video'] = gr.Gallery(
                    label="Result", show_label=False, interactive=False, height=390,
                    object_fit="contain", columns=2, preview=True
                )
        
        create_lora_ui(components, "wan2_1_img2video_lora", accordion_label="LoRA Settings")

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "wan2_1_img2video_lora")

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)