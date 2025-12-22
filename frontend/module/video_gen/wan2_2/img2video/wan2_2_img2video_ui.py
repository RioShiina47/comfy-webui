import gradio as gr
import random
import os
import yaml
import traceback
from PIL import Image
from .wan2_2_img2video_logic import process_inputs as process_inputs_logic
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "wan2_2_img2video_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "img2video",
    "run_button_text": "🎬 Generate from Image"
}

ASPECT_RATIO_PRESETS = {
    "16:9 (Landscape)": (1280, 720),
    "9:16 (Portrait)": (720, 1280),
    "1:1 (Square)": (960, 960),
    "4:3 (Classic TV)": (1088, 816),
    "3:4 (Classic Portrait)": (816, 1088),
    "3:2 (Photography)": (1152, 768),
    "2:3 (Photography Portrait)": (768, 1152),
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.2 I2V (A14B) - Lightning")
        gr.Markdown("💡 **Tip:** Upload an image and select an aspect ratio. The image will be resized to fit.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['start_image'] = gr.Image(type="pil", label="Input Image", height=294)
            
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Prompt (Motion Description)", lines=4, placeholder="Describe the motion or change you want to see...")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=4)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )
                
                with gr.Row():
                    components['video_length'] = gr.Slider(label="Video Length (frames)", minimum=8, maximum=81, step=1, value=81)
                
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

                with gr.Row():
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Video(
                    label="Result", show_label=False, interactive=False, height=378
                )
        
        create_lora_ui(components, "high_noise", accordion_label="High Noise LoRA Settings")
        create_lora_ui(components, "low_noise", accordion_label="Low Noise LoRA Settings")

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_ui_chain_events(components, "high_noise")
    register_ui_chain_events(components, "low_noise")

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    selected_ratio = local_ui_values.get('aspect_ratio')
    width, height = ASPECT_RATIO_PRESETS[selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    return process_inputs_logic(local_ui_values, seed_override)

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files[-1] if files else None)
)