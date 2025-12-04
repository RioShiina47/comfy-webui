import gradio as gr
import random
import os
import yaml
import traceback
from .wan2_2_txt2video_logic import process_inputs as process_inputs_logic
from core.utils import create_batched_run_generation


UI_INFO = {
    "workflow_recipe": "wan2_2_txt2video_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "txt2video",
    "run_button_text": "🎬 Generate Video"
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
        gr.Markdown("## Wan 2.2 T2V (A14B) - Lightning")
        gr.Markdown("💡 **Tip:** Video generation uses fixed optimization settings (4 steps, CFG 1.0).")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt for video generation...")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)
                
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )
                    components['video_length'] = gr.Slider(label="Video Length (frames)", minimum=8, maximum=81, step=1, value=81)
                
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Video(
                    label="Result", 
                    show_label=False,
                    interactive=False,
                    height=468
                )
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    selected_ratio = local_ui_values.get('aspect_ratio', "16:9 (Landscape)") 
    width, height = ASPECT_RATIO_PRESETS[selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height

    return process_inputs_logic(local_ui_values, seed_override)

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files[-1] if files else None)
)