import gradio as gr
import random
import os
import yaml
import traceback
from core.comfy_api import run_workflow_and_get_output

from .wan2_2_txt2video_logic import process_inputs as process_inputs_logic


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

def run_generation(ui_values):
    all_output_files = []
    
    try:
        batch_count = int(ui_values.get('batch_count', 1))
        original_seed = int(ui_values.get('seed', -1))

        for i in range(batch_count):
            current_seed = original_seed + i if original_seed != -1 else None
            batch_msg = f" (Batch {i + 1}/{batch_count})" if batch_count > 1 else ""
            
            yield (f"Status: Preparing{batch_msg}...", None)
            
            workflow, extra_data = process_inputs(ui_values, seed_override=current_seed)
            workflow_package = (workflow, extra_data)
            
            for status, output_path in run_workflow_and_get_output(workflow_package):
                status_msg = f"Status: {status.replace('Status: ', '')}{batch_msg}"
                
                final_video = None
                if output_path and isinstance(output_path, list):
                    all_output_files.extend(output_path)
                    final_video = all_output_files[-1]

                yield (status_msg, final_video)

    except Exception as e:
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return

    yield ("Status: Loaded successfully!", all_output_files[-1] if all_output_files else None)