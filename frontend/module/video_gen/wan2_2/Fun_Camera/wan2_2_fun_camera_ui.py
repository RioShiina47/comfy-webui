import gradio as gr
import traceback

from .wan2_2_fun_camera_logic import process_inputs, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation


UI_INFO = {
    "workflow_recipe": "wan2_2_fun_camera_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "Fun Camera",
    "run_button_text": "🎥 Generate with Camera"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.2 Fun Camera (14B) - Lightning")
        gr.Markdown("💡 **Tip:** Please select an aspect ratio for the output video.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['start_image'] = gr.Image(type="pil", label="Start Image", height=255)
            
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the scene.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )
                    components['video_length'] = gr.Slider(label="Video Length (frames)", minimum=8, maximum=81, step=1, value=81)
                
                with gr.Row():
                    components['camera_pose'] = gr.Dropdown(
                        label="Camera Pose", 
                        choices=["Zoom In", "Zoom Out", "Pan Left", "Pan Right", "Tilt Up", "Tilt Down", "Roll Clockwise", "Roll Counter-Clockwise"],
                        value="Zoom In"
                    )
                    components['speed'] = gr.Slider(label="Speed", minimum=0.1, maximum=5.0, step=0.1, value=1.0)
                
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1, visible=True)

                with gr.Row():
                    components['fx'] = gr.Slider(label="Focal X (fx)", minimum=0.0, maximum=1.0, step=0.01, value=0.5)
                    components['fy'] = gr.Slider(label="Focal Y (fy)", minimum=0.0, maximum=1.0, step=0.01, value=0.5)
                with gr.Row():
                    components['cx'] = gr.Slider(label="Center X (cx)", minimum=0.0, maximum=1.0, step=0.01, value=0.5)
                    components['cy'] = gr.Slider(label="Center Y (cy)", minimum=0.0, maximum=1.0, step=0.01, value=0.5)

            with gr.Column(scale=1):
                components['output_video'] = gr.Video(label="Result", show_label=False, interactive=False, height=488)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files[-1] if files else None)
)