import gradio as gr
import traceback
from .video_depth_anything_logic import process_inputs, MODEL_CHOICES, PRECISION_CHOICES, COLORMAP_CHOICES
from core.utils import create_simple_run_generation

UI_INFO = {
    "main_tab": "Tools",
    "sub_tab": "Video-Depth-Anything",
    "run_button_text": "🕹️ Run Video Depth Preprocessor"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Video Depth Anything")
        gr.Markdown("💡 **Tip:** Upload a video to generate a depth map video. The model is specifically designed for video inputs.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_video'] = gr.Video(label="Input Video")
                
                with gr.Accordion("Settings", open=True):
                    components['model'] = gr.Dropdown(label="Model", choices=MODEL_CHOICES, value=MODEL_CHOICES[0])
                    components['colormap'] = gr.Dropdown(label="Colormap", choices=COLORMAP_CHOICES, value="gray")
                    components['input_size'] = gr.Slider(label="Input Size", minimum=256, maximum=1024, step=2, value=518, info="The input size for the model.")
                    components['max_res'] = gr.Slider(label="Max Resolution", minimum=640, maximum=3840, step=64, value=1920, info="The maximum resolution for processing.")
                    components['precision'] = gr.Dropdown(label="Precision", choices=PRECISION_CHOICES, value="fp16")
                
            with gr.Column(scale=1):
                gr.Markdown("### Result")
                components['output_video'] = gr.Video(label="Result Video", show_label=False, interactive=False)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_simple_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)