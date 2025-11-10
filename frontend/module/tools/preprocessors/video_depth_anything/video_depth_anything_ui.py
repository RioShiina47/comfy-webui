import gradio as gr
import random
import os
import shutil
import traceback
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix

UI_INFO = {
    "workflow_recipe": "video_depth_anything_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "Video-Depth-Anything",
    "run_button_text": "🕹️ Run Video Depth Preprocessor"
}

MODEL_CHOICES = [
    "video_depth_anything_vits.pth",
    "video_depth_anything_vitb.pth",
    "video_depth_anything_vitl.pth"
]
PRECISION_CHOICES = ["fp16", "fp32"]
COLORMAP_CHOICES = ["gray", "viridis", "plasma", "inferno", "magma", "cividis", "turbo"]

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

def save_temp_video(file_obj) -> str:
    if file_obj is None: return None
    ext = os.path.splitext(file_obj)[1] or ".mp4"
    temp_filename = f"temp_video_depth_{random.randint(1000, 9999)}{ext}"
    save_path = os.path.join(COMFYUI_INPUT_PATH, temp_filename)
    shutil.copy(file_obj, save_path)
    print(f"Saved temporary input video to: {save_path}")
    return temp_filename

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    input_video = local_ui_values.get('input_video')
    if input_video is None:
        raise gr.Error("Please provide an input video.")
    
    local_ui_values['filename_prefix'] = get_filename_prefix()
    local_ui_values['input_video_filename'] = save_temp_video(input_video)
    
    metadata = get_media_metadata(input_video, is_video=True)
    local_ui_values['fps'] = metadata.get('fps', 30)
    
    recipe_path = UI_INFO["workflow_recipe"]
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None

def run_generation(ui_values):
    final_files = []
    try:
        yield ("Status: Preparing...", None)
        
        workflow, extra_data = process_inputs(ui_values)
        workflow_package = (workflow, extra_data)
        
        for status, output_files in run_workflow_and_get_output(workflow_package):
            if output_files and isinstance(output_files, list):
                final_files = output_files
            
            yield (status, final_files[0] if final_files else None)

    except Exception as e:
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return

    yield ("Status: Loaded successfully!", final_files[0] if final_files else None)