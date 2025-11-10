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
    "workflow_recipe": "depth_anything_tensorrt_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "Depth-Anything-Tensorrt",
    "run_button_text": "🕹️ Run Depth Preprocessor"
}

ENGINE_CHOICES = [
    "v2_depth_anything_vitl-fp16.engine",
    "v2_depth_anything_vitb-fp16.engine",
    "v2_depth_anything_vits-fp16.engine"
]

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Depth Anything (TensorRT)")
        gr.Markdown("💡 **Tip:** Upload an image or video to generate a depth map using a TensorRT-accelerated model.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_type'] = gr.Radio(["Image", "Video"], label="Input Type", value="Image")
                components['input_image'] = gr.Image(type="pil", label="Input Image", visible=True)
                components['input_video'] = gr.Video(label="Input Video", visible=False)
                
                components['engine'] = gr.Dropdown(
                    label="Engine File", 
                    choices=ENGINE_CHOICES, 
                    value=ENGINE_CHOICES[0]
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### Result")
                components['output_gallery'] = gr.Gallery(label="Image Output", show_label=False, object_fit="contain", height=488, visible=True, interactive=False)
                components['output_video'] = gr.Video(label="Result Video", show_label=False, visible=False, interactive=False)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_gallery'], components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    def update_input_visibility(choice):
        is_image = choice == "Image"
        return {
            components['input_image']: gr.update(visible=is_image),
            components['input_video']: gr.update(visible=not is_image),
            components['output_gallery']: gr.update(visible=is_image),
            components['output_video']: gr.update(visible=not is_image),
        }
    components['input_type'].change(fn=update_input_visibility, inputs=[components['input_type']], outputs=list(update_input_visibility("Image").keys()), show_api=False)

def save_temp_file(file_obj, name_prefix: str, is_video=False) -> str:
    if file_obj is None: return None
    if is_video:
        ext = os.path.splitext(file_obj)[1] or ".mp4"
        temp_filename = f"temp_{name_prefix}_{random.randint(1000, 9999)}{ext}"
        save_path = os.path.join(COMFYUI_INPUT_PATH, temp_filename)
        shutil.copy(file_obj, save_path)
    else:
        temp_filename = f"temp_{name_prefix}_{random.randint(1000, 9999)}.png"
        save_path = os.path.join(COMFYUI_INPUT_PATH, temp_filename)
        file_obj.save(save_path, "PNG")
    print(f"Saved temporary input file to: {save_path}")
    return temp_filename

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    is_video = local_ui_values.get('input_type') == "Video"
    
    input_file_obj = local_ui_values.get('input_video') if is_video else local_ui_values.get('input_image')
    if input_file_obj is None:
        raise gr.Error(f"Please provide an input {local_ui_values.get('input_type').lower()}.")
    
    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    if is_video:
        local_ui_values['input_video_filename'] = save_temp_file(input_file_obj, "depth_input", is_video=True)
        metadata = get_media_metadata(input_file_obj, is_video=True)
        local_ui_values['fps'] = metadata.get('fps', 30)
    else:
        local_ui_values['input_image_filename'] = save_temp_file(input_file_obj, "depth_input", is_video=False)
    
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
            
            yield (status, final_files)

    except Exception as e:
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return

    yield ("Status: Loaded successfully!", final_files)