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
    "workflow_recipe": "rmbg_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "RMBG",
    "run_button_text": "🎨 Remove Background"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Remove Background (RMBG)")
        gr.Markdown("💡 **Tip:** Upload an image or video to remove its background. You can choose different models and fine-tune the mask.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_type'] = gr.Radio(["Image", "Video"], label="Input Type", value="Image")
                components['input_image'] = gr.Image(type="pil", label="Input Image", visible=True)
                components['input_video'] = gr.Video(label="Input Video", visible=False)
                
                with gr.Accordion("Advanced Settings", open=True):
                    components['model'] = gr.Dropdown(
                        label="Model", 
                        choices=["RMBG-2.0", "INSPYRENET", "BEN", "BEN2"], 
                        value="RMBG-2.0"
                    )
                    components['process_res'] = gr.Slider(label="Processing Resolution", minimum=256, maximum=2048, step=64, value=1024)
                    components['sensitivity'] = gr.Slider(label="Sensitivity", minimum=0, maximum=10, step=1, value=1)
                    components['mask_blur'] = gr.Slider(label="Mask Blur", minimum=0, maximum=50, step=1, value=0)
                    components['mask_offset'] = gr.Slider(label="Mask Offset", minimum=-50, maximum=50, step=1, value=0)
                    components['refine_foreground'] = gr.Checkbox(label="Refine Foreground", value=False)
                    components['invert_output'] = gr.Checkbox(label="Invert Output", value=False)
                    components['background'] = gr.Radio(
                        label="Background Type",
                        choices=["Alpha", "Solid Color"],
                        value="Alpha"
                    )
                    components['background_color'] = gr.ColorPicker(label="Background Color", value="#222222")

            with gr.Column(scale=1):
                gr.Markdown("### Result")
                components['output_gallery'] = gr.Gallery(label="Image Output", show_label=False, object_fit="contain", height=488, visible=True, interactive=False, columns=2, preview=True)
                components['output_video_result'] = gr.Video(label="Result Video", show_label=True, visible=False, interactive=False)
                components['output_video_mask'] = gr.Video(label="Mask Video", show_label=True, visible=False, interactive=False)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_gallery'], components['output_video_result'], components['output_video_mask'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    def update_input_visibility(choice):
        is_image = choice == "Image"
        return {
            components['input_image']: gr.update(visible=is_image),
            components['input_video']: gr.update(visible=not is_image),
            components['output_gallery']: gr.update(visible=is_image),
            components['output_video_result']: gr.update(visible=not is_image),
            components['output_video_mask']: gr.update(visible=not is_image),
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
    
    unique_prefix = get_filename_prefix()
    local_ui_values['filename_prefix_result'] = f"{unique_prefix}_result"
    local_ui_values['filename_prefix_mask'] = f"{unique_prefix}_mask"
    
    if is_video:
        local_ui_values['input_video_filename'] = save_temp_file(input_file_obj, "rmbg_input", is_video=True)
        metadata = get_media_metadata(input_file_obj, is_video=True)
        local_ui_values['fps'] = metadata.get('fps', 30)
    else:
        local_ui_values['input_image_filename'] = save_temp_file(input_file_obj, "rmbg_input", is_video=False)
        
    recipe_path = UI_INFO["workflow_recipe"]
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)

    if is_video:
        workflow[assembler.node_map['rmbg_node']]['inputs']['image'] = [assembler.node_map['get_frames'], 0]
    else:
        workflow[assembler.node_map['rmbg_node']]['inputs']['image'] = [assembler.node_map['load_image'], 0]
        
    return workflow, None

def run_generation(ui_values):
    final_files = []
    try:
        yield ("Status: Preparing...", None)
        
        workflow, extra_data = process_inputs(ui_values)
        workflow_package = (workflow, extra_data)
        
        for status, output_files in run_workflow_and_get_output(workflow_package):
            if output_files and isinstance(output_files, list):
                new_files = [f for f in output_files if f not in final_files]
                if new_files:
                    final_files.extend(new_files)
            
            yield (status, final_files)

    except Exception as e:
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return

    yield ("Status: Loaded successfully!", final_files)