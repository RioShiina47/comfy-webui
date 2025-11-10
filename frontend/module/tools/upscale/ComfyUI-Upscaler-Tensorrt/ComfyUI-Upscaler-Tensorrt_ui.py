# FILE: ui/tools/upscale_ui.py

import gradio as gr
import random
import os
import shutil
import tempfile
import subprocess
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix

UI_INFO = {
    "workflow_recipe": "Upscaler_Tensorrt_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "Upscaler-Tensorrt",
    "run_button_text": "🚀 Upscale"
}

UPSCALE_MODELS = [
    "4x-AnimeSharp",
    "4x-UltraSharp",
    "4x-WTP-UDS-Esrgan",
    "4x_NMKD-Siax_200k",
    "4x_RealisticRescaler_100000_G",
    "4x_foolhardy_Remacri",
    "RealESRGAN_x4",
    "4xNomos2_otf_esrgan",
    "4x_UniversalUpscalerV2-Neutral_115000_swaG",
    "4x-ClearRealityV1",
    "4x-UltraSharpV2_Lite"
]

MAX_DIMENSION = 1920

def extract_audio_ffmpeg(video_path):
    if not video_path: return None
    audio_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".aac").name
    command = ['ffmpeg', '-y', '-i', video_path, '-vn', '-c:a', 'aac', audio_output_file]
    try:
        print(f"Extracting audio from {video_path}...")
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Audio extracted to {audio_output_file}")
        return audio_output_file
    except FileNotFoundError:
        gr.Warning("ffmpeg not found. Audio will not be preserved.")
        return None
    except subprocess.CalledProcessError:
        print("No audio stream found in the video or an error occurred.")
        return None
    except Exception as e:
        print(f"An exception occurred during audio extraction: {e}")
        return None

def merge_video_audio_ffmpeg(video_path, audio_path):
    if not video_path or not audio_path: return video_path
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    command = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-shortest', output_file]
    try:
        print("Merging video and audio...")
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Final video saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"An exception occurred while running ffmpeg for merging: {e}")
        gr.Warning("Failed to merge audio back into the video. Returning silent video.")
        return video_path

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Upscaler (TensorRT)")
        gr.Markdown("💡 **Tip:** Upload an image or video to upscale. Inputs larger than 1920x1920 will be downscaled first to maintain compatibility.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_type'] = gr.Radio(["Image", "Video"], label="Input Type", value="Image")
                components['input_image'] = gr.Image(type="pil", label="Input Image", visible=True)
                components['input_video'] = gr.Video(label="Input Video", visible=False)
                components['upscaler_model'] = gr.State(value="4x-UltraSharpV2_Lite")
                components['resize_to'] = gr.Dropdown(
                    label="Resize Output To", 
                    choices=["none", "HD", "FHD", "2k", "4k", "2x", "3x"], 
                    value="none",
                    info="Resize the final upscaled output to a specific size or multiplier."
                )
                components['precision'] = gr.State(value="fp16")
                
            with gr.Column(scale=1):
                components['output_gallery'] = gr.Gallery(label="Image Output", show_label=False, object_fit="contain", height=508, visible=True, interactive=False)
                components['output_video'] = gr.Video(label="Video Output", show_label=False, height=488, visible=False, interactive=False)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_gallery'], components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    def update_input_visibility(choice):
        is_image = choice == "Image"
        resize_to_default = "none" if is_image else "4k"
        return {
            components['input_image']: gr.update(visible=is_image),
            components['input_video']: gr.update(visible=not is_image),
            components['output_gallery']: gr.update(visible=is_image),
            components['output_video']: gr.update(visible=not is_image),
            components['resize_to']: gr.update(value=resize_to_default)
        }
    components['input_type'].change(fn=update_input_visibility, inputs=[components['input_type']], outputs=list(update_input_visibility("Image").keys()), show_api=False)

def save_temp_file(file_obj, name_prefix: str, is_video=False):
    if file_obj is None: return None
    if is_video:
        ext = os.path.splitext(file_obj)[1] or ".mp4"
        temp_filename = f"temp_{name_prefix}_{random.randint(1000, 9999)}{ext}"
        save_path = os.path.join(COMFYUI_INPUT_PATH, temp_filename)
        shutil.copy(file_obj, save_path)
    else: # is image
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

    metadata = get_media_metadata(input_file_obj, is_video=is_video)
    width, height = metadata['width'], metadata['height']
    if width == 0 or height == 0:
        raise gr.Error("Could not get the dimensions of the input file.")

    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        aspect_ratio = width / height
        if width > height:
            new_width = MAX_DIMENSION
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = MAX_DIMENSION
            new_width = int(new_height * aspect_ratio)
        gr.Info(f"Input {width}x{height} is too large. Downscaling to {new_width}x{new_height} before upscaling.")
        local_ui_values['downscale_width'] = new_width
        local_ui_values['downscale_height'] = new_height
    else:
        local_ui_values['downscale_width'] = width
        local_ui_values['downscale_height'] = height

    if is_video:
        local_ui_values['input_video_filename'] = save_temp_file(input_file_obj, "upscale_input", is_video=True)
        local_ui_values['output_fps'] = metadata.get('fps', 24)
    else:
        local_ui_values['input_image_filename'] = save_temp_file(input_file_obj, "upscale_input", is_video=False)

    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    recipe_path = UI_INFO["workflow_recipe"]
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None

def run_generation(ui_values):
    original_run_button_text = UI_INFO["run_button_text"]
    
    yield ("Status: Preparing...", None, None, gr.update(value="Stop", variant="stop"))
    
    is_video = ui_values.get('input_type') == "Video"
    input_file_path = ui_values.get('input_video') if is_video else ui_values.get('input_image')
    if not input_file_path:
        raise gr.Error("Input file is missing.")

    temp_audio_path = None
    silent_output_path = None
    final_output_path = None
    all_output_files = []
    
    try:
        if is_video:
            yield ("Status: Extracting audio...", gr.update(), None, gr.update())
            temp_audio_path = extract_audio_ffmpeg(input_file_path)

        workflow, extra_data = process_inputs(ui_values)
        workflow_package = (workflow, extra_data)
        
        for status, output_files in run_workflow_and_get_output(workflow_package):
            if output_files and isinstance(output_files, list):
                all_output_files = output_files
            yield (status, all_output_files, gr.update())

        if is_video and all_output_files:
            silent_output_path = all_output_files[0]
            yield ("Status: Merging audio...", gr.update(), silent_output_path, gr.update())
            final_output_path = merge_video_audio_ffmpeg(silent_output_path, temp_audio_path)
            yield ("Status: Merging complete!", [final_output_path], gr.update())
        else:
            final_output_path = all_output_files[0] if all_output_files else None

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (f"Error: {e}", None, None, gr.update(value=original_run_button_text, variant="primary"))

    finally:
        print("Upscale task finished. Cleaning up temporary files...")
        cleanup_paths = [temp_audio_path]
        if is_video and temp_audio_path and silent_output_path != final_output_path:
            cleanup_paths.append(silent_output_path)
            
        for path in cleanup_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Removed temp file: {path}")
                except Exception as e:
                    print(f"Error removing temp file {path}: {e}")
        
        yield ("Status: Ready", gr.update(), gr.update(), gr.update(value=original_run_button_text, variant="primary"))