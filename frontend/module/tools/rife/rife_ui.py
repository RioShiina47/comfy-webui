import gradio as gr
import random
import os
import shutil
import math
import tempfile
import subprocess
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix

UI_INFO = {
    "workflow_recipe": "rife_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "RIFE",
    "run_button_text": "🚀 Interpolate Video"
}

def extract_audio_ffmpeg(video_path):
    if not video_path:
        return None
    
    audio_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".aac").name
    command = ['ffmpeg', '-i', video_path, '-vn', '-c:a', 'aac', '-y', audio_output_file]
    
    try:
        print(f"Extracting audio from {video_path}...")
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Audio extracted to {audio_output_file}")
        return audio_output_file
    except FileNotFoundError:
        gr.Warning("ffmpeg not found. Audio will not be processed.")
        return None
    except subprocess.CalledProcessError:
        print("No audio stream found in the video or an error occurred.")
        return None
    except Exception as e:
        print(f"An exception occurred during audio extraction: {e}")
        return None

def merge_video_audio_ffmpeg(video_path, audio_path, multiplier, is_slow_motion):
    if not video_path or not audio_path:
        return video_path

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    
    command = ['ffmpeg', '-i', video_path, '-i', audio_path]
    
    if is_slow_motion:
        audio_filter_parts = []
        speed_factor = 1.0 / float(multiplier)
        
        while speed_factor < 0.5:
            audio_filter_parts.append("atempo=0.5")
            speed_factor *= 2.0
        
        if speed_factor < 1.0:
            audio_filter_parts.append(f"atempo={speed_factor}")

        audio_filter_str = ",".join(audio_filter_parts)
        print(f"[ffmpeg] Slow motion: Applying audio filter: '{audio_filter_str}'")
        command.extend(['-filter:a', audio_filter_str, '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_file])

    else:
        print("[ffmpeg] Interpolation: Merging audio without speed change.")
        command.extend(['-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_file])

    try:
        print(f"Merging video and audio. Command: {' '.join(command)}")
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
        gr.Markdown("## RIFE Video Frame Interpolation (TensorRT)")
        gr.Markdown("💡 **Tip:** Upload a video and select the interpolation settings. This uses a pre-built RIFE TensorRT engine (`rife49_ensemble_True_scale_1_sim.engine`) for acceleration.")
        
        components['use_cuda_graph'] = gr.State(True)
        components['keep_model_loaded'] = gr.State(False)
        components['clear_cache_after_n_frames'] = gr.State(100)
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_video'] = gr.Video(label="Input Video")

                components['multiplier'] = gr.Slider(label="Multiplier", minimum=2, maximum=8, step=1, value=2)
                components['fps_mode'] = gr.Radio(
                    label="Framerate Mode",
                    choices=["Keep Original Duration (Interpolate)", "Keep Original Framerate (Slow Motion)"],
                    value="Keep Original Duration (Interpolate)"
                )
                
            with gr.Column(scale=1):
                components['output_video'] = gr.Video(label="Result", show_label=False, interactive=False, height=488)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def save_temp_video_file(video_path):
    if not video_path:
        return None
    
    ext = os.path.splitext(video_path)[1] or ".mp4"
    filename = f"temp_rife_input_{random.randint(1000, 9999)}{ext}"
    save_path = os.path.join(COMFYUI_INPUT_PATH, filename)
    shutil.copy(video_path, save_path)
    print(f"Saved temporary video file to: {save_path}")
    return filename

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    input_video_path = local_ui_values.get('input_video')
    if not input_video_path:
        raise gr.Error("Please upload an input video file.")
        
    local_ui_values['input_video_filename'] = save_temp_video_file(input_video_path)
    
    metadata = get_media_metadata(input_video_path, is_video=True)
    input_fps = metadata.get('fps', 24)
    multiplier = int(local_ui_values.get('multiplier', 2))
    
    is_slow_motion = "Slow Motion" in local_ui_values.get('fps_mode', "")
    if is_slow_motion:
        local_ui_values['output_fps'] = round(input_fps)
    else:
        local_ui_values['output_fps'] = min(round(input_fps * multiplier), 240)

    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    recipe_path = UI_INFO["workflow_recipe"]
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None

def run_generation(ui_values):
    original_run_button_text = UI_INFO["run_button_text"]
    
    yield (
        "Status: Preparing...",
        None,
        gr.update(value="Stop", variant="stop")
    )
    
    input_video_path = ui_values.get('input_video')
    if not input_video_path:
        raise gr.Error("Input video is missing.")

    temp_audio_path = None
    silent_video_path = None
    final_video_path = None
    
    try:
        yield ("Status: Extracting audio...", None, gr.update())
        temp_audio_path = extract_audio_ffmpeg(input_video_path)

        workflow, extra_data = process_inputs(ui_values)
        workflow_package = (workflow, extra_data)
        
        for status, output_files in run_workflow_and_get_output(workflow_package):
            if output_files and isinstance(output_files, list):
                silent_video_path = output_files[0]
            
            yield (status, silent_video_path, gr.update())

        if silent_video_path:
            yield ("Status: Merging audio...", silent_video_path, gr.update())
            is_slow_motion = "Slow Motion" in ui_values.get('fps_mode', "")
            multiplier = int(ui_values.get('multiplier', 2))
            final_video_path = merge_video_audio_ffmpeg(silent_video_path, temp_audio_path, multiplier, is_slow_motion)
            yield ("Status: Merging complete!", final_video_path, gr.update())
        else:
             raise RuntimeError("RIFE workflow did not produce a video file.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (
            f"Error: {e}",
            None,
            gr.update(value=original_run_button_text, variant="primary")
        )

    finally:
        print("RIFE generation task finished. Cleaning up temporary files...")
        cleanup_paths = [temp_audio_path, silent_video_path]
        for path in cleanup_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Removed temp file: {path}")
                except Exception as e:
                    print(f"Error removing temp file {path}: {e}")

        yield (
            "Status: Ready",
            final_video_path or gr.update(),
            gr.update(value=original_run_button_text, variant="primary")
        )