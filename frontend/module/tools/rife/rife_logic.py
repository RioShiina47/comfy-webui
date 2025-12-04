import os
import tempfile
import subprocess

from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_video

WORKFLOW_RECIPE_PATH = "rife_recipe.yaml"

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
        print("Warning: ffmpeg not found. Audio will not be processed.")
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
        return video_path

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    input_video_path = local_ui_values.get('input_video')
    if not input_video_path:
        raise ValueError("Please upload an input video file.")
        
    local_ui_values['input_video_filename'] = save_temp_video(input_video_path)
    
    metadata = get_media_metadata(input_video_path, is_video=True)
    input_fps = metadata.get('fps', 24)
    multiplier = int(local_ui_values.get('multiplier', 2))
    
    is_slow_motion = "Slow Motion" in local_ui_values.get('fps_mode', "")
    if is_slow_motion:
        local_ui_values['output_fps'] = round(input_fps)
    else:
        local_ui_values['output_fps'] = min(round(input_fps * multiplier), 240)

    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None