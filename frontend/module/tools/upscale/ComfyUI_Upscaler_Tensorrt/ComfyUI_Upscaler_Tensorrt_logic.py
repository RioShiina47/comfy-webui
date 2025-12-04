import os
import tempfile
import subprocess
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, save_temp_video

WORKFLOW_RECIPE_PATH = "Upscaler_Tensorrt_recipe.yaml"
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
        print("Warning: ffmpeg not found. Audio will not be preserved.")
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
        return video_path

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    is_video = local_ui_values.get('input_type') == "Video"
    
    input_file_obj = local_ui_values.get('input_video') if is_video else local_ui_values.get('input_image')
    if input_file_obj is None:
        raise ValueError(f"Please provide an input {local_ui_values.get('input_type').lower()}.")

    metadata = get_media_metadata(input_file_obj, is_video=is_video)
    width, height = metadata['width'], metadata['height']
    if width == 0 or height == 0:
        raise ValueError("Could not get the dimensions of the input file.")

    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        aspect_ratio = width / height
        if width > height:
            new_width = MAX_DIMENSION
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = MAX_DIMENSION
            new_width = int(new_height * aspect_ratio)
        print(f"Info: Input {width}x{height} is too large. Downscaling to {new_width}x{new_height} before upscaling.")
        local_ui_values['downscale_width'] = new_width
        local_ui_values['downscale_height'] = new_height
    else:
        local_ui_values['downscale_width'] = width
        local_ui_values['downscale_height'] = height

    if is_video:
        local_ui_values['input_video_filename'] = save_temp_video(input_file_obj)
        local_ui_values['output_fps'] = metadata.get('fps', 24)
    else:
        local_ui_values['input_image_filename'] = save_temp_image(input_file_obj)

    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None