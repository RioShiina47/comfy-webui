import gradio as gr
import uuid
import json
import urllib.parse
import websocket
import os
import requests
import shutil
from PIL import Image
from io import BytesIO
import base64

from .wan2_2_img2video_logic import process_inputs
from core.comfy_api import queue_prompt
from core.backend_manager import backend_manager
from core.config import SERVER_PORT, GRADIO_SERVER_NAME, COMFYUI_OUTPUT_PATH
from core.workflow_utils import get_filename_prefix

def _download_and_save_image(image_url: str) -> Image.Image:
    try:
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download image from URL: {image_url}. Error: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to process image from URL: {image_url}. Error: {e}")

def VideoGen_img2video(
    prompt: str,
    image_url: str = None,
    image_data: str = None,
    negative_prompt: str = "",
    width: int = 1280,
    height: int = 720,
    video_length_in_frames: int = 81
) -> str:
    """
    Generates a short video clip from an initial image and a text description of the desired motion. The output video will have a frame rate of 16 FPS.
    You must provide either 'image_url' or 'image_data' for the initial image.

    Example resolutions for common aspect ratios:
        "16:9": (1280, 720)
        "9:16": (720, 1280)
        "1:1": (960, 960)
        "4:3": (1088, 816)
        "3:4": (816, 1088)
        "3:2": (1152, 768)
        "2:3": (768, 1152)

    Args:
        prompt (str): A detailed description of the desired motion or action in the video.
        image_url (str, optional): The public URL of the starting image.
        image_data (str, optional): The base64 encoded string of the starting image.
        negative_prompt (str): A description of what to avoid in the video.
        width (int): The width of the generated video in pixels. Defaults to 1280.
        height (int): The height of the generated video in pixels. Defaults to 720.
        video_length_in_frames (int): The number of frames for the output video. Must be between 8 and 81. Defaults to 81.
    
    Returns:
        str: A publicly accessible URL to the generated video file.
    """
    print(f"[MCP Img2Video] Received request. Prompt: {prompt}")

    input_image_pil = None
    if image_url:
        print(f"  - Image source: URL ({image_url})")
        input_image_pil = _download_and_save_image(image_url)
    elif image_data:
        print("  - Image source: Base64 data")
        try:
            if "," in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            input_image_pil = Image.open(BytesIO(image_bytes))
        except Exception as e:
            raise RuntimeError(f"Failed to decode base64 image data. Error: {e}")
    else:
        raise ValueError("Either 'image_url' or 'image_data' must be provided.")

    params = {
        'positive_prompt': prompt,
        'negative_prompt': negative_prompt,
        'start_image': input_image_pil,
        'width': width,
        'height': height,
        'video_length': video_length_in_frames,
        'seed': -1,
    }

    workflow, extra_data = process_inputs(params)
    
    client_id = uuid.uuid4().hex
    
    prompt_response = queue_prompt(workflow, client_id, extra_data)
    if not prompt_response or 'prompt_id' not in prompt_response:
        active_url = backend_manager.get_active_backend_url()
        raise RuntimeError(f"Failed to queue prompt to ComfyUI backend at {active_url}.")
        
    prompt_id = prompt_response['prompt_id']

    active_url = backend_manager.get_active_backend_url()
    ws_url = f"ws://{urllib.parse.urlparse(active_url).netloc}/ws?clientId={client_id}"
    ws = None
    try:
        ws = websocket.create_connection(ws_url)
        while True:
            out = ws.recv()
            if not isinstance(out, str): continue
            
            message = json.loads(out)
            if message.get('type') == 'executed' and (data := message.get('data', {})):
                if data.get('prompt_id') == prompt_id and (output_data := data.get('output')):
                    for value in output_data.values():
                        if isinstance(value, list) and value and isinstance(value[0], dict) and 'filename' in value[0]:
                            output_info = value[0]
                            filename = output_info['filename']
                            subfolder = output_info.get('subfolder', '')
                            
                            absolute_path = os.path.join(COMFYUI_OUTPUT_PATH, subfolder, filename)
                            
                            base_url = f"http://{GRADIO_SERVER_NAME}:{SERVER_PORT}"
                            final_url = f"{base_url}/gradio_api/file={urllib.parse.quote(absolute_path)}"
                            
                            print(f"[MCP Img2Video] Generation complete. Returning URL: {final_url}")
                            return final_url
            
            if message.get('type') == 'status' and message.get('data', {}).get('status', {}).get('exec_info', {}).get('queue_remaining') == 0:
                break
    except Exception as e:
        raise RuntimeError(f"An error occurred while waiting for generation result: {e}")
    finally:
        if ws: ws.close()
    
    raise RuntimeError("Video generation failed; no output file was reported by the backend.")

MCP_FUNCTIONS = [VideoGen_img2video]