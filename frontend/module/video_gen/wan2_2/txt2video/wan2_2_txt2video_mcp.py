import gradio as gr
import random
import os
import yaml
import uuid
import json
import urllib.parse
import websocket

from .wan2_2_txt2video_logic import process_inputs
from core.comfy_api import queue_prompt
from core.backend_manager import backend_manager
from core.config import SERVER_PORT, GRADIO_SERVER_NAME, COMFYUI_OUTPUT_PATH


def VideoGen_txt2video(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1280,
    height: int = 720,
    video_length_in_frames: int = 81
) -> str:
    """
    Generates a short video clip from a text description. The output video will have a frame rate of 16 FPS.

    Example resolutions for common aspect ratios:
        "16:9": (1280, 720)
        "9:16": (720, 1280)
        "1:1": (960, 960)
        "4:3": (1088, 816)
        "3:4": (816, 1088)
        "3:2": (1152, 768)
        "2:3": (768, 1152)

    Args:
        prompt (str): A detailed description of the video content, style, and action.
        negative_prompt (str): A description of what to avoid in the video.
        width (int): The width of the generated video in pixels. Defaults to 1280.
        height (int): The height of the generated video in pixels. Defaults to 720.
        video_length_in_frames (int): The number of frames for the output video. Must be between 8 and 81. Defaults to 81.
    
    Returns:
        str: A publicly accessible URL to the generated video file.
    """
    params = {
        'positive_prompt': prompt,
        'negative_prompt': negative_prompt,
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
            if not isinstance(out, str):
                continue
            
            message = json.loads(out)
            msg_type = message.get('type')

            if msg_type == 'status' and message.get('data', {}).get('status', {}).get('exec_info', {}).get('queue_remaining') == 0:
                break

            elif msg_type == 'executed':
                data = message.get('data', {})
                if data.get('prompt_id') == prompt_id:
                    output_data = data.get('output', {})
                    for key, value in output_data.items():
                        if isinstance(value, list) and value and isinstance(value[0], dict) and 'filename' in value[0]:
                            output_info = value[0]
                            filename = output_info['filename']
                            subfolder = output_info.get('subfolder', '')
                            
                            absolute_path = os.path.join(COMFYUI_OUTPUT_PATH, subfolder, filename)
                            
                            base_url = f"http://{GRADIO_SERVER_NAME}:{SERVER_PORT}"
                            final_url = f"{base_url}/gradio_api/file={urllib.parse.quote(absolute_path)}"
                            
                            print(f"[MCP T2V Tool] Generation complete. Returning URL: {final_url}")
                            return final_url
    except Exception as e:
        raise RuntimeError(f"An error occurred while waiting for generation result: {e}")
    finally:
        if ws:
            ws.close()
    
    raise RuntimeError("Video generation failed; no output file was reported by the backend.")

MCP_FUNCTIONS = [VideoGen_txt2video]