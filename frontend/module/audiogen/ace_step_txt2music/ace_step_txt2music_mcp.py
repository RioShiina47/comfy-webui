import gradio as gr
import uuid
import json
import urllib.parse
import websocket
import os

from .ace_step_txt2music_logic import process_inputs
from core.comfy_api import queue_prompt
from core.backend_manager import backend_manager
from core.config import SERVER_PORT, GRADIO_SERVER_NAME, COMFYUI_OUTPUT_PATH

def AudioGen_txt2music(
    tags: str,
    lyrics: str = "[instrumental]",
    seconds: int = 30,
    negative_prompt: str = ""
) -> str:
    """
    Generates a music clip from a text description and optional lyrics using the ACE-Step model.

    Args:
        tags (str): A detailed description of the music style, genre, instruments, mood, etc. (e.g., "epic, cinematic, orchestral").
        lyrics (str, optional): The lyrics for the song. Use "[instrumental]" for music without vocals. Defaults to "[instrumental]".
        seconds (int): The duration of the generated audio in seconds. Must be between 5 and 300. Defaults to 30.
        negative_prompt (str, optional): A description of what to avoid in the audio. Defaults to "".
    
    Returns:
        str: A publicly accessible URL to the generated audio file.
    """
    print(f"[MCP Txt2Music] Received request. Tags: {tags}")
    
    params = {
        'tags': tags,
        'lyrics': lyrics,
        'seconds': seconds,
        'negative_prompt': negative_prompt,
        'seed': -1,
        'steps': 50,
        'cfg': 5.0,
        'sampler_name': "euler",
        'scheduler': "simple",
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
                            
                            print(f"[MCP Txt2Music] Generation complete. Returning URL: {final_url}")
                            return final_url
            
            if message.get('type') == 'status' and message.get('data', {}).get('status', {}).get('exec_info', {}).get('queue_remaining') == 0:
                break
    except Exception as e:
        raise RuntimeError(f"An error occurred while waiting for generation result: {e}")
    finally:
        if ws: ws.close()
    
    raise RuntimeError("Audio generation failed; no output file was reported by the backend.")

MCP_FUNCTIONS = [AudioGen_txt2music]