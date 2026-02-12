import gradio as gr
import uuid
import json
import urllib.parse
import websocket
import os
import requests
from io import BytesIO
import base64
import tempfile

from .ace_step_music2music_logic import process_inputs
from core.comfy_api import queue_prompt
from core.backend_manager import backend_manager
from core.config import SERVER_PORT, GRADIO_SERVER_NAME, COMFYUI_OUTPUT_PATH

def _download_and_save_audio(audio_url: str = None, audio_data: str = None) -> str:
    temp_file = None
    try:
        if audio_url:
            response = requests.get(audio_url, timeout=20)
            response.raise_for_status()
            audio_bytes = response.content
        elif audio_data:
            if "," in audio_data:
                audio_data = audio_data.split(',')[1]
            audio_bytes = base64.b64decode(audio_data)
        else:
            raise ValueError("Either 'audio_url' or 'audio_data' must be provided.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_file = f.name
        return temp_file
    except Exception as e:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        raise RuntimeError(f"Failed to process input audio. Error: {e}")

def AudioGen_music2music(
    tags: str,
    audio_url: str = None,
    audio_data: str = None,
    lyrics: str = "[instrumental]",
    similarity: float = 0.7,
    negative_prompt: str = ""
) -> str:
    """
    Re-composes a music clip based on a text description, optional lyrics, and an initial audio file.
    You must provide either 'audio_url' or 'audio_data' for the initial audio.

    Args:
        tags (str): A detailed description of the desired changes in style, genre, instruments, etc.
        audio_url (str, optional): The public URL of the initial audio file.
        audio_data (str, optional): The base64 encoded string of the initial audio file.
        lyrics (str, optional): The lyrics for the song. Use "[instrumental]" for music without vocals. Defaults to "[instrumental]".
        similarity (float): How similar the output should be to the original audio. Value between 0.0 (very different) and 1.0 (very similar). Defaults to 0.7.
        negative_prompt (str, optional): A description of what to avoid in the audio. Defaults to "".
    
    Returns:
        str: A publicly accessible URL to the generated audio file.
    """
    print(f"[MCP Music2Music] Received request. Tags: {tags}")

    temp_audio_path = None
    try:
        temp_audio_path = _download_and_save_audio(audio_url=audio_url, audio_data=audio_data)

        params = {
            'tags': tags,
            'lyrics': lyrics,
            'input_audio': temp_audio_path,
            'similarity': similarity,
            'negative_prompt': negative_prompt,
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
                                
                                print(f"[MCP Music2Music] Generation complete. Returning URL: {final_url}")
                                return final_url
                
                if message.get('type') == 'status' and message.get('data', {}).get('status', {}).get('exec_info', {}).get('queue_remaining') == 0:
                    break
        finally:
            if ws: ws.close()
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    raise RuntimeError("Audio generation failed; no output file was reported by the backend.")

MCP_FUNCTIONS = [AudioGen_music2music]