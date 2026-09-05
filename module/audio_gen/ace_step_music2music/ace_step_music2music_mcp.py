import gradio as gr
import os
import requests
import base64
import tempfile

from .ace_step_music2music_logic import process_inputs
from core.comfy_api import execute_workflow_and_wait, resolve_output_file_url


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
    negative_prompt: str = "",
    request: gr.Request = None
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
        
        result = execute_workflow_and_wait((workflow, extra_data))

        output_files = result.get('output_files_info', [])
        downloaded_files = result.get('files', [])

        if not output_files:
            raise RuntimeError("Audio generation failed; no output file was reported by the backend.")

        local_download = downloaded_files[0] if downloaded_files else None
        final_url = resolve_output_file_url(output_files[0], request=request, local_download_path=local_download)

        print(f"[MCP Music2Music] Generation complete. Returning URL: {final_url}")
        return final_url

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


MCP_FUNCTIONS = [AudioGen_music2music]