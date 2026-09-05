import gradio as gr

from .ace_step_txt2music_logic import process_inputs
from core.comfy_api import execute_workflow_and_wait, resolve_output_file_url


def AudioGen_txt2music(
    tags: str,
    lyrics: str = "[instrumental]",
    seconds: int = 30,
    negative_prompt: str = "",
    request: gr.Request = None
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
    
    result = execute_workflow_and_wait((workflow, extra_data))

    output_files = result.get('output_files_info', [])
    downloaded_files = result.get('files', [])

    if not output_files:
        raise RuntimeError("Audio generation failed; no output file was reported by the backend.")

    local_download = downloaded_files[0] if downloaded_files else None
    final_url = resolve_output_file_url(output_files[0], request=request, local_download_path=local_download)

    print(f"[MCP Txt2Music] Generation complete. Returning URL: {final_url}")
    return final_url


MCP_FUNCTIONS = [AudioGen_txt2music]