import gradio as gr

from .minimax_music_3_logic import process_inputs
from core.comfy_api import execute_workflow_and_wait, resolve_output_file_url


def AudioGen_MiniMax_Music_3(
    caption: str,
    lyrics: str = "",
    max_duration: int = 60,
    cfg_scale: float = 1.7,
    top_k: int = 50,
    steps: int = 30,
    seed: int = -1,
    request: gr.Request = None
) -> str:
    """
    Generates a music clip from a style description and optional structured lyrics using the MiniMax Music-3 model.

    Args:
        caption (str): Description of the music style, genre, mood, instruments, and vocal traits (e.g., "Female Vocals, Pop ballad, Emotional, Piano, Strings, 120 BPM").
        lyrics (str, optional): Structured song lyrics with section tags like [Intro], [Verse], [Chorus], [Outro]. Leave empty for instrumental music. Defaults to "".
        max_duration (int): Duration of the generated audio in seconds (between 5 and 300). Defaults to 60.
        cfg_scale (float): Classifier-Free Guidance scale (between 1.0 and 10.0). Defaults to 1.7.
        top_k (int): Top-K sampling parameter for text encoder (between 1 and 100). Defaults to 50.
        steps (int): Sampling steps (between 10 and 100). Defaults to 30.
        seed (int): Seed for random generation (-1 for random). Defaults to -1.
    
    Returns:
        str: A publicly accessible URL to the generated audio file.
    """
    print(f"[MCP MiniMax Music 3] Received request. Caption: {caption}")
    
    params = {
        'caption': caption,
        'lyrics': lyrics,
        'max_duration': max_duration,
        'cfg_scale': cfg_scale,
        'top_k': top_k,
        'steps': steps,
        'seed': seed,
        'batch_size': 1,
        'sampler_name': "euler",
        'scheduler': "simple",
        'tile_size': 1536,
        'overlap': 64,
    }

    workflow, extra_data = process_inputs(params)
    
    result = execute_workflow_and_wait((workflow, extra_data))

    output_files = result.get('output_files_info', [])
    downloaded_files = result.get('files', [])

    if not output_files:
        raise RuntimeError("Audio generation failed; no output file was reported by the backend.")

    local_download = downloaded_files[0] if downloaded_files else None
    final_url = resolve_output_file_url(output_files[0], request=request, local_download_path=local_download)

    print(f"[MCP MiniMax Music 3] Generation complete. Returning URL: {final_url}")
    return final_url


MCP_FUNCTIONS = [AudioGen_MiniMax_Music_3]
