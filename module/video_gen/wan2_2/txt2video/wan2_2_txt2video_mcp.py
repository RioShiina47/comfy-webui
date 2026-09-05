import gradio as gr

from .wan2_2_txt2video_logic import process_inputs
from core.comfy_api import execute_workflow_and_wait, resolve_output_file_url


def VideoGen_txt2video(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1280,
    height: int = 720,
    video_length_in_frames: int = 81,
    request: gr.Request = None
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
    
    result = execute_workflow_and_wait((workflow, extra_data))
    
    output_files = result.get('output_files_info', [])
    downloaded_files = result.get('files', [])

    if not output_files:
        raise RuntimeError("Video generation failed; no output file was reported by the backend.")

    local_download = downloaded_files[0] if downloaded_files else None
    final_url = resolve_output_file_url(output_files[0], request=request, local_download_path=local_download)

    print(f"[MCP T2V Tool] Generation complete. Returning URL: {final_url}")
    return final_url


MCP_FUNCTIONS = [VideoGen_txt2video]