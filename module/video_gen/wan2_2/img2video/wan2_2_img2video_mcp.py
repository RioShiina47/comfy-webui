import gradio as gr
import os
import requests
from PIL import Image
from io import BytesIO
import base64

from .wan2_2_img2video_logic import process_inputs
from core.comfy_api import execute_workflow_and_wait, resolve_output_file_url


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
    video_length_in_frames: int = 81,
    request: gr.Request = None
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
    
    result = execute_workflow_and_wait((workflow, extra_data))

    output_files = result.get('output_files_info', [])
    downloaded_files = result.get('files', [])

    if not output_files:
        raise RuntimeError("Video generation failed; no output file was reported by the backend.")

    local_download = downloaded_files[0] if downloaded_files else None
    final_url = resolve_output_file_url(output_files[0], request=request, local_download_path=local_download)

    print(f"[MCP Img2Video] Generation complete. Returning URL: {final_url}")
    return final_url


MCP_FUNCTIONS = [VideoGen_img2video]