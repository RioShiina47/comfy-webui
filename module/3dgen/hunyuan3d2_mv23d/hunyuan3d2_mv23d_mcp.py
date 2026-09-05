import gradio as gr
import os
import requests
from PIL import Image
from io import BytesIO
import base64
import time

from .hunyuan3d2_mv23d_logic import process_inputs
from core.backend_manager import backend_manager
from core.comfy_api import execute_workflow_and_wait, format_gradio_file_url


def _download_and_decode_image(image_url: str = None, image_data: str = None) -> Image.Image:
    try:
        if image_url:
            response = requests.get(image_url, timeout=20)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        elif image_data:
            if "," in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            return Image.open(BytesIO(image_bytes))
        else:
            raise ValueError("Either 'image_url' or 'image_data' must be provided for an image.")
    except Exception as e:
        raise RuntimeError(f"Failed to process input image. Error: {e}")


def ModelGen_multiview2model(
    front_image_url: str = None,
    front_image_data: str = None,
    back_image_url: str = None,
    back_image_data: str = None,
    left_image_url: str = None,
    left_image_data: str = None,
    request: gr.Request = None
) -> dict[str, str]:
    """
    Generates a 3D model from three input images (front, back, left) using the Hunyuan3D-2mv model.
    You must provide either a URL or base64 data for each of the three views.

    Args:
        front_image_url (str, optional): The public URL of the front view image.
        front_image_data (str, optional): The base64 encoded string of the front view image.
        back_image_url (str, optional): The public URL of the back view image.
        back_image_data (str, optional): The base64 encoded string of the back view image.
        left_image_url (str, optional): The public URL of the left view image.
        left_image_data (str, optional): The base64 encoded string of the left view image.
    
    Returns:
        dict[str, str]: A dictionary containing publicly accessible URLs to the generated 3D model files ('shape_model_url' and 'textured_model_url').
    """
    print(f"[MCP MultiView2Model] Received request.")
    
    front_img = _download_and_decode_image(image_url=front_image_url, image_data=front_image_data)
    back_img = _download_and_decode_image(image_url=back_image_url, image_data=back_image_data)
    left_img = _download_and_decode_image(image_url=left_image_url, image_data=left_image_data)

    params = {
        'input_front_image': front_img,
        'input_back_image': back_img,
        'input_left_image': left_img,
        'seed': -1,
    }

    backend_manager.switch_backend('3d_backend')

    workflow, extra_data = process_inputs(params)
    expected_files = extra_data.get("expected_files", {})
    
    execute_workflow_and_wait((workflow, extra_data))

    found_all = False
    for _ in range(10):
        if os.path.exists(expected_files.get("shape")) and os.path.exists(expected_files.get("textured")):
            found_all = True
            break
        time.sleep(1)

    if not found_all:
        raise RuntimeError("3D model generation failed; output files were not found after execution.")

    shape_url = format_gradio_file_url(expected_files['shape'], request)
    textured_url = format_gradio_file_url(expected_files['textured'], request)
    
    result = {
        "shape_model_url": shape_url,
        "textured_model_url": textured_url
    }
    
    print(f"[MCP MultiView2Model] Generation complete. Returning URLs: {result}")
    return result


MCP_FUNCTIONS = [ModelGen_multiview2model]