import gradio as gr
import os
import requests
from PIL import Image
from io import BytesIO
import base64
import time
import glob

from .pixal3d_trellis2_img23d_logic import process_inputs
from core.backend_manager import backend_manager
from core.comfy_api import execute_workflow_and_wait, format_gradio_file_url
from core.config import COMFYUI_OUTPUT_PATH


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
            raise ValueError("Either 'image_url' or 'image_data' must be provided.")
    except Exception as e:
        raise RuntimeError(f"Failed to process input image. Error: {e}")


def _3DGen_Pixal3D_TRELLIS_2(
    image_url: str = None,
    image_data: str = None,
    seed: int = -1,
    request: gr.Request = None
) -> dict[str, str]:
    """
    Generates a 3D model from a single input image using TRELLIS-2 / Pixal3D with automatic PBR texture baking.
    You must provide either 'image_url' or 'image_data' for the input image.

    Args:
        image_url (str, optional): The public URL of the input image.
        image_data (str, optional): The base64 encoded string of the input image.
        seed (int, optional): Random seed for reproducible generation. Defaults to -1.
        request (gr.Request, optional): Gradio request object for URL resolution.

    Returns:
        dict[str, str]: A dictionary containing publicly accessible URLs to the generated 3D model files ('textured_model_url' and 'shape_model_url').
    """
    print("[MCP 3DGen Pixal3D TRELLIS 2] Received request.")
    
    input_image_pil = _download_and_decode_image(image_url=image_url, image_data=image_data)

    params = {
        'input_image': input_image_pil,
        'seed': seed,
    }

    backend_manager.switch_backend('default')

    workflow, extra_data = process_inputs(params)
    expected_files = extra_data.get("expected_files", {})
    
    exec_result = execute_workflow_and_wait((workflow, extra_data))
    downloaded_files = exec_result.get('files', []) if isinstance(exec_result, dict) else (exec_result or [])

    shape_file = None
    textured_file = None

    if downloaded_files:
        for f in downloaded_files:
            if isinstance(f, str) and f.endswith(".glb"):
                fname_lower = os.path.basename(f).lower()
                if "shape" in fname_lower and not shape_file:
                    shape_file = f
                elif "textured" in fname_lower and not textured_file:
                    textured_file = f
                elif not textured_file:
                    textured_file = f
                elif not shape_file:
                    shape_file = f

    if not shape_file or not textured_file:
        shape_path = expected_files.get("shape")
        textured_path = expected_files.get("textured")
        prefix_shape = expected_files.get("prefix_shape")
        prefix_textured = expected_files.get("prefix_textured")

        for _ in range(10):
            if not shape_file and shape_path and os.path.exists(shape_path):
                shape_file = shape_path
            if not textured_file and textured_path and os.path.exists(textured_path):
                textured_file = textured_path
            
            if (not shape_file or not textured_file) and COMFYUI_OUTPUT_PATH and os.path.exists(COMFYUI_OUTPUT_PATH):
                if not shape_file and prefix_shape:
                    matches = glob.glob(os.path.join(COMFYUI_OUTPUT_PATH, f"{prefix_shape}*.glb".replace('/', os.sep)))
                    if matches:
                        shape_file = matches[-1]
                if not textured_file and prefix_textured:
                    matches = glob.glob(os.path.join(COMFYUI_OUTPUT_PATH, f"{prefix_textured}*.glb".replace('/', os.sep)))
                    if matches:
                        textured_file = matches[-1]

            if shape_file and textured_file:
                break
            time.sleep(1)

    if not textured_file and not shape_file:
        raise RuntimeError("3D model generation failed; output files were not found after execution.")

    result = {}
    if textured_file:
        result["textured_model_url"] = format_gradio_file_url(textured_file, request)
    if shape_file:
        result["shape_model_url"] = format_gradio_file_url(shape_file, request)

    print(f"[MCP 3DGen Pixal3D TRELLIS 2] Generation complete. Returning URLs: {result}")
    return result


_3DGen_Pixal3D_TRELLIS_2.__name__ = "3DGen_Pixal3D_TRELLIS_2"
globals()["3DGen_Pixal3D_TRELLIS_2"] = _3DGen_Pixal3D_TRELLIS_2
MCP_FUNCTIONS = [_3DGen_Pixal3D_TRELLIS_2]
