import gradio as gr
import uuid
import json
import urllib.parse
import websocket
import os
import requests
import shutil
from PIL import Image
from io import BytesIO
import base64

from .qwen_image_edit_logic import process_inputs_logic
from core.comfy_api import queue_prompt
from core.backend_manager import backend_manager
from core.config import SERVER_PORT, GRADIO_SERVER_NAME, COMFYUI_OUTPUT_PATH
from core.workflow_utils import get_filename_prefix

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

def ImageEdit(
    prompt: str,
    image_url: str = None,
    image_data: str = None,
    reference_image_urls: list[str] = None,
    reference_image_data: list[str] = None,
    negative_prompt: str = "",
    width: int = 1328,
    height: int = 1328,
) -> str:
    """
    Edits an image based on a text instruction using the Qwen-Image-Edit model.
    Accepts one main image and up to 4 additional reference images.
    You must provide either 'image_url' or 'image_data' for the main image.

    Example resolutions for common aspect ratios:
        "1:1 (Square)": (1328, 1328)
        "16:9 (Landscape)": (1664, 928)
        "9:16 (Portrait)": (928, 1664)
        "4:3 (Classic)": (1472, 1104)
        "3:4 (Classic Portrait)": (1104, 1472)
        "3:2 (Photography)": (1536, 1024)
        "2:3 (Photography Portrait)": (1024, 1536)

    Args:
        prompt (str): A detailed description of the desired edit.
        image_url (str, optional): The public URL of the main image to be edited.
        image_data (str, optional): The base64 encoded string of the main image to be edited.
        reference_image_urls (list[str], optional): A list of public URLs for up to 4 reference images.
        reference_image_data (list[str], optional): A list of base64 encoded strings for up to 4 reference images.
        negative_prompt (str): A description of what to avoid in the image.
        width (int): The width of the generated image in pixels. Defaults to 1328.
        height (int): The height of the generated image in pixels. Defaults to 1328.
    
    Returns:
        str: A publicly accessible URL to the generated image file.
    """
    print(f"[MCP ImageEdit] Received request. Prompt: {prompt}")
    
    input_image_pil = None
    if image_url:
        print(f"  - Main image source: URL ({image_url})")
        input_image_pil = _download_and_save_image(image_url)
    elif image_data:
        print("  - Main image source: Base64 data")
        try:
            if "," in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            input_image_pil = Image.open(BytesIO(image_bytes))
        except Exception as e:
            raise RuntimeError(f"Failed to decode base64 image data for main image. Error: {e}")
    else:
        raise ValueError("Either 'image_url' or 'image_data' must be provided for the main image.")

    reference_pils = []
    if reference_image_urls:
        print(f"  - Found {len(reference_image_urls)} reference image URLs.")
        for i, ref_url in enumerate(reference_image_urls[:4]):
            try:
                reference_pils.append(_download_and_save_image(ref_url))
            except Exception as e:
                print(f"  - Warning: Skipping reference image URL {i+1} due to error: {e}")
    
    if reference_image_data:
        print(f"  - Found {len(reference_image_data)} reference image data blobs.")
        for i, ref_data in enumerate(reference_image_data):
            if len(reference_pils) >= 4: break
            try:
                if "," in ref_data:
                    ref_data = ref_data.split(',')[1]
                ref_bytes = base64.b64decode(ref_data)
                reference_pils.append(Image.open(BytesIO(ref_bytes)))
            except Exception as e:
                print(f"  - Warning: Skipping reference image data {i+1} due to error: {e}")

    reference_pils = reference_pils[:4]
    
    params = {
        'positive_prompt': prompt,
        'negative_prompt': negative_prompt,
        'width': width,
        'height': height,
        'seed': -1,
        'model_version': 'Qwen-Image-Edit-2511',
        'input_image': input_image_pil,
        'ref_count_state': len(reference_pils),
        'ref_image_inputs': reference_pils,
        
        'steps': 8,
        'cfg': 1.0,
        'sampler_name': "euler",
        'scheduler': "simple",
        'batch_size': 1,
    }

    workflow, extra_data = process_inputs_logic(params)
    
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
                            
                            print(f"[MCP ImageEdit] Generation complete. Returning URL: {final_url}")
                            return final_url
            
            if message.get('type') == 'status' and message.get('data', {}).get('status', {}).get('exec_info', {}).get('queue_remaining') == 0:
                break
    except Exception as e:
        raise RuntimeError(f"An error occurred while waiting for generation result: {e}")
    finally:
        if ws: ws.close()
    
    raise RuntimeError("Image generation failed; no output file was reported by the backend.")

MCP_FUNCTIONS = [ImageEdit]