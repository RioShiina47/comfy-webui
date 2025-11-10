import gradio as gr
import uuid
import json
import urllib.parse
import websocket
import os
import requests
from PIL import Image
from io import BytesIO
import base64
import time

from .qwen_vl_logic import process_inputs
from core.comfy_api import queue_prompt
from core.backend_manager import backend_manager
from core.config import SERVER_PORT, GRADIO_SERVER_NAME, COMFYUI_OUTPUT_PATH

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

def Vision_Query(
    image_url: str = None,
    image_data: str = None,
    custom_prompt: str = "Describe this image in detail.",
    thinking: bool = False
) -> str:
    """
    Analyzes an image and answers a question or follows an instruction based on its visual content, acting as the vision module for an agent.
    You must provide either 'image_url' or 'image_data'.

    Args:
        image_url (str, optional): The public URL of the input image to be analyzed.
        image_data (str, optional): The base64 encoded string of the input image to be analyzed.
        custom_prompt (str): The specific question or instruction for the model regarding the image. For example: "Describe this image in detail.", "What is the main subject in this photo?", "Write a short story inspired by this image.". Defaults to "Describe this image in detail.".
        thinking (bool, optional): If True, uses the 'Thinking' model variant, which is better for complex reasoning and chain-of-thought tasks. Defaults to False, which uses the standard 'Instruct' model.
    
    Returns:
        str: The generated text response from the model.
    """
    print(f"[MCP Vision_Query] Received request.")
    
    input_image_pil = _download_and_decode_image(image_url=image_url, image_data=image_data)

    params = {
        'input_image': input_image_pil,
        'preset_prompt': "Custom",
        'custom_prompt': custom_prompt,
        'model_mode': "Thinking" if thinking else "Instruct",
        'quantization': "None (FP16)",
        'max_tokens': 1024,
        'keep_model_loaded': False,
    }

    workflow, extra_data = process_inputs(params)
    expected_text_file_path = extra_data.get("expected_text_file_path")
    
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
            if message.get('type') == 'status' and message.get('data', {}).get('status', {}).get('exec_info', {}).get('queue_remaining') == 0:
                break
    finally:
        if ws: ws.close()

    text_content = None
    try:
        for _ in range(10): 
            if expected_text_file_path and os.path.exists(expected_text_file_path):
                with open(expected_text_file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
                break
            time.sleep(0.5)

        if text_content is None:
            raise RuntimeError("Description generation failed; output text file was not found.")

        print(f"[MCP Vision_Query] Generation complete.")
        return text_content
    finally:
        if expected_text_file_path and os.path.exists(expected_text_file_path):
            os.remove(expected_text_file_path)

MCP_FUNCTIONS = [Vision_Query]