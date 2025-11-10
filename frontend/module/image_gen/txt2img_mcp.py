import gradio as gr
import uuid
import json
import urllib.parse
import websocket
import os

from .image_gen_logic import process_inputs
from core.comfy_api import queue_prompt
from core.backend_manager import backend_manager
from core.config import SERVER_PORT, GRADIO_SERVER_NAME, COMFYUI_OUTPUT_PATH

def ImageGen_txt2img(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1328,
    height: int = 1328,
) -> str:
    """
    Generates an image from a text description. For best results, the recommended total resolution is close to 1328x1328.

    Example resolutions for common aspect ratios:
        "1:1": (1328, 1328)
        "16:9": (1664, 928)
        "9:16": (928, 1664)
        "4:3": (1472, 1104)
        "3:4": (1104, 1472)
        "3:2": (1584, 1056)
        "2:3": (1056, 1584)

    Args:
        prompt (str): A detailed description of the image content.
        negative_prompt (str): A description of what to avoid in the image.
        width (int): The width of the generated image in pixels. Defaults to 1328.
        height (int): The height of the generated image in pixels. Defaults to 1328.
    
    Returns:
        str: A publicly accessible URL to the generated image.
    """
    ui_values = {
        'txt2img_positive_prompt': prompt,
        'txt2img_negative_prompt': negative_prompt,
        'txt2img_width': width,
        'txt2img_height': height,

        'txt2img_model_name': "QwenLM/Qwen-Image",
        'txt2img_steps': 8,
        'txt2img_cfg': 1.0,
        'txt2img_sampler_name': "euler",
        'txt2img_scheduler': "simple",
        'txt2img_seed': -1, 

        'txt2img_model_type_state': 'qwen-image',
        'txt2img_batch_count': 1,
        'txt2img_batch_size': 1,

        'txt2img_lora_count_state': 0,
        'txt2img_controlnet_count_state': 0,
        'txt2img_ipadapter_count_state': 0,
        'txt2img_embedding_count_state': 0,
        'txt2img_style_count_state': 0,
        'txt2img_conditioning_count_state': 0,
        'txt2img_vae_source': 'None'
    }

    workflow, extra_data = process_inputs('txt2img', ui_values)
    
    client_id = uuid.uuid4().hex
    
    prompt_response = queue_prompt(workflow, client_id, extra_data)
    if not prompt_response or 'prompt_id' not in prompt_response:
        active_url = backend_manager.get_active_backend_url()
        raise RuntimeError(f"Failed to queue prompt to the ComfyUI backend at {active_url}.")
        
    prompt_id = prompt_response['prompt_id']

    active_url = backend_manager.get_active_backend_url()
    ws_url = f"ws://{urllib.parse.urlparse(active_url).netloc}/ws?clientId={client_id}"
    ws = None
    try:
        ws = websocket.create_connection(ws_url)
        while True:
            out = ws.recv()
            if not isinstance(out, str):
                continue
            
            message = json.loads(out)
            msg_type = message.get('type')

            if msg_type == 'status' and message.get('data', {}).get('status', {}).get('exec_info', {}).get('queue_remaining') == 0:
                break

            elif msg_type == 'executed':
                data = message.get('data', {})
                if data.get('prompt_id') == prompt_id:
                    output_data = data.get('output', {})
                    for key, value in output_data.items():
                        if isinstance(value, list) and value and isinstance(value[0], dict) and 'filename' in value[0]:
                            output_info = value[0]
                            filename = output_info['filename']
                            subfolder = output_info.get('subfolder', '')
                            
                            absolute_path = os.path.join(COMFYUI_OUTPUT_PATH, subfolder, filename)
                            
                            base_url = f"http://{GRADIO_SERVER_NAME}:{SERVER_PORT}"
                            final_url = f"{base_url}/gradio_api/file={urllib.parse.quote(absolute_path)}"
                            
                            print(f"[MCP Txt2Img Tool] Generation complete. Returning URL: {final_url}")
                            return final_url
    except Exception as e:
        raise RuntimeError(f"An error occurred while waiting for the generation result: {e}")
    finally:
        if ws:
            ws.close()
    
    raise RuntimeError("Image generation failed; the backend did not report any output files.")

MCP_FUNCTIONS = [ImageGen_txt2img]