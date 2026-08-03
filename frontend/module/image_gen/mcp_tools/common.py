"""
MCP Common Utilities & Data Structures
Contains YAML loading utilities, config file paths, task definitions, and async task database.
"""

import os
import time
import urllib.parse
import urllib.request
import uuid
import json
import base64
import io
import yaml
from typing import Dict, Any
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YAML_DIR = os.path.join(_PROJECT_ROOT, "yaml")

_MODEL_ARCHITECTURES_PATH = os.path.join(_YAML_DIR, "model_architectures.yaml")
_MODEL_LIST_PATH = os.path.join(_YAML_DIR, "model_list.yaml")
_MODEL_DEFAULTS_PATH = os.path.join(_YAML_DIR, "model_defaults.yaml")
_IMAGE_GEN_FEATURES_PATH = os.path.join(_YAML_DIR, "image_gen_features.yaml")
_CHAIN_FEATURES_PATH = os.path.join(_YAML_DIR, "chain_features.yaml")
_CONSTANTS_PATH = os.path.join(_YAML_DIR, "constants.yaml")


def _parse_image_param(image_param: Any) -> Any:
    """Parse a Base64 Data URI, local file path, or PIL.Image into a PIL Image object. HTTP URLs are not supported."""
    if isinstance(image_param, Image.Image):
        return image_param

    if not isinstance(image_param, str) or not image_param.strip():
        return None

    image_param = image_param.strip()

    # Reject HTTP / HTTPS URL
    if image_param.startswith("http://") or image_param.startswith("https://"):
        raise ValueError(
            "Image URLs are not supported. Please supply the image directly as a Base64 Data URI (e.g., 'data:image/png;base64,...')."
        )

    # Base64 Data URI (e.g. data:image/png;base64,...)
    if image_param.startswith("data:image/"):
        _, encoded = image_param.split(",", 1) if "," in image_param else ("", image_param)
        data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(data))

    # Base64 string without header
    if len(image_param) > 100 and not os.path.exists(image_param):
        try:
            data = base64.b64decode(image_param)
            return Image.open(io.BytesIO(data))
        except Exception:
            pass

    # Local file path
    if os.path.exists(image_param):
        return Image.open(image_param)

    raise ValueError(
        "Invalid image parameter format. Expected a Base64 Data URI (e.g., 'data:image/png;base64,...') or local file path."
    )


def _load_yaml(filepath: str) -> dict:
    """Safely load a YAML file, returning an empty dict if the file does not exist."""
    if not os.path.exists(filepath):
        print(f"Warning: YAML file not found: {filepath}")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_COMMON_OPTIONAL_INPUTS = [
    "steps", "cfg", "sampler", "scheduler", "seed",
    "negative_prompt", "batch_size", "chain", "async_execution",
]

_TASK_DEFINITIONS = [
    {
        "task_type": "txt2img",
        "display_name": "Text-to-Image",
        "description": "Generate images from text prompts. Canvas width and height must be specified.",
        "required_inputs": ["prompt", "width", "height"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
    },
    {
        "task_type": "img2img",
        "display_name": "Image-to-Image",
        "description": "Perform global repaint and style transfer based on a source image. Denoise strength must be specified.",
        "required_inputs": ["prompt", "image", "denoise"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
    },
    {
        "task_type": "inpaint",
        "display_name": "Inpaint",
        "description": "Repaint specified masked regions of the input image (with alpha mask/channel).",
        "required_inputs": ["prompt", "image"],
        "optional_inputs": ["denoise"] + _COMMON_OPTIONAL_INPUTS,
    },
    {
        "task_type": "outpaint",
        "display_name": "Outpaint",
        "description": "Extend the canvas outward from the source image. Padding pixel values for top, bottom, left, and right must be specified.",
        "required_inputs": ["prompt", "image", "pad_left", "pad_right", "pad_top", "pad_bottom"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
    },
    {
        "task_type": "hires_fix",
        "display_name": "Hi-Res Fix / Upscale",
        "description": "Enhance details and upscale an existing low-resolution image.",
        "required_inputs": ["prompt", "image", "upscale_by"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
    },
]

_TASKS_DB: Dict[str, Dict[str, Any]] = {}


class DummyProgress:
    def __call__(self, progress=0.0, desc=None):
        pass


def _get_public_base_url() -> str:
    """Auto-resolve the publicly accessible base URL (including protocol and port)."""
    # 1. Explicit environment variable override
    public_url = os.getenv("PUBLIC_URL") or os.getenv("BASE_URL")
    if public_url:
        return public_url.rstrip("/")

    # 2. Hugging Face Space environment variable
    space_host = os.getenv("SPACE_HOST")
    if space_host:
        if not space_host.startswith("http://") and not space_host.startswith("https://"):
            return f"https://{space_host}"
        return space_host.rstrip("/")

    # 3. Local Gradio config fallback
    try:
        from core.config import GRADIO_SERVER_NAME, SERVER_PORT
    except ImportError:
        GRADIO_SERVER_NAME = "127.0.0.1"
        SERVER_PORT = 7860

    server_name = os.getenv("GRADIO_SERVER_NAME", GRADIO_SERVER_NAME)
    if server_name == "0.0.0.0":
        server_name = "127.0.0.1"
    port = os.getenv("GRADIO_SERVER_PORT", str(SERVER_PORT))

    return f"http://{server_name}:{port}"


def _execute_imagegen_pipeline(task_id: str, params: dict):
    """Execute the image generation pipeline via ComfyUI backend and update _TASKS_DB."""
    start_time = time.time()
    try:
        _TASKS_DB[task_id]["status"] = "processing"
        _TASKS_DB[task_id]["progress"] = 10
        _TASKS_DB[task_id]["updated_at"] = int(start_time)

        from ..image_gen_logic import process_inputs
        from .get_model_list import ImageGen_get_model_list
        from .get_model_features import ImageGen_get_model_features
        from core.comfy_api import queue_prompt
        from core.backend_manager import backend_manager
        from core.config import SERVER_PORT, GRADIO_SERVER_NAME, COMFYUI_OUTPUT_PATH
        import websocket

        task_type = params["task_type"]
        model = params["model"]
        prompt = params["prompt"]

        model_defaults = _load_yaml(_MODEL_DEFAULTS_PATH)
        model_list = _load_yaml(_MODEL_LIST_PATH)
        checkpoints = model_list.get("Checkpoint", {}) or model_list.get("Checkpoints", {})
        found_arch = None
        for arch_name, arch_data in checkpoints.items():
            if isinstance(arch_data, dict):
                for m in arch_data.get("models", []):
                    if m.get("display_name") == model:
                        found_arch = arch_name
                        break
            if found_arch:
                break

        arch_defaults_section = model_defaults.get(found_arch, {}) if found_arch else {}
        arch_level_defaults = arch_defaults_section.get("_defaults", {})
        model_specific_defaults = arch_defaults_section.get(model, {})
        global_defaults = model_defaults.get("Default", {})
        merged_defaults = {**global_defaults, **arch_level_defaults, **model_specific_defaults}

        steps = params.get("steps") if params.get("steps") is not None else merged_defaults.get("steps", 20)
        cfg = params.get("cfg") if params.get("cfg") is not None else merged_defaults.get("cfg", 1.0)
        sampler = params.get("sampler") or merged_defaults.get("sampler_name", "euler")
        scheduler = params.get("scheduler") or merged_defaults.get("scheduler", "simple")

        prefix = task_type
        model_type_state = found_arch.lower().replace(" ", "-").replace(".", "") if found_arch else "sdxl"

        ui_values = {
            f"{prefix}_model_name": model,
            f"{prefix}_model_type_state": model_type_state,
            f"{prefix}_positive_prompt": prompt,
            f"{prefix}_negative_prompt": params.get("negative_prompt", merged_defaults.get("negative_prompt", "")),
            f"{prefix}_width": params.get("width", 1024),
            f"{prefix}_height": params.get("height", 1024),
            f"{prefix}_steps": steps,
            f"{prefix}_cfg": cfg,
            f"{prefix}_sampler_name": sampler,
            f"{prefix}_scheduler": scheduler,
            f"{prefix}_seed": params.get("seed", -1),
            f"{prefix}_batch_count": 1,
            f"{prefix}_batch_size": params.get("batch_size", 1),
            f"{prefix}_denoise": params.get("denoise", 1.0),
            f"{prefix}_lora_count_state": 0,
            f"{prefix}_controlnet_count_state": 0,
            f"{prefix}_ipadapter_count_state": 0,
            f"{prefix}_embedding_count_state": 0,
            f"{prefix}_style_count_state": 0,
            f"{prefix}_conditioning_count_state": 0,
            f"{prefix}_vae_source": "None",
        }

        if "image" in params and params["image"]:
            pil_img = _parse_image_param(params["image"])
            if pil_img:
                if task_type in ("img2img", "hires_fix"):
                    ui_values[f"{prefix}_input_image"] = pil_img
                    if task_type == "img2img":
                        ui_values[f"{prefix}_denoise"] = params.get("denoise", 0.7)
                    else:
                        ui_values[f"{prefix}_upscale_by"] = params.get("upscale_by", 2.0)
                        ui_values[f"{prefix}_denoise"] = params.get("denoise", 0.55)
                elif task_type == "inpaint":
                    if pil_img.mode == "RGBA":
                        bg = Image.new("RGB", pil_img.size, (0, 0, 0))
                        bg.paste(pil_img, mask=pil_img.split()[3])
                        ui_values[f"{prefix}_input_image_dict"] = {"background": bg, "layers": [pil_img]}
                    else:
                        ui_values[f"{prefix}_input_image_dict"] = {"background": pil_img, "layers": [pil_img]}
                    ui_values[f"{prefix}_denoise"] = params.get("denoise", 1.0)
                elif task_type == "outpaint":
                    ui_values[f"{prefix}_input_image"] = pil_img
                    ui_values[f"{prefix}_pad_left"] = params.get("pad_left", 0)
                    ui_values[f"{prefix}_pad_right"] = params.get("pad_right", 0)
                    ui_values[f"{prefix}_pad_top"] = params.get("pad_top", 0)
                    ui_values[f"{prefix}_pad_bottom"] = params.get("pad_bottom", 0)
                    ui_values[f"{prefix}_feathering"] = params.get("feathering", 10)

        chain = params.get("chain", [])
        if chain:
            for item in chain:
                itype = item.get("injector_type")
                if itype == "lora":
                    ui_values[f"{prefix}_loras_sources"] = ui_values.get(f"{prefix}_loras_sources", []) + [item.get("lora_source", "File")]
                    ui_values[f"{prefix}_loras_ids"] = ui_values.get(f"{prefix}_loras_ids", []) + [item.get("lora_value", "")]
                    ui_values[f"{prefix}_loras_scales"] = ui_values.get(f"{prefix}_loras_scales", []) + [item.get("scale", 1.0)]
                elif itype in ("controlnet", "krea2_controlnet", "anima_controlnet_lllite"):
                    parsed_cn_img = _parse_image_param(item.get("image"))
                    if parsed_cn_img:
                        ui_values[f"{prefix}_controlnet_images"] = ui_values.get(f"{prefix}_controlnet_images", []) + [parsed_cn_img]
                        ui_values[f"{prefix}_controlnet_strengths"] = ui_values.get(f"{prefix}_controlnet_strengths", []) + [item.get("strength", 1.0)]
                        ui_values[f"{prefix}_controlnet_filepaths"] = ui_values.get(f"{prefix}_controlnet_filepaths", []) + [item.get("control_net_name", "")]
                elif itype in ("ipadapter", "flux1_ipadapter", "sd3_ipadapter"):
                    parsed_ipa_img = _parse_image_param(item.get("image"))
                    if parsed_ipa_img:
                        ui_values[f"{prefix}_ipadapter_images"] = ui_values.get(f"{prefix}_ipadapter_images", []) + [parsed_ipa_img]
                        ui_values[f"{prefix}_ipadapter_weights"] = ui_values.get(f"{prefix}_ipadapter_weights", []) + [item.get("weight", 1.0)]
                        ui_values[f"{prefix}_ipadapter_final_preset"] = item.get("preset", "STANDARD (medium strength)")
                elif itype == "style":
                    parsed_style_img = _parse_image_param(item.get("image"))
                    if parsed_style_img:
                        ui_values[f"{prefix}_style_images"] = ui_values.get(f"{prefix}_style_images", []) + [parsed_style_img]
                        ui_values[f"{prefix}_style_strengths"] = ui_values.get(f"{prefix}_style_strengths", []) + [item.get("strength", 1.0)]

        _TASKS_DB[task_id]["progress"] = 30

        workflow, extra_data = process_inputs(task_type, ui_values)

        _TASKS_DB[task_id]["progress"] = 50

        client_id = uuid.uuid4().hex
        prompt_response = queue_prompt(workflow, client_id, extra_data)
        if not prompt_response or 'prompt_id' not in prompt_response:
            active_url = backend_manager.get_active_backend_url()
            raise RuntimeError(f"Failed to queue prompt to the ComfyUI backend at {active_url}.")

        prompt_id = prompt_response['prompt_id']

        active_url = backend_manager.get_active_backend_url()
        ws_url = f"ws://{urllib.parse.urlparse(active_url).netloc}/ws?clientId={client_id}"
        ws = None
        images = []
        base_url = _get_public_base_url()

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
                                for output_info in value:
                                    filename = output_info['filename']
                                    subfolder = output_info.get('subfolder', '')
                                    absolute_path = os.path.join(COMFYUI_OUTPUT_PATH, subfolder, filename)
                                    final_url = f"{base_url}/gradio_api/file={urllib.parse.quote(absolute_path)}"
                                    images.append(final_url)
                        if images:
                            break
        finally:
            if ws:
                ws.close()

        if not images:
            raise RuntimeError("Image generation failed; the backend did not report any output files.")

        execution_time = round(time.time() - start_time, 2)
        _TASKS_DB[task_id]["status"] = "completed"
        _TASKS_DB[task_id]["progress"] = 100
        _TASKS_DB[task_id]["completed_at"] = int(time.time())
        _TASKS_DB[task_id]["result"] = {
            "images": images,
            "seed": params.get("seed", -1),
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
            "execution_time_seconds": execution_time,
        }

    except Exception as e:
        _TASKS_DB[task_id]["status"] = "failed"
        _TASKS_DB[task_id]["progress"] = 0
        _TASKS_DB[task_id]["failed_at"] = int(time.time())
        _TASKS_DB[task_id]["error"] = {
            "code": "EXECUTION_ERROR",
            "message": str(e),
        }
