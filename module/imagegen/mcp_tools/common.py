"""
MCP Common Utilities & Data Structures
Contains YAML loading utilities, config file paths, task definitions, and async task database.
"""

import os
import time
import urllib.parse
import urllib.request
import urllib.error
import uuid
import json
import base64
import io
import yaml
from typing import Dict, Any
from PIL import Image

_MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_IMAGE_DOWNLOAD_TIMEOUT = 30  # seconds
_ALLOWED_IMAGE_CONTENT_TYPES = frozenset([
    "image/png", "image/jpeg", "image/jpg", "image/gif",
    "image/webp", "image/bmp", "image/tiff",
])

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YAML_DIR = os.path.join(_PROJECT_ROOT, "yaml")

_MODEL_ARCHITECTURES_PATH = os.path.join(_YAML_DIR, "model_architectures.yaml")
_MODEL_LIST_PATH = os.path.join(_YAML_DIR, "model_list.yaml")
_MODEL_DEFAULTS_PATH = os.path.join(_YAML_DIR, "model_defaults.yaml")
_IMAGE_GEN_FEATURES_PATH = os.path.join(_YAML_DIR, "image_gen_features.yaml")
_CHAIN_FEATURES_PATH = os.path.join(_YAML_DIR, "chain_features.yaml")
_TASK_FEATURES_PATH = os.path.join(_YAML_DIR, "task_features.yaml")
_CONSTANTS_PATH = os.path.join(_YAML_DIR, "constants.yaml")


def _get_ipadapter_presets_by_arch() -> Dict[str, list]:
    """Load IPAdapter presets from yaml/ipadapter.yaml for SD1.5 and SDXL."""
    ipadapter_yaml_path = os.path.join(_YAML_DIR, "ipadapter.yaml")
    data = _load_yaml(ipadapter_yaml_path)
    res = {}
    for arch in ("SD1.5", "SDXL"):
        std = data.get("IPAdapter_presets", {}).get(arch, [])
        face = data.get("IPAdapter_FaceID_presets", {}).get(arch, [])
        res[arch] = list(std) + list(face)
    return res



def _download_image_from_url(url: str) -> Image.Image:
    """Download an image from an HTTP/HTTPS URL and return it as a PIL Image.

    Security measures:
    - Timeout to prevent hanging on slow/malicious servers.
    - Response size cap to prevent memory exhaustion.
    - Content-Type validation to reject non-image responses.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "ImageGen-MCP/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=_IMAGE_DOWNLOAD_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type and content_type not in _ALLOWED_IMAGE_CONTENT_TYPES:
                raise ValueError(
                    f"URL returned non-image Content-Type '{content_type}'. "
                    f"Expected one of: {', '.join(sorted(_ALLOWED_IMAGE_CONTENT_TYPES))}."
                )

            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > _MAX_IMAGE_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Image at URL is too large ({int(content_length)} bytes). "
                    f"Maximum allowed size is {_MAX_IMAGE_DOWNLOAD_BYTES} bytes."
                )

            chunks = []
            total = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_IMAGE_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Image download exceeded maximum allowed size of "
                        f"{_MAX_IMAGE_DOWNLOAD_BYTES} bytes."
                    )
                chunks.append(chunk)

            data = b"".join(chunks)

    except urllib.error.URLError as e:
        raise ValueError(f"Failed to download image from URL: {e}") from e
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP error {e.code} when downloading image from URL: {e.reason}") from e

    if not data:
        raise ValueError("Downloaded image data is empty.")

    return Image.open(io.BytesIO(data))


def _parse_image_param(image_param: Any) -> Any:
    """Parse a Base64 Data URI, HTTP/HTTPS URL, local file path, or PIL.Image into a PIL Image object."""
    if isinstance(image_param, Image.Image):
        return image_param

    if not isinstance(image_param, str) or not image_param.strip():
        return None

    image_param = image_param.strip()

    # HTTP / HTTPS URL — download the image
    if image_param.startswith("http://") or image_param.startswith("https://"):
        return _download_image_from_url(image_param)

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
        "Invalid image parameter format. Expected an HTTP/HTTPS URL, a Base64 Data URI (e.g., 'data:image/png;base64,...'), or a local file path."
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
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    if server_name == "0.0.0.0":
        server_name = "127.0.0.1"
    port = os.getenv("GRADIO_SERVER_PORT", "7860")

    return f"http://{server_name}:{port}"


def _execute_imagegen_pipeline(task_id: str, params: dict):
    """Execute the image generation pipeline via ComfyUI backend and update _TASKS_DB."""
    start_time = time.time()
    try:
        _TASKS_DB[task_id]["status"] = "processing"
        _TASKS_DB[task_id]["progress"] = 10
        _TASKS_DB[task_id]["updated_at"] = int(start_time)

        from ..imagegen_logic import process_inputs
        from .get_model_list import ImageGen_get_model_list
        from .get_model_features import ImageGen_get_model_features
        from core.comfy_api import execute_workflow_and_wait, resolve_output_file_url


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
                    src = item.get("source") or item.get("lora_source", "File")
                    val = item.get("lora_value") or item.get("lora_id") or item.get("value", "")
                    scale = float(item.get("scale", item.get("strength", 1.0)))
                    ui_values[f"{prefix}_loras_sources"] = ui_values.get(f"{prefix}_loras_sources", []) + [src]
                    ui_values[f"{prefix}_loras_ids"] = ui_values.get(f"{prefix}_loras_ids", []) + [val]
                    ui_values[f"{prefix}_loras_file_dropdowns"] = ui_values.get(f"{prefix}_loras_file_dropdowns", []) + [val]
                    ui_values[f"{prefix}_loras_scales"] = ui_values.get(f"{prefix}_loras_scales", []) + [scale]
                elif itype == "embedding":
                    src = item.get("source") or item.get("embedding_source", "Civitai")
                    val = item.get("embedding_value") or item.get("embedding_id") or item.get("value", "")
                    ui_values[f"{prefix}_embeddings_sources"] = ui_values.get(f"{prefix}_embeddings_sources", []) + [src]
                    ui_values[f"{prefix}_embeddings_ids"] = ui_values.get(f"{prefix}_embeddings_ids", []) + [val]
                elif itype == "conditioning":
                    p = item.get("prompt", "")
                    if p:
                        ui_values[f"{prefix}_conditioning_prompts"] = ui_values.get(f"{prefix}_conditioning_prompts", []) + [p]
                        ui_values[f"{prefix}_conditioning_widths"] = ui_values.get(f"{prefix}_conditioning_widths", []) + [int(item.get("width", 512))]
                        ui_values[f"{prefix}_conditioning_heights"] = ui_values.get(f"{prefix}_conditioning_heights", []) + [int(item.get("height", 512))]
                        ui_values[f"{prefix}_conditioning_xs"] = ui_values.get(f"{prefix}_conditioning_xs", []) + [int(item.get("x", 0))]
                        ui_values[f"{prefix}_conditioning_ys"] = ui_values.get(f"{prefix}_conditioning_ys", []) + [int(item.get("y", 0))]
                        ui_values[f"{prefix}_conditioning_strengths"] = ui_values.get(f"{prefix}_conditioning_strengths", []) + [float(item.get("strength", 1.0))]
                elif itype == "controlnet":
                    parsed_cn_img = _parse_image_param(item.get("image"))
                    if parsed_cn_img:
                        ui_values[f"{prefix}_controlnet_images"] = ui_values.get(f"{prefix}_controlnet_images", []) + [parsed_cn_img]
                        ui_values[f"{prefix}_controlnet_strengths"] = ui_values.get(f"{prefix}_controlnet_strengths", []) + [float(item.get("strength", 1.0))]
                        ui_values[f"{prefix}_controlnet_filepaths"] = ui_values.get(f"{prefix}_controlnet_filepaths", []) + [item.get("filepath") or item.get("control_net_name", "")]
                elif itype == "anima_controlnet_lllite":
                    parsed_cn_img = _parse_image_param(item.get("image"))
                    if parsed_cn_img:
                        ui_values[f"{prefix}_anima_controlnet_lllite_images"] = ui_values.get(f"{prefix}_anima_controlnet_lllite_images", []) + [parsed_cn_img]
                        ui_values[f"{prefix}_anima_controlnet_lllite_strengths"] = ui_values.get(f"{prefix}_anima_controlnet_lllite_strengths", []) + [float(item.get("strength", 1.0))]
                        ui_values[f"{prefix}_anima_controlnet_lllite_filepaths"] = ui_values.get(f"{prefix}_anima_controlnet_lllite_filepaths", []) + [item.get("filepath") or item.get("control_net_name", "")]
                        ui_values[f"{prefix}_anima_controlnet_lllite_start_percents"] = ui_values.get(f"{prefix}_anima_controlnet_lllite_start_percents", []) + [float(item.get("start_percent", 0.0))]
                        ui_values[f"{prefix}_anima_controlnet_lllite_end_percents"] = ui_values.get(f"{prefix}_anima_controlnet_lllite_end_percents", []) + [float(item.get("end_percent", 1.0))]
                elif itype == "krea2_controlnet":
                    parsed_cn_img = _parse_image_param(item.get("image"))
                    if parsed_cn_img:
                        ui_values[f"{prefix}_krea2_controlnet_images"] = ui_values.get(f"{prefix}_krea2_controlnet_images", []) + [parsed_cn_img]
                        ui_values[f"{prefix}_krea2_controlnet_strengths"] = ui_values.get(f"{prefix}_krea2_controlnet_strengths", []) + [float(item.get("strength", 1.0))]
                        ui_values[f"{prefix}_krea2_controlnet_filepaths"] = ui_values.get(f"{prefix}_krea2_controlnet_filepaths", []) + [item.get("filepath") or item.get("control_net_name", "")]
                elif itype == "diffsynth_controlnet":
                    parsed_cn_img = _parse_image_param(item.get("image"))
                    if parsed_cn_img:
                        ui_values[f"{prefix}_diffsynth_controlnet_images"] = ui_values.get(f"{prefix}_diffsynth_controlnet_images", []) + [parsed_cn_img]
                        ui_values[f"{prefix}_diffsynth_controlnet_strengths"] = ui_values.get(f"{prefix}_diffsynth_controlnet_strengths", []) + [float(item.get("strength", 1.0))]
                        ui_values[f"{prefix}_diffsynth_controlnet_filepaths"] = ui_values.get(f"{prefix}_diffsynth_controlnet_filepaths", []) + [item.get("filepath") or item.get("control_net_name", "")]
                elif itype == "ipadapter":
                    parsed_ipa_img = _parse_image_param(item.get("image"))
                    if parsed_ipa_img:
                        ui_values[f"{prefix}_ipadapter_images"] = ui_values.get(f"{prefix}_ipadapter_images", []) + [parsed_ipa_img]
                        ui_values[f"{prefix}_ipadapter_weights"] = ui_values.get(f"{prefix}_ipadapter_weights", []) + [float(item.get("weight", 1.0))]
                        ui_values[f"{prefix}_ipadapter_lora_strengths"] = ui_values.get(f"{prefix}_ipadapter_lora_strengths", []) + [float(item.get("lora_strength", 0.6))]
                        if "preset" in item:
                            ui_values[f"{prefix}_ipadapter_final_preset"] = item["preset"]
                        if "final_weight" in item:
                            ui_values[f"{prefix}_ipadapter_final_weight"] = float(item["final_weight"])
                        if "embeds_scaling" in item:
                            ui_values[f"{prefix}_ipadapter_embeds_scaling"] = item["embeds_scaling"]
                        if "combine_method" in item:
                            ui_values[f"{prefix}_ipadapter_combine_method"] = item["combine_method"]
                        if "final_lora_strength" in item:
                            ui_values[f"{prefix}_ipadapter_final_lora_strength"] = float(item["final_lora_strength"])
                elif itype == "flux1_ipadapter":
                    parsed_ipa_img = _parse_image_param(item.get("image"))
                    if parsed_ipa_img:
                        ui_values[f"{prefix}_flux1_ipadapter_images"] = ui_values.get(f"{prefix}_flux1_ipadapter_images", []) + [parsed_ipa_img]
                        ui_values[f"{prefix}_flux1_ipadapter_weights"] = ui_values.get(f"{prefix}_flux1_ipadapter_weights", []) + [float(item.get("weight", 0.6))]
                        ui_values[f"{prefix}_flux1_ipadapter_start_percents"] = ui_values.get(f"{prefix}_flux1_ipadapter_start_percents", []) + [float(item.get("start_percent", item.get("start_at", 0.0)))]
                        ui_values[f"{prefix}_flux1_ipadapter_end_percents"] = ui_values.get(f"{prefix}_flux1_ipadapter_end_percents", []) + [float(item.get("end_percent", item.get("end_at", 0.6)))]
                elif itype == "sd3_ipadapter":
                    parsed_ipa_img = _parse_image_param(item.get("image"))
                    if parsed_ipa_img:
                        ui_values[f"{prefix}_sd3_ipadapter_images"] = ui_values.get(f"{prefix}_sd3_ipadapter_images", []) + [parsed_ipa_img]
                        ui_values[f"{prefix}_sd3_ipadapter_weights"] = ui_values.get(f"{prefix}_sd3_ipadapter_weights", []) + [float(item.get("weight", 0.5))]
                        ui_values[f"{prefix}_sd3_ipadapter_start_percents"] = ui_values.get(f"{prefix}_sd3_ipadapter_start_percents", []) + [float(item.get("start_percent", item.get("start_at", 0.0)))]
                        ui_values[f"{prefix}_sd3_ipadapter_end_percents"] = ui_values.get(f"{prefix}_sd3_ipadapter_end_percents", []) + [float(item.get("end_percent", item.get("end_at", 1.0)))]
                elif itype in ("style", "flux1_style"):
                    parsed_style_img = _parse_image_param(item.get("image"))
                    if parsed_style_img:
                        ui_values[f"{prefix}_style_images"] = ui_values.get(f"{prefix}_style_images", []) + [parsed_style_img]
                        ui_values[f"{prefix}_style_strengths"] = ui_values.get(f"{prefix}_style_strengths", []) + [float(item.get("strength", item.get("weight", 1.0)))]
                elif itype in ("reference_latent", "reference_edit"):
                    img = _parse_image_param(item.get("image"))
                    if img:
                        ui_values[f"{prefix}_reference_latent_images"] = ui_values.get(f"{prefix}_reference_latent_images", []) + [img]
                elif itype == "hidream_o1_reference":
                    img = _parse_image_param(item.get("image"))
                    if img:
                        ui_values[f"{prefix}_hidream_o1_reference_images"] = ui_values.get(f"{prefix}_hidream_o1_reference_images", []) + [img]
                elif itype == "sensenova_reference":
                    img = _parse_image_param(item.get("image"))
                    if img:
                        ui_values[f"{prefix}_sensenova_reference_images"] = ui_values.get(f"{prefix}_sensenova_reference_images", []) + [img]
                elif itype in ("joyai_image", "joyai_reference", "joyai_reference_edit"):
                    img = _parse_image_param(item.get("image"))
                    if img:
                        ui_values[f"{prefix}_joyai_image_images"] = ui_values.get(f"{prefix}_joyai_image_images", []) + [img]
                elif itype in ("reference_image", "mage_flow_reference_edit"):
                    img = _parse_image_param(item.get("image"))
                    if img:
                        ui_values[f"{prefix}_reference_image_images"] = ui_values.get(f"{prefix}_reference_image_images", []) + [img]
                elif itype in ("boogu_image_edit", "boogu_edit"):
                    parsed_boogu_img = _parse_image_param(item.get("image"))
                    if parsed_boogu_img:
                        ui_values[f"{prefix}_boogu_image_edit_images"] = ui_values.get(f"{prefix}_boogu_image_edit_images", []) + [parsed_boogu_img]
                elif itype == "qwen_image_edit":
                    parsed_qwen_img = _parse_image_param(item.get("image"))
                    if parsed_qwen_img:
                        ui_values[f"{prefix}_qwen_image_edit_images"] = ui_values.get(f"{prefix}_qwen_image_edit_images", []) + [parsed_qwen_img]
                elif itype == "krea2_identity_edit":
                    parsed_identity_img = _parse_image_param(item.get("image"))
                    if parsed_identity_img:
                        ui_values[f"{prefix}_krea2_identity_edit_images"] = ui_values.get(f"{prefix}_krea2_identity_edit_images", []) + [parsed_identity_img]
                elif itype == "krea2_style_reference":
                    parsed_style_ref_img = _parse_image_param(item.get("image"))
                    if parsed_style_ref_img:
                        ui_values[f"{prefix}_krea2_style_reference_images"] = ui_values.get(f"{prefix}_krea2_style_reference_images", []) + [parsed_style_ref_img]
                elif itype == "vae":
                    src = item.get("source") or item.get("vae_source", "File")
                    val = item.get("vae_value") or item.get("value") or item.get("vae_id") or item.get("vae_name", "")
                    ui_values[f"{prefix}_vae_override_source"] = src
                    ui_values[f"{prefix}_vae_override_id"] = val
                    ui_values[f"{prefix}_vae_override_file"] = val
                elif itype == "pid":
                    is_enabled = item.get("enabled", True)
                    if isinstance(is_enabled, str):
                        is_enabled = is_enabled.upper() in ("ON", "TRUE", "1")
                    ui_values[f"{prefix}_pid_settings"] = "ON" if is_enabled else "OFF"
                    ui_values["pid_settings"] = "ON" if is_enabled else "OFF"

            # Auto set default IPAdapter final settings if IPAdapter images are present
            if f"{prefix}_ipadapter_images" in ui_values and ui_values[f"{prefix}_ipadapter_images"]:
                if f"{prefix}_ipadapter_final_preset" not in ui_values:
                    ui_values[f"{prefix}_ipadapter_final_preset"] = "STANDARD (medium strength)"
                if f"{prefix}_ipadapter_final_weight" not in ui_values:
                    ui_values[f"{prefix}_ipadapter_final_weight"] = 1.0
                if f"{prefix}_ipadapter_embeds_scaling" not in ui_values:
                    ui_values[f"{prefix}_ipadapter_embeds_scaling"] = "V only"
                if f"{prefix}_ipadapter_combine_method" not in ui_values:
                    ui_values[f"{prefix}_ipadapter_combine_method"] = "concat"

        _TASKS_DB[task_id]["progress"] = 30

        workflow, extra_data = process_inputs(task_type, ui_values)

        _TASKS_DB[task_id]["progress"] = 50

        result = execute_workflow_and_wait((workflow, extra_data))

        output_files = result.get('output_files_info', [])
        downloaded_files = result.get('files', [])

        if not output_files:
            raise RuntimeError("Image generation failed; the backend did not report any output files.")

        images = []
        for i, info in enumerate(output_files):
            local_download = downloaded_files[i] if i < len(downloaded_files) else None
            url = resolve_output_file_url(info, local_download_path=local_download)
            images.append(url)


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
