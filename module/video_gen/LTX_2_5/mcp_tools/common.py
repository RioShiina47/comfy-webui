from __future__ import annotations

"""
MCP Common Utilities & Data Structures for Lightricks LTX-2.5 Video Generation.
Contains media parsing utilities, task definitions, task database, and execution pipeline.
"""

import os
import time
import uuid
import urllib.parse
import urllib.request
import urllib.error
import base64
import io
from typing import Dict, Any, Optional
from PIL import Image

from core.config import COMFYUI_INPUT_PATH
from core.comfy_api import execute_workflow_and_wait, resolve_output_file_url
from ..ltx2_5_logic import process_inputs, RESOLUTION_PRESETS

_MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024   # 50 MB
_IMAGE_DOWNLOAD_TIMEOUT = 30                   # seconds
_MAX_AUDIO_DOWNLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
_AUDIO_DOWNLOAD_TIMEOUT = 60                   # seconds

_ALLOWED_IMAGE_CONTENT_TYPES = frozenset([
    "image/png", "image/jpeg", "image/jpg", "image/gif",
    "image/webp", "image/bmp", "image/tiff",
])

ASPECT_RATIO_PRESETS = RESOLUTION_PRESETS["768p"]

_TASKS_DB: Dict[str, Any] = {}


def _download_image_from_url(url: str) -> Image.Image:
    """Download an image from an HTTP/HTTPS URL and return it as a PIL Image."""
    req = urllib.request.Request(url, headers={"User-Agent": "LTX2.5-VideoGen-MCP/1.0"})
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
                    raise ValueError(f"Image download exceeded {_MAX_IMAGE_DOWNLOAD_BYTES} bytes.")
                chunks.append(chunk)

            raw_bytes = b"".join(chunks)
            img = Image.open(io.BytesIO(raw_bytes))
            img.load()
            return img
    except urllib.error.URLError as e:
        raise ValueError(f"Failed to download image from '{url}': {e.reason}")
    except Exception as e:
        raise ValueError(f"Error processing image from '{url}': {e}")


def _parse_image_param(image_param: Any) -> Optional[Image.Image]:
    """Parse a PIL Image, local file path, HTTP/HTTPS URL, or Base64 data URI into a PIL.Image."""
    if image_param is None:
        return None

    if isinstance(image_param, Image.Image):
        return image_param

    if not isinstance(image_param, str) or not image_param.strip():
        return None

    image_param = image_param.strip()

    if os.path.exists(image_param):
        try:
            img = Image.open(image_param)
            img.load()
            return img
        except Exception as e:
            raise ValueError(f"Failed to open local image file '{image_param}': {e}")

    if image_param.startswith("http://") or image_param.startswith("https://"):
        return _download_image_from_url(image_param)

    if image_param.startswith("data:image/"):
        try:
            _, encoded = image_param.split(",", 1) if "," in image_param else ("", image_param)
            data = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(data))
            img.load()
            return img
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image data URI: {e}")

    if len(image_param) > 100:
        try:
            data = base64.b64decode(image_param)
            img = Image.open(io.BytesIO(data))
            img.load()
            return img
        except Exception:
            pass

    raise ValueError(
        "Invalid image parameter format. Expected a file path, Base64 Data URI, or an HTTP/HTTPS URL."
    )


def _download_audio_from_url(url: str) -> str:
    """Download an audio file from HTTP/HTTPS URL and save it to COMFYUI_INPUT_PATH."""
    req = urllib.request.Request(url, headers={"User-Agent": "LTX2.5-VideoGen-MCP/1.0"})
    parsed_path = urllib.parse.urlparse(url).path
    suffix = os.path.splitext(parsed_path)[1].lower() or ".mp3"
    temp_path = os.path.join(COMFYUI_INPUT_PATH, f"mcp_download_{uuid.uuid4().hex[:10]}{suffix}")

    try:
        with urllib.request.urlopen(req, timeout=_AUDIO_DOWNLOAD_TIMEOUT) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > _MAX_AUDIO_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Audio at URL is too large ({int(content_length)} bytes). "
                    f"Maximum allowed size is {_MAX_AUDIO_DOWNLOAD_BYTES} bytes."
                )

            total = 0
            with open(temp_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MAX_AUDIO_DOWNLOAD_BYTES:
                        raise ValueError(f"Audio download exceeded {_MAX_AUDIO_DOWNLOAD_BYTES} bytes.")
                    f.write(chunk)
        return temp_path
    except urllib.error.URLError as e:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except Exception: pass
        raise ValueError(f"Failed to download audio from '{url}': {e.reason}")
    except Exception as e:
        if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except Exception: pass
        raise ValueError(f"Error saving audio from '{url}': {e}")


def _parse_audio_param(audio_param: Any) -> Optional[str]:
    """Parse a file path, URL, or Base64 string into a local audio file path."""
    if not audio_param:
        return None

    if not isinstance(audio_param, str) or not audio_param.strip():
        return None

    audio_param = audio_param.strip()

    if os.path.exists(audio_param):
        return audio_param

    if audio_param.startswith("http://") or audio_param.startswith("https://"):
        return _download_audio_from_url(audio_param)

    if audio_param.startswith("data:audio/"):
        try:
            _, encoded = audio_param.split(",", 1) if "," in audio_param else ("", audio_param)
            data = base64.b64decode(encoded)
            temp_path = os.path.join(COMFYUI_INPUT_PATH, f"mcp_upload_{uuid.uuid4().hex[:10]}.mp3")
            with open(temp_path, "wb") as f:
                f.write(data)
            return temp_path
        except Exception as e:
            raise ValueError(f"Failed to decode base64 audio data URI: {e}")

    if len(audio_param) > 200:
        try:
            data = base64.b64decode(audio_param)
            temp_path = os.path.join(COMFYUI_INPUT_PATH, f"mcp_upload_{uuid.uuid4().hex[:10]}.mp3")
            with open(temp_path, "wb") as f:
                f.write(data)
            return temp_path
        except Exception:
            pass

    raise ValueError(
        "Invalid audio parameter format. Expected an existing local file path, Base64 Data URI, or an HTTP/HTTPS URL."
    )


_COMMON_OPTIONAL_INPUTS = [
    "negative_prompt",
    "resolution",
    "aspect_ratio",
    "width",
    "height",
    "duration",
    "fps",
    "seed",
    "use_spatial_upscaler",
    "use_temporal_upscaler",
    "loras",
    "async_execution",
]

_GLOBAL_OPTIONAL_INPUTS_SCHEMA = {
    "negative_prompt": {
        "type": "string",
        "description": "Negative prompt specifying elements to avoid in the generated video.",
        "default": "pc game, console game, video game, cartoon, childish, ugly"
    },
    "resolution": {
        "type": "string",
        "enum": ["544p", "768p", "1080p"],
        "description": "Standard resolution preset tier (default: 768p).",
        "default": "768p"
    },
    "aspect_ratio": {
        "type": "string",
        "enum": list(ASPECT_RATIO_PRESETS.keys()),
        "description": "Aspect ratio preset.",
        "default": "16:9 (Widescreen)"
    },
    "width": {
        "type": "integer",
        "description": "Custom video width in pixels. Overrides aspect_ratio preset if specified.",
    },
    "height": {
        "type": "integer",
        "description": "Custom video height in pixels. Overrides aspect_ratio preset if specified.",
    },
    "duration": {
        "type": "number",
        "description": "Video duration in seconds (e.g. 5.0).",
        "default": 5.0
    },
    "fps": {
        "type": "string",
        "enum": ["24fps", "25fps"],
        "description": "Video frame rate.",
        "default": "24fps"
    },
    "seed": {
        "type": "integer",
        "description": "Random seed for deterministic reproduction (-1 for random).",
        "default": -1
    },
    "use_spatial_upscaler": {
        "type": "boolean",
        "description": "Enable 2x spatial latent upscaler for high definition video.",
        "default": False
    },
    "use_temporal_upscaler": {
        "type": "boolean",
        "description": "Enable 2x temporal latent upscaler for ultra smooth motion.",
        "default": False
    },
    "loras": {
        "type": "array",
        "description": "List of LoRA objects, e.g. [{'source': 'Hugging Face', 'id_or_url': 'repo/lora.safetensors', 'scale': 1.0}].",
    },
    "async_execution": {
        "type": "boolean",
        "description": "If True, runs in background and returns task_id immediately for polling.",
        "default": False
    }
}

_TASK_DEFINITIONS = [
    {
        "task_type": "t2va",
        "display_name": "Text-to-Video & Audio (T2VA)",
        "description": "Generate synchronized video and audio directly from a text prompt. Supports 2x spatial and temporal upscalers.",
        "required_inputs": ["prompt"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
        "optional_inputs_schema": _GLOBAL_OPTIONAL_INPUTS_SCHEMA,
        "aspect_ratio_presets": ASPECT_RATIO_PRESETS,
        "example_json_params": {
            "task_type": "t2va",
            "prompt": "A cinematic shot of a futuristic sports car driving through neon city streets at night",
            "negative_prompt": "pc game, console game, video game, cartoon, childish, ugly",
            "resolution": "768p",
            "aspect_ratio": "16:9 (Widescreen)",
            "duration": 5.0,
            "fps": "24fps",
            "seed": -1,
            "use_spatial_upscaler": False,
            "use_temporal_upscaler": False
        }
    },
    {
        "task_type": "i2va",
        "display_name": "Image-to-Video & Audio (I2VA)",
        "description": "Animate an initial image into a video with synchronized audio and motion guided by text prompt.",
        "required_inputs": ["prompt", "start_image"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
        "optional_inputs_schema": _GLOBAL_OPTIONAL_INPUTS_SCHEMA,
        "aspect_ratio_presets": ASPECT_RATIO_PRESETS,
        "example_json_params": {
            "task_type": "i2va",
            "prompt": "A majestic lion standing on a rock looking at the sunrise and roaring proudly",
            "start_image": "https://example.com/lion.png",
            "negative_prompt": "pc game, console game, video game, cartoon, childish, ugly",
            "resolution": "768p",
            "aspect_ratio": "16:9 (Widescreen)",
            "duration": 5.0,
            "seed": -1
        }
    },
    {
        "task_type": "ta2va",
        "display_name": "Text & Audio-to-Video & Audio (TA2VA)",
        "description": "Generate video synchronized to an input audio track driven by a text prompt.",
        "required_inputs": ["prompt", "audio_file"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
        "optional_inputs_schema": _GLOBAL_OPTIONAL_INPUTS_SCHEMA,
        "aspect_ratio_presets": ASPECT_RATIO_PRESETS,
        "example_json_params": {
            "task_type": "ta2va",
            "prompt": "Musicians playing jazz instruments on a softly lit stage in harmony with the music",
            "audio_file": "https://example.com/jazz.mp3",
            "negative_prompt": "pc game, console game, video game, cartoon, childish, ugly",
            "resolution": "768p",
            "aspect_ratio": "16:9 (Widescreen)",
            "seed": -1
        }
    },
    {
        "task_type": "ia2va",
        "display_name": "Image & Audio-to-Video & Audio (IA2VA)",
        "description": "Animate a starting image synchronized with an uploaded audio track.",
        "required_inputs": ["prompt", "start_image", "audio_file"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
        "optional_inputs_schema": _GLOBAL_OPTIONAL_INPUTS_SCHEMA,
        "aspect_ratio_presets": ASPECT_RATIO_PRESETS,
        "example_json_params": {
            "task_type": "ia2va",
            "prompt": "The singer in the image sings with expressive facial motion synchronized with the vocal audio",
            "start_image": "https://example.com/singer.png",
            "audio_file": "https://example.com/vocal.mp3",
            "negative_prompt": "pc game, console game, video game, cartoon, childish, ugly",
            "resolution": "768p",
            "aspect_ratio": "16:9 (Widescreen)",
            "seed": -1
        }
    },
    {
        "task_type": "flf2va",
        "display_name": "First & Last Frame-to-Video & Audio (FLF2VA)",
        "description": "Generate video smoothly morphing and transitioning from first frame to last frame.",
        "required_inputs": ["prompt", "start_image", "end_image"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
        "optional_inputs_schema": _GLOBAL_OPTIONAL_INPUTS_SCHEMA,
        "aspect_ratio_presets": ASPECT_RATIO_PRESETS,
        "example_json_params": {
            "task_type": "flf2va",
            "prompt": "A beautiful rose flower blooming smoothly from a closed bud to full bloom",
            "start_image": "https://example.com/bud.png",
            "end_image": "https://example.com/bloom.png",
            "negative_prompt": "pc game, console game, video game, cartoon, childish, ugly",
            "resolution": "768p",
            "aspect_ratio": "16:9 (Widescreen)",
            "duration": 5.0,
            "seed": -1
        }
    }
]


def _execute_ltx2_5_pipeline(task_id: str, params: dict, request: Any = None):
    """Executes the Lightricks LTX-2.5 video generation workflow via ComfyUI."""
    start_time = time.time()
    try:
        _TASKS_DB[task_id]["status"] = "processing"
        _TASKS_DB[task_id]["progress"] = 10
        _TASKS_DB[task_id]["updated_at"] = int(start_time)

        task_type = str(params.get("task_type") or params.get("task") or "T2VA").upper()
        if task_type not in ("T2VA", "I2VA", "TA2VA", "IA2VA", "FLF2VA"):
            task_type = "T2VA"

        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "pc game, console game, video game, cartoon, childish, ugly")
        resolution = str(params.get("resolution", "768p")).lower()
        aspect_ratio = params.get("aspect_ratio", "16:9 (Widescreen)")
        duration = float(params.get("duration", 5.0))
        fps = params.get("fps", "24fps")
        seed = int(params.get("seed", -1))
        use_spatial = bool(params.get("use_spatial_upscaler", False))
        use_temporal = bool(params.get("use_temporal_upscaler", False))

        loras_param = params.get("loras")
        loras_list = loras_param if isinstance(loras_param, list) else []

        ui_values = {
            "task": task_type,
            "positive_prompt": prompt,
            "negative_prompt": negative_prompt,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "fps": fps,
            "seed": seed,
            "batch_count": 1,
            "use_spatial_upscaler": use_spatial,
            "use_temporal_upscaler": use_temporal,
            "loras": loras_list,
        }

        if params.get("width") and params.get("height"):
            ui_values["width"] = int(params["width"])
            ui_values["height"] = int(params["height"])

        # Handle task specific inputs
        if task_type in ("I2VA", "IA2VA", "FLF2VA"):
            ui_values["start_image"] = _parse_image_param(params.get("start_image"))

        if task_type == "FLF2VA":
            ui_values["end_image"] = _parse_image_param(params.get("end_image"))

        if task_type in ("TA2VA", "IA2VA"):
            ui_values["audio_file"] = _parse_audio_param(params.get("audio_file"))

        _TASKS_DB[task_id]["progress"] = 30

        # Assemble workflow
        workflow, extra_data = process_inputs(ui_values)

        _TASKS_DB[task_id]["progress"] = 50

        # Execute on ComfyUI backend
        result = execute_workflow_and_wait((workflow, extra_data))

        output_files = result.get('output_files_info', [])
        downloaded_files = result.get('files', [])

        if not output_files:
            raise RuntimeError("Video generation failed; no output files were generated by ComfyUI.")

        video_urls = []
        for idx, file_info in enumerate(output_files):
            local_download = downloaded_files[idx] if idx < len(downloaded_files) else None
            file_url = resolve_output_file_url(file_info, request=request, local_download_path=local_download)
            video_urls.append(file_url)

        primary_video_url = video_urls[0] if video_urls else None
        local_video_path = downloaded_files[0] if downloaded_files else None

        execution_time = round(time.time() - start_time, 2)
        _TASKS_DB[task_id]["status"] = "completed"
        _TASKS_DB[task_id]["progress"] = 100
        _TASKS_DB[task_id]["completed_at"] = int(time.time())
        _TASKS_DB[task_id]["result"] = {
            "task_type": task_type.lower(),
            "videos": video_urls,
            "primary_video_url": primary_video_url,
            "video_path": local_video_path,
            "seed": ui_values.get("seed"),
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
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
