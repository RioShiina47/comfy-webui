"""
MCP Common Utilities & Data Structures for MiniMax-H3 Video Generation.
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
from typing import Dict, Any
from PIL import Image

from core.config import COMFYUI_INPUT_PATH
from core.comfy_api import execute_workflow_and_wait, resolve_output_file_url
from module.video_gen.H3.h3_logic import process_inputs, RESOLUTION_PRESETS, calculate_h3_frame_length

_MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024   # 50 MB
_IMAGE_DOWNLOAD_TIMEOUT = 30                   # seconds
_MAX_VIDEO_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
_VIDEO_DOWNLOAD_TIMEOUT = 60                   # seconds
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
    req = urllib.request.Request(url, headers={"User-Agent": "H3-VideoGen-MCP/1.0"})
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
                    raise ValueError(f"Image download exceeded maximum allowed size of {_MAX_IMAGE_DOWNLOAD_BYTES} bytes.")
                chunks.append(chunk)

            data = b"".join(chunks)

    except urllib.error.URLError as e:
        raise ValueError(f"Failed to download image from URL: {e}") from e
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP error {e.code} when downloading image from URL: {e.reason}") from e

    if not data:
        raise ValueError("Downloaded image data is empty.")

    return Image.open(io.BytesIO(data))


def _download_file_from_url(url: str, suffix: str = ".mp4", max_bytes: int = _MAX_VIDEO_DOWNLOAD_BYTES, timeout: int = _VIDEO_DOWNLOAD_TIMEOUT) -> str:
    """Download a media file from an HTTP/HTTPS URL and save it to a temporary local path in ComfyUI input."""
    req = urllib.request.Request(url, headers={"User-Agent": "H3-VideoGen-MCP/1.0"})
    os.makedirs(COMFYUI_INPUT_PATH, exist_ok=True)
    temp_path = os.path.join(COMFYUI_INPUT_PATH, f"mcp_download_{uuid.uuid4().hex[:10]}{suffix}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(
                    f"File at URL is too large ({int(content_length)} bytes). "
                    f"Maximum allowed size is {max_bytes} bytes."
                )

            total = 0
            with open(temp_path, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"File download exceeded maximum allowed size of {max_bytes} bytes.")
                    f.write(chunk)
        return temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise ValueError(f"Failed to download media file from URL: {e}") from e


def _parse_image_param(image_param: Any) -> Any:
    """Parse a Base64 Data URI, HTTP/HTTPS URL, file path, or PIL.Image into a PIL Image object."""
    if isinstance(image_param, Image.Image):
        return image_param

    if not isinstance(image_param, str) or not image_param.strip():
        return None

    image_param = image_param.strip()

    if os.path.exists(image_param):
        return Image.open(image_param)

    if image_param.startswith("http://") or image_param.startswith("https://"):
        return _download_image_from_url(image_param)

    if image_param.startswith("data:image/"):
        _, encoded = image_param.split(",", 1) if "," in image_param else ("", image_param)
        data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(data))

    if len(image_param) > 100:
        try:
            data = base64.b64decode(image_param)
            return Image.open(io.BytesIO(data))
        except Exception:
            pass

    raise ValueError("Invalid image parameter format. Expected a file path, Base64 Data URI, or an HTTP/HTTPS URL.")


def _parse_video_param(video_param: Any) -> Any:
    """Parse a video parameter: returns a local file path, downloading from URL or saving from Base64 if necessary."""
    if not isinstance(video_param, str) or not video_param.strip():
        return None

    video_param = video_param.strip()

    if os.path.exists(video_param):
        return video_param

    if video_param.startswith("http://") or video_param.startswith("https://"):
        return _download_file_from_url(video_param, suffix=".mp4", max_bytes=_MAX_VIDEO_DOWNLOAD_BYTES, timeout=_VIDEO_DOWNLOAD_TIMEOUT)

    if video_param.startswith("data:video/") or len(video_param) > 200:
        _, encoded = video_param.split(",", 1) if "," in video_param else ("", video_param)
        data = base64.b64decode(encoded)
        os.makedirs(COMFYUI_INPUT_PATH, exist_ok=True)
        temp_path = os.path.join(COMFYUI_INPUT_PATH, f"mcp_upload_{uuid.uuid4().hex[:10]}.mp4")
        with open(temp_path, "wb") as f:
            f.write(data)
        return temp_path

    return video_param


def _parse_audio_param(audio_param: Any) -> Any:
    """Parse an audio parameter: returns a local file path, downloading from URL or saving from Base64 if necessary."""
    if not isinstance(audio_param, str) or not audio_param.strip():
        return None

    audio_param = audio_param.strip()

    if os.path.exists(audio_param):
        return audio_param

    if audio_param.startswith("http://") or audio_param.startswith("https://"):
        return _download_file_from_url(audio_param, suffix=".mp3", max_bytes=_MAX_AUDIO_DOWNLOAD_BYTES, timeout=_AUDIO_DOWNLOAD_TIMEOUT)

    if audio_param.startswith("data:audio/") or len(audio_param) > 200:
        _, encoded = audio_param.split(",", 1) if "," in audio_param else ("", audio_param)
        data = base64.b64decode(encoded)
        os.makedirs(COMFYUI_INPUT_PATH, exist_ok=True)
        temp_path = os.path.join(COMFYUI_INPUT_PATH, f"mcp_upload_{uuid.uuid4().hex[:10]}.mp3")
        with open(temp_path, "wb") as f:
            f.write(data)
        return temp_path

    return audio_param


_GLOBAL_OPTIONAL_INPUTS_SCHEMA = {
    "seed": {
        "type": "integer",
        "default": -1,
        "description": "Random seed for video generation (-1 for random seed, >=0 for deterministic reproduction)."
    },
    "steps": {
        "type": "integer",
        "default": 20,
        "description": "Number of inference sampling steps (1-50, default: 20)."
    },
    "cfg": {
        "type": "number",
        "default": 6.0,
        "description": "CFG scale for text conditioning (default: 6.0)."
    },
    "flow_shift": {
        "type": "number",
        "default": 3.0,
        "description": "Flow shift parameter for scheduler (default: 3.0)."
    },
    "loras": {
        "type": "array",
        "description": "List of LoRA configurations to apply. E.g., [{'source': 'Hugging Face', 'id_or_url': 'repo/lora.safetensors', 'scale': 1.0}] or [{'lora_name': 'my_lora.safetensors', 'strength_model': 1.0}].",
        "items": {"type": "object"}
    },
    "h3_controlnets": {
        "type": "array",
        "description": "List of H3 ControlNet configurations. E.g., [{'video': 'https://example.com/pose.mp4', 'strength': 1.0, 'controlnet_type': 'openpose'}].",
        "items": {"type": "object"}
    },
    "control_video": {
        "type": "string",
        "description": "Convenience parameter: URL, file path, or Base64 Data URI of a single control video for H3 ControlNet."
    },
    "control_strength": {
        "type": "number",
        "default": 1.0,
        "description": "Strength for single convenience control video (default: 1.0)."
    },
    "controlnet_type": {
        "type": "string",
        "default": "openpose",
        "description": "ControlNet conditioning type (openpose, depth, etc., default: openpose)."
    },
    "h3_guides": {
        "type": "array",
        "description": "List of MiniMax H3 timeline Keyframe Guide configurations. E.g., [{'image': 'https://example.com/frame.png', 'time_seconds': 1.5}].",
        "items": {"type": "object"}
    }
}

_COMMON_OPTIONAL_INPUTS = [
    "seed", "steps", "cfg", "flow_shift", "loras", "h3_controlnets", "control_video", "control_strength", "controlnet_type", "h3_guides"
]

_TASK_DEFINITIONS = [
    {
        "task_type": "t2va",
        "display_name": "Text-to-Video & Audio (T2VA)",
        "description": "Generate videos from text prompt without input frames. Supports LoRAs, H3 ControlNet, and timeline Keyframe Guides.",
        "required_inputs": ["prompt", "width", "height", "duration"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
        "optional_inputs_schema": _GLOBAL_OPTIONAL_INPUTS_SCHEMA,
        "aspect_ratio_presets": ASPECT_RATIO_PRESETS,
        "example_json_params": {
            "task_type": "t2va",
            "prompt": "A futuristic city with flying cars at sunset, cinematic 4k",
            "width": 1344,
            "height": 768,
            "duration": 5.0,
            "steps": 20,
            "seed": -1
        }
    },
    {
        "task_type": "i2va",
        "display_name": "Image-to-Video & Audio (I2VA)",
        "description": "Generate videos starting from a single initial image frame (first frame). Supports LoRAs, H3 ControlNet, and timeline Keyframe Guides.",
        "required_inputs": ["prompt", "width", "height", "duration", "first_frame_image"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
        "optional_inputs_schema": _GLOBAL_OPTIONAL_INPUTS_SCHEMA,
        "aspect_ratio_presets": ASPECT_RATIO_PRESETS,
        "example_json_params": {
            "task_type": "i2va",
            "prompt": "A majestic lion standing on a rock looking at the horizon",
            "width": 1344,
            "height": 768,
            "duration": 5.0,
            "first_frame_image": "https://example.com/first_frame.png",
            "steps": 20,
            "seed": -1
        }
    },
    {
        "task_type": "flf2va",
        "display_name": "First & Last Frame-to-Video & Audio (FLF2VA)",
        "description": "Generate videos guided by both a starting frame (first frame) and ending frame (last frame). Supports LoRAs, H3 ControlNet, and timeline Keyframe Guides.",
        "required_inputs": ["prompt", "width", "height", "duration", "first_frame_image", "last_frame_image"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
        "optional_inputs_schema": _GLOBAL_OPTIONAL_INPUTS_SCHEMA,
        "aspect_ratio_presets": ASPECT_RATIO_PRESETS,
        "example_json_params": {
            "task_type": "flf2va",
            "prompt": "Smooth transition of a flower blooming from start to finish",
            "width": 1344,
            "height": 768,
            "duration": 5.0,
            "first_frame_image": "https://example.com/first_frame.png",
            "last_frame_image": "https://example.com/last_frame.png",
            "steps": 20,
            "seed": -1
        }
    },
    {
        "task_type": "ref2va",
        "display_name": "Reference-to-Video & Audio (REF2VA)",
        "description": "Generate videos with multi-modal references (up to 9 ref images, 3 ref videos, and 3 ref audios). Supports LoRAs, H3 ControlNet, and timeline Keyframe Guides.",
        "required_inputs": ["prompt", "width", "height", "duration"],
        "optional_inputs": [
            *_COMMON_OPTIONAL_INPUTS,
            "ref_image1", "ref_image2", "ref_image3", "ref_image4", "ref_image5", "ref_image6", "ref_image7", "ref_image8", "ref_image9",
            "ref_video1", "ref_video2", "ref_video3",
            "ref_audio1", "ref_audio2", "ref_audio3",
            "ref_images", "ref_videos", "ref_audios"
        ],
        "optional_inputs_schema": _GLOBAL_OPTIONAL_INPUTS_SCHEMA,
        "aspect_ratio_presets": ASPECT_RATIO_PRESETS,
        "example_json_params": {
            "task_type": "ref2va",
            "prompt": "A character dancing in a cyberpunk room guided by reference media",
            "width": 1344,
            "height": 768,
            "duration": 5.0,
            "steps": 20,
            "ref_image1": "https://example.com/char.png",
            "ref_video1": "https://example.com/motion.mp4",
            "seed": -1
        }
    }
]


def _execute_h3_pipeline(task_id: str, params: dict, request: Any = None):
    """Executes the MiniMax-H3 video generation workflow via ComfyUI."""
    start_time = time.time()
    try:
        _TASKS_DB[task_id]["status"] = "processing"
        _TASKS_DB[task_id]["progress"] = 10
        _TASKS_DB[task_id]["updated_at"] = int(start_time)

        raw_task = str(params.get("task_type") or params.get("task") or "T2VA").upper()
        if raw_task in ("T2VA", "I2VA", "FLF2VA", "REF2VA"):
            task_type = raw_task
        elif raw_task == "FL2VA":
            task_type = "FLF2VA"
        else:
            task_type = "T2VA"

        prompt = params.get("prompt", "")
        width = int(params.get("width", 1344))
        height = int(params.get("height", 768))
        duration = float(params.get("duration", 5.0))
        steps = int(params.get("steps", 20))
        seed = int(params.get("seed", -1))
        cfg = float(params.get("cfg", 6.0))
        flow_shift = float(params.get("flow_shift", 3.0))

        # LoRA parsing
        loras_param = params.get("loras")
        loras_list = loras_param if isinstance(loras_param, list) else []

        # ControlNet parsing
        parsed_cns = []
        h3_cns_param = params.get("h3_controlnets")
        if h3_cns_param and isinstance(h3_cns_param, list):
            for item in h3_cns_param:
                if isinstance(item, dict):
                    vid_raw = item.get("video") or item.get("control_video") or item.get("file")
                    vid_parsed = _parse_video_param(vid_raw) if vid_raw else None
                    if vid_parsed:
                        new_item = dict(item)
                        new_item["video"] = vid_parsed
                        parsed_cns.append(new_item)
        elif params.get("control_video"):
            vid_raw = params.get("control_video")
            vid_parsed = _parse_video_param(vid_raw) if vid_raw else None
            if vid_parsed:
                parsed_cns.append({
                    "video": vid_parsed,
                    "controlnet_type": params.get("controlnet_type", "openpose"),
                    "strength": float(params.get("control_strength", 1.0)),
                })

        # Keyframe Guides parsing
        parsed_guides = []
        raw_guides = params.get("h3_guides") or params.get("guides")
        if raw_guides and isinstance(raw_guides, list):
            for item in raw_guides:
                if isinstance(item, dict):
                    guide_entry = {}
                    if item.get("image"):
                        guide_entry["image"] = _parse_image_param(item["image"])
                    if item.get("video"):
                        guide_entry["video"] = _parse_video_param(item["video"])
                    if item.get("audio"):
                        guide_entry["audio"] = _parse_audio_param(item["audio"])
                    if item.get("frame_idx") is not None:
                        guide_entry["frame_idx"] = item["frame_idx"]
                    if item.get("time_seconds") is not None or item.get("time") is not None:
                        guide_entry["time_seconds"] = item.get("time_seconds") if item.get("time_seconds") is not None else item.get("time")
                    if guide_entry:
                        parsed_guides.append(guide_entry)

        ui_values = {
            "task": task_type,
            "prompt": prompt,
            "width": width,
            "height": height,
            "duration": duration,
            "steps": steps,
            "seed": seed,
            "cfg": cfg,
            "flow_shift": flow_shift,
            "loras": loras_list,
            "h3_controlnets": parsed_cns,
            "h3_guides": parsed_guides,
        }

        # Handle task specific inputs
        if task_type in ("T2VA", "I2VA", "FLF2VA"):
            if task_type in ("I2VA", "FLF2VA"):
                first_frame_raw = params.get("first_frame_image") or params.get("first_frame")
                ui_values["first_frame"] = _parse_image_param(first_frame_raw) if first_frame_raw else None
            if task_type == "FLF2VA":
                last_frame_raw = params.get("last_frame_image") or params.get("last_frame")
                ui_values["last_frame"] = _parse_image_param(last_frame_raw) if last_frame_raw else None

        elif task_type == "REF2VA":
            ref_images = []
            for i in range(1, 10):
                val = params.get(f"ref_image{i}")
                if val:
                    parsed_img = _parse_image_param(val)
                    if parsed_img:
                        ref_images.append(parsed_img)
            if not ref_images and isinstance(params.get("ref_images"), list):
                for val in params["ref_images"]:
                    parsed_img = _parse_image_param(val)
                    if parsed_img:
                        ref_images.append(parsed_img)
            ui_values["ref_images"] = ref_images

            ref_videos = []
            for i in range(1, 4):
                val = params.get(f"ref_video{i}")
                if val:
                    parsed_vid = _parse_video_param(val)
                    if parsed_vid:
                        ref_videos.append(parsed_vid)
            if not ref_videos and isinstance(params.get("ref_videos"), list):
                for val in params["ref_videos"]:
                    parsed_vid = _parse_video_param(val)
                    if parsed_vid:
                        ref_videos.append(parsed_vid)
            ui_values["ref_videos"] = ref_videos

            ref_audios = []
            for i in range(1, 4):
                val = params.get(f"ref_audio{i}")
                if val:
                    parsed_aud = _parse_audio_param(val)
                    if parsed_aud:
                        ref_audios.append(parsed_aud)
            if not ref_audios and isinstance(params.get("ref_audios"), list):
                for val in params["ref_audios"]:
                    parsed_aud = _parse_audio_param(val)
                    if parsed_aud:
                        ref_audios.append(parsed_aud)
            ui_values["ref_audios"] = ref_audios

        _TASKS_DB[task_id]["progress"] = 30

        # Assemble workflow
        workflow, extra_data = process_inputs(ui_values)

        _TASKS_DB[task_id]["progress"] = 50

        # Execute on ComfyUI backend
        result = execute_workflow_and_wait((workflow, extra_data))

        output_files = result.get('output_files_info', [])
        downloaded_files = result.get('files', [])

        if not output_files:
            raise RuntimeError("Video generation failed; no output files were reported by the backend.")

        videos = []
        for i, info in enumerate(output_files):
            local_download = downloaded_files[i] if i < len(downloaded_files) else None
            url = resolve_output_file_url(info, request=request, local_download_path=local_download)
            videos.append(url)

        execution_time = round(time.time() - start_time, 2)
        _TASKS_DB[task_id]["status"] = "completed"
        _TASKS_DB[task_id]["progress"] = 100
        _TASKS_DB[task_id]["completed_at"] = int(time.time())
        _TASKS_DB[task_id]["result"] = {
            "task_type": task_type.lower(),
            "videos": videos,
            "video_url": videos[0] if videos else None,
            "seed": seed,
            "width": width,
            "height": height,
            "duration": duration,
            "steps": steps,
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
