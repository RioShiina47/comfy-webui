from __future__ import annotations

"""
MCP Tool: VideoGen_LTX_2_5_run
Unified Lightricks LTX-2.5 video generation task submission and execution interface.
"""

import time
import uuid
import threading
import gradio as gr
from .common import (
    _TASK_DEFINITIONS,
    _TASKS_DB,
    _execute_ltx2_5_pipeline,
)
from .error_schema import make_validation_error


def VideoGen_LTX_2_5_run(params: dict, request: gr.Request = None) -> dict:
    """
    Unified Lightricks LTX-2.5 video generation task execution interface.

    [SUPPORTED TASK TYPES]
    - t2va: Text-to-Video & Audio. Required: prompt.
    - i2va: Image-to-Video & Audio. Required: prompt, start_image.
    - ta2va: Text & Audio-to-Video & Audio. Required: prompt, audio_file.
    - ia2va: Image & Audio-to-Video & Audio. Required: prompt, start_image, audio_file.
    - flf2va: First & Last Frame-to-Video & Audio. Required: prompt, start_image, end_image.

    [GLOBAL OPTIONAL PARAMETERS]
    - negative_prompt (str): Negative prompt (default: "pc game, console game, video game, cartoon, childish, ugly").
    - resolution (str): Resolution preset: "544p", "768p", or "1080p" (default: "768p").
    - aspect_ratio (str): Aspect ratio: "16:9 (Widescreen)", "9:16 (Vertical)", "1:1 (Square)", "4:3 (Classic TV)", "3:4 (Classic Portrait)", "3:2 (Photography)", "2:3 (Photography Portrait)" (default: "16:9 (Widescreen)").
    - width (int): Custom width in pixels (overrides aspect_ratio preset if specified).
    - height (int): Custom height in pixels (overrides aspect_ratio preset if specified).
    - duration (float): Video duration in seconds (default: 5.0).
    - fps (str): Frame rate: "24fps" or "25fps" (default: "24fps").
    - seed (int): Random seed (-1 for random seed, >=0 for deterministic reproduction). Default: -1.
    - use_spatial_upscaler (bool): Enable 2x spatial latent upscaler (default: False).
    - use_temporal_upscaler (bool): Enable 2x temporal latent upscaler (default: False).
    - loras (list[dict]): List of LoRA configurations (e.g., [{"source": "Hugging Face", "id_or_url": "repo/lora.safetensors", "scale": 1.0}]).
    - async_execution (bool): If True, returns immediately with task_id for polling via VideoGen_LTX_2_5_get_task_status. Default: False.

    [Example (t2va)]
    {
        "task_type": "t2va",
        "prompt": "A futuristic sports car driving through a cyber city at sunset, 4k resolution",
        "negative_prompt": "pc game, console game, video game, cartoon, childish, ugly",
        "resolution": "768p",
        "aspect_ratio": "16:9 (Widescreen)",
        "duration": 5.0,
        "seed": -1
    }
    """
    if not isinstance(params, dict):
        return make_validation_error("Request params must be an object.")

    valid_tasks = [t["task_type"] for t in _TASK_DEFINITIONS]
    task_type = str(params.get("task_type") or params.get("task") or "").lower()

    if not task_type or task_type not in valid_tasks:
        return make_validation_error(
            f"Invalid or missing 'task_type'. Must be one of {valid_tasks}.",
            invalid_fields={"task_type": f"Must be in {valid_tasks}"},
        )

    task_def = next((t for t in _TASK_DEFINITIONS if t["task_type"] == task_type), None)
    required_fields = task_def["required_inputs"] if task_def else ["prompt"]

    missing = []
    for req_field in required_fields:
        if req_field not in params or params[req_field] is None or params[req_field] == "":
            missing.append(req_field)

    if missing:
        return make_validation_error(
            f"Missing required parameter(s) for task '{task_type}': {', '.join(missing)}",
            missing_fields=missing,
        )

    task_id = f"ltx2_5_task_{uuid.uuid4().hex[:10]}"
    created_at = int(time.time())

    _TASKS_DB[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "progress": 0,
        "created_at": created_at,
    }

    async_exec = bool(params.get("async_execution", False))

    if async_exec:
        t = threading.Thread(target=_execute_ltx2_5_pipeline, args=(task_id, params, request), daemon=True)
        t.start()
        return {
            "status": "queued",
            "task_id": task_id,
            "poll_interval_ms": 2000,
            "message": "Task queued successfully. Poll VideoGen_LTX_2_5_get_task_status for progress and results.",
        }
    else:
        _execute_ltx2_5_pipeline(task_id, params, request)
        return _TASKS_DB[task_id]
