"""
MCP Tool: VideoGen_H3_run
Unified MiniMax-H3 video generation task submission and execution interface.
"""

import time
import uuid
import threading
import gradio as gr
from .common import (
    _TASK_DEFINITIONS,
    _TASKS_DB,
    _execute_h3_pipeline,
)
from .error_schema import make_validation_error


def VideoGen_H3_run(params: dict, request: gr.Request = None) -> dict:
    """
    Unified MiniMax-H3 video generation task execution interface.

    [SUPPORTED TASK TYPES]
    - t2va: Text-to-Video & Audio. Required: prompt, width, height, duration.
    - i2va: Image-to-Video & Audio. Required: prompt, width, height, duration, first_frame_image.
    - flf2va: First & Last Frame-to-Video & Audio. Required: prompt, width, height, duration, first_frame_image, last_frame_image.
    - ref2va: Reference-to-Video & Audio. Required: prompt, width, height, duration. Optional: ref_image1..9, ref_video1..3, ref_audio1..3, h3_guides.

    [GLOBAL OPTIONAL PARAMETERS]
    - steps (int): Inference sampling steps (default: 20).
    - seed (int): Random seed (-1 for random seed, >=0 for deterministic reproduction). Default: -1.
    - cfg (float): Classifier-Free Guidance scale (default: 6.0).
    - flow_shift (float): Flow shift for scheduler (default: 3.0).
    - async_execution (bool): If True, returns immediately with task_id for polling. Default: False.
    - loras (list[dict]): List of LoRA configurations (e.g., [{"source": "Hugging Face", "id_or_url": "repo/lora.safetensors", "scale": 1.0}]).
    - h3_controlnets (list[dict]): List of H3 ControlNet configurations (e.g., [{"video": "https://.../pose.mp4", "strength": 1.0}]).
    - h3_guides (list[dict]): List of keyframe guide configurations (e.g., [{"image": "https://.../keyframe.png", "time_seconds": 1.5}]).
    - control_video (str): Single convenience control video URL/path/base64 for H3 ControlNet.
    - control_strength (float): Strength for single convenience control video (default: 1.0).

    [Example (t2va)]
    {
        "task_type": "t2va",
        "prompt": "A futuristic city with flying cars at sunset, cinematic 4k",
        "width": 1344,
        "height": 768,
        "duration": 5.0,
        "steps": 20,
        "seed": -1
    }
    """
    if not isinstance(params, dict):
        return make_validation_error("Request params must be an object.")

    valid_tasks = [t["task_type"] for t in _TASK_DEFINITIONS]
    raw_task = str(params.get("task_type") or params.get("task") or "").lower()

    if not raw_task or raw_task not in valid_tasks:
        return make_validation_error(
            f"Invalid or missing 'task_type'. Must be one of {valid_tasks}.",
            invalid_fields={"task_type": f"Must be in {valid_tasks}"},
        )

    task_def = next((t for t in _TASK_DEFINITIONS if t["task_type"] == raw_task), None)
    required_fields = task_def["required_inputs"] if task_def else ["prompt", "width", "height", "duration"]

    missing = []
    for req_field in required_fields:
        if req_field not in params or params[req_field] is None or params[req_field] == "":
            missing.append(req_field)
    if missing:
        return make_validation_error(
            f"Missing required parameter(s) for task '{raw_task}': {', '.join(missing)}",
            missing_fields=missing,
        )

    task_id = f"h3_task_{uuid.uuid4().hex[:10]}"
    created_at = int(time.time())

    _TASKS_DB[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "progress": 0,
        "created_at": created_at,
    }

    async_exec = bool(params.get("async_execution", False))

    if async_exec:
        t = threading.Thread(target=_execute_h3_pipeline, args=(task_id, params, request), daemon=True)
        t.start()
        return {
            "status": "queued",
            "task_id": task_id,
            "poll_interval_ms": 2000,
            "message": "Task queued successfully. Poll VideoGen_H3_get_task_status for progress and results.",
        }
    else:
        _execute_h3_pipeline(task_id, params, request)
        return _TASKS_DB[task_id]
