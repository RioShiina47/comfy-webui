"""
MCP Tool: run_imagegen
Unified image generation task submission and execution interface.
"""

import time
import uuid
import threading
from .common import (
    _load_yaml,
    _MODEL_LIST_PATH,
    _TASK_DEFINITIONS,
    _TASKS_DB,
    _execute_imagegen_pipeline,
)
from .error_schema import make_validation_error, make_not_found_error


def ImageGen_run_imagegen(params: dict) -> dict:
    """Unified image generation task execution interface."""
    if not isinstance(params, dict):
        return make_validation_error("Request params must be an object.")

    missing = []
    for req_field in ["task_type", "model", "prompt"]:
        if req_field not in params or not params[req_field]:
            missing.append(req_field)
    if missing:
        return make_validation_error(
            f"Missing required parameter(s): {', '.join(missing)}",
            missing_fields=missing,
        )

    task_type = params["task_type"]
    valid_tasks = [t["task_type"] for t in _TASK_DEFINITIONS]
    if task_type not in valid_tasks:
        return make_validation_error(
            f"Invalid task_type '{task_type}'. Must be one of {valid_tasks}.",
            invalid_fields={"task_type": f"Must be in {valid_tasks}"},
        )

    model_list = _load_yaml(_MODEL_LIST_PATH)
    checkpoints = model_list.get("Checkpoint", {}) or model_list.get("Checkpoints", {})
    all_models = set()
    for arch_name, arch_data in checkpoints.items():
        if isinstance(arch_data, dict):
            for m in arch_data.get("models", []):
                all_models.add(m.get("display_name"))

    if params["model"] not in all_models:
        return make_not_found_error("model", params["model"])

    task_id = f"img_task_{uuid.uuid4().hex[:10]}"
    created_at = int(time.time())

    _TASKS_DB[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "progress": 0,
        "created_at": created_at,
    }

    async_exec = params.get("async_execution", False)

    if async_exec:
        t = threading.Thread(target=_execute_imagegen_pipeline, args=(task_id, params), daemon=True)
        t.start()
        return {
            "status": "queued",
            "task_id": task_id,
            "poll_interval_ms": 2000,
            "message": "Task queued successfully. Poll get_task_status for results.",
        }
    else:
        _execute_imagegen_pipeline(task_id, params)
        return _TASKS_DB[task_id]
