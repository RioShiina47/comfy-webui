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
    """
    Unified image generation task execution interface.

    [OPTIONAL CONTROL PARAMETERS]
    - seed (int): Random seed for generation. Default: -1 (random seed). Specify >=0 for deterministic reproducibility.
    - batch_size (int): Number of images generated in a single batch (1 to 16, default: 1).
    - negative_prompt (str): Text prompt specifying undesirable elements to avoid.
    - steps (int), cfg (float), sampler (str), scheduler (str): Inference hyperparams (auto-applied from model defaults if omitted).
    - chain (list): Array of injector objects (LoRA, ControlNet, IP-Adapter, etc.).

    [Paste-and-Run json_params Example (Basic)]
    {
        "task_type": "txt2img",
        "model": "stabilityai/SDXL-Base-1.0",
        "prompt": "A majestic lion jumping from a big stone at night",
        "width": 1024,
        "height": 1024
    }

    [Paste-and-Run json_params Example (With chain)]
    {
        "task_type": "txt2img",
        "model": "stabilityai/SDXL-Base-1.0",
        "prompt": "A majestic lion jumping from a big stone at night",
        "width": 1024,
        "height": 1024,
        "chain": [
            {
                "injector_type": "lora",
                "source": "Civitai",
                "lora_value": "12345",
                "scale": 1.0
            }
        ]
    }
    """
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

    if "chain" in params and params["chain"] is not None:
        chain_val = params["chain"]
        if isinstance(chain_val, dict):
            return make_validation_error(
                "Parameter 'chain' must be a JSON array (list) of injector objects [{'injector_type': 'lora', ...}], but received a dictionary. "
                "Do NOT structure chain as a dict like {'lora': [...]}. "
                "Example correct format: [{'injector_type': 'lora', 'source': 'Civitai', 'lora_value': '12345', 'scale': 1.0}]",
                invalid_fields={"chain": "Expected list of objects, received dict"},
            )
        if not isinstance(chain_val, list):
            return make_validation_error(
                "Parameter 'chain' must be a JSON array (list) of injector objects.",
                invalid_fields={"chain": f"Expected list, received {type(chain_val).__name__}"},
            )
        for idx, item in enumerate(chain_val):
            if not isinstance(item, dict):
                return make_validation_error(
                    f"Item at chain[{idx}] must be an object (dict) containing 'injector_type'. "
                    f"Example: {{'injector_type': 'lora', 'source': 'Civitai', 'lora_value': '12345', 'scale': 1.0}}",
                    invalid_fields={f"chain[{idx}]": f"Expected dict, received {type(item).__name__}"},
                )
            if "injector_type" not in item or not item["injector_type"]:
                return make_validation_error(
                    f"Item at chain[{idx}] is missing required string field 'injector_type'. "
                    f"Example: {{'injector_type': 'lora', 'source': 'Civitai', 'lora_value': '12345', 'scale': 1.0}}",
                    missing_fields=[f"chain[{idx}].injector_type"],
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
