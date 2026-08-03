"""
MCP & Gradio Integration Module

Provides:
1. register_high_level_mcp_apis: Expose only 8 high-level abstract API/MCP endpoints (using gr.api without polluting the visual UI structure)
2. cleanup_dependencies_api_names: Force cleanup of show_api attribute for non-high-level APIs in dependencies
3. patch_gradio_api_suppression: No-op implementation retained for backward compatibility
"""

import json
import gradio as gr

from .get_task_list import ImageGen_get_task_list
from .get_model_architecture_list import ImageGen_get_model_architecture_list
from .get_model_list import ImageGen_get_model_list
from .get_feature_list import ImageGen_get_feature_list
from .get_model_features import ImageGen_get_model_features
from .get_chain_schema import ImageGen_get_chain_schema
from .run_imagegen import ImageGen_run_imagegen
from .get_task_status import ImageGen_get_task_status

HIGH_LEVEL_MCP_API_NAMES = {
    "ImageGen_get_task_list",
    "ImageGen_get_model_architecture_list",
    "ImageGen_get_model_list",
    "ImageGen_get_feature_list",
    "ImageGen_get_model_features",
    "ImageGen_run_imagegen",
    "ImageGen_get_task_status",
    "ImageGen_get_chain_schema",
}


def sanitize_keys(obj):
    """Recursively ensure all dictionary keys are converted to str type to avoid Gradio 5 orjson TypeError: Dict key must be str."""
    if isinstance(obj, dict):
        return {str(k): sanitize_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_keys(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_keys(x) for x in obj)
    return obj


def patch_gradio_api_suppression():
    """Retained for backward compatibility (no-op)."""
    pass


def cleanup_dependencies_api_names(demo):
    """
    Clean up residual auto-generated API names in demo.fns and demo.dependencies.
    Force only the 8 high-level abstract MCP APIs to be exposed as public endpoints.
    """
    for fn in demo.fns.values():
        api_name = getattr(fn, "api_name", None)
        if api_name not in HIGH_LEVEL_MCP_API_NAMES:
            fn.show_api = False

    deps = getattr(demo, "dependencies", None)
    if deps is None and hasattr(demo, "config") and isinstance(demo.config, dict):
        deps = demo.config.get("dependencies", [])

    if deps:
        for dep in deps:
            if isinstance(dep, dict):
                api_name = dep.get("api_name")
                if api_name not in HIGH_LEVEL_MCP_API_NAMES:
                    dep["show_api"] = False

    print("[MCP Protection] Cleaned up demo dependencies. Suppressed atomic API endpoints.")


def register_high_level_mcp_apis(demo):
    """
    Explicitly register 8 high-level abstract MCP API endpoints on the Gradio demo using gr.api.
    Using gr.api() never adds any visual UI components (such as Row, Textbox, Button, etc.), avoiding duplicate interface rendering.
    """
    def get_task_list() -> list:
        """[Recommended Discovery Flow Step 1] Get a list of all supported image generation task types (txt2img, img2img, inpaint, outpaint, hires_fix) along with their required and optional parameter lists. Recommended flow: get_task_list -> get_model_architecture_list -> get_model_list -> [Path 1: Call run_imagegen directly (pass only required params) | Path 2: Call get_model_features to get official default hyperparams -> run_imagegen]."""
        return sanitize_keys(ImageGen_get_task_list())

    def get_model_architecture_list() -> list:
        """[Recommended Discovery Flow Step 2] Get a list of all supported model architectures (e.g., SD1.5, SDXL, FLUX, etc.) along with their default resolutions. It is recommended to call this tool before get_model_list to obtain valid model_architecture parameters for precise model filtering."""
        return sanitize_keys(ImageGen_get_model_architecture_list())

    def get_model_list(model_architecture: str = "") -> list | dict:
        """[Recommended Discovery Flow Step 3] Query the list of available image generation models. After obtaining models, choose one of two paths: 1. [Path 1 (Recommended - Minimal Mode)] Call run_imagegen directly with only required parameters. Do NOT guess steps/cfg/sampler/scheduler from experience; the server will automatically apply the model's optimal default hyperparameters. 2. [Path 2 (Explicit Alignment Mode)] First call get_model_features to query the model's officially recommended hyperparameters, then pass them to run_imagegen."""
        arch = model_architecture.strip() if model_architecture else None
        return sanitize_keys(ImageGen_get_model_list(arch))

    def get_feature_list() -> list:
        """Get the list of supported advanced features along with their usage constraints and parameter schemas."""
        return sanitize_keys(ImageGen_get_feature_list())

    def get_model_features(model: str = "") -> dict:
        """Query metadata for the specified model, including supported task types, extended features, and official default inference parameters (steps, cfg, sampler, scheduler). This tool MUST be called when explicitly obtaining a model's optimal default hyperparameters (Path 2). Guessing or fabricating hyperparameters without querying is strictly prohibited."""
        return sanitize_keys(ImageGen_get_model_features(model.strip()))

    def run_imagegen(json_params: str = "{}") -> dict:
        """[Recommended Discovery Flow Step 4] Unified image generation task execution interface. Supports txt2img, img2img, and other tasks with chainable extended features. [IMPORTANT PARAMETER RULES] Do NOT guess or fabricate inference hyperparameters such as steps, cfg, sampler, scheduler! Path 1 (Recommended): Pass only required parameters (task_type, model, prompt, width, height), leave optional hyperparams empty (server uses optimal defaults). Path 2: If explicit hyperparams are needed, you MUST first call get_model_features to obtain official defaults before passing them."""
        try:
            if isinstance(json_params, dict):
                params = json_params
            else:
                params = json.loads(json_params or "{}")
        except Exception as e:
            return {"error": {"code": "INVALID_JSON", "message": f"Failed to parse JSON params: {e}"}}
        return sanitize_keys(ImageGen_run_imagegen(params))

    def get_task_status(task_id: str = "") -> dict:
        """Query the progress, status, and final generated results of an async image generation task."""
        return sanitize_keys(ImageGen_get_task_status(task_id.strip()))

    def get_chain_schema(chain_type: str = "") -> dict:
        """Get the complete parameter schema and usage examples for a specified chain/injector type."""
        return sanitize_keys(ImageGen_get_chain_schema(chain_type.strip()))

    funcs = [
        get_task_list,
        get_model_architecture_list,
        get_model_list,
        get_feature_list,
        get_model_features,
        run_imagegen,
        get_task_status,
        get_chain_schema,
    ]

    for func in funcs:
        gr.api(func)

    for fn in demo.fns.values():
        if getattr(fn, "api_name", None) in HIGH_LEVEL_MCP_API_NAMES:
            fn.show_api = True

    print("[MCP Integration] Successfully registered 8 High-Level Abstract MCP APIs via gr.api().")
