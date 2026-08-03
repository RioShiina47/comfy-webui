def __getattr__(name):
    if name in ("types", "server", "client", "shared"):
        raise ImportError(f"No module named 'mcp.{name}' in local mcp package")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from .get_task_list import ImageGen_get_task_list
from .get_model_architecture_list import ImageGen_get_model_architecture_list
from .get_model_list import ImageGen_get_model_list
from .get_feature_list import ImageGen_get_feature_list
from .get_model_features import ImageGen_get_model_features
from .get_chain_schema import ImageGen_get_chain_schema
from .run_imagegen import ImageGen_run_imagegen
from .get_task_status import ImageGen_get_task_status
from .error_schema import make_error, make_validation_error, make_not_found_error
from .mcp_gradio_integration import (
    register_high_level_mcp_apis,
    cleanup_dependencies_api_names,
    patch_gradio_api_suppression,
    HIGH_LEVEL_MCP_API_NAMES,
)

MCP_FUNCTIONS = [
    ImageGen_get_task_list,
    ImageGen_get_model_architecture_list,
    ImageGen_get_model_list,
    ImageGen_get_feature_list,
    ImageGen_get_model_features,
    ImageGen_run_imagegen,
    ImageGen_get_task_status,
    ImageGen_get_chain_schema,
]

__all__ = [
    "ImageGen_get_task_list",
    "ImageGen_get_model_architecture_list",
    "ImageGen_get_model_list",
    "ImageGen_get_feature_list",
    "ImageGen_get_model_features",
    "ImageGen_get_chain_schema",
    "ImageGen_run_imagegen",
    "ImageGen_get_task_status",
    "make_error",
    "make_validation_error",
    "make_not_found_error",
    "register_high_level_mcp_apis",
    "cleanup_dependencies_api_names",
    "patch_gradio_api_suppression",
    "HIGH_LEVEL_MCP_API_NAMES",
    "MCP_FUNCTIONS",
]
