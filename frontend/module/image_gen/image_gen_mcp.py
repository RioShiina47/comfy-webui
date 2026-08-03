"""
MCP Module for ImageGen in comfy-webui.
Exposes MCP_FUNCTIONS from .mcp_tools package.
"""

from .mcp_tools import (
    MCP_FUNCTIONS,
    ImageGen_get_task_list,
    ImageGen_get_model_architecture_list,
    ImageGen_get_model_list,
    ImageGen_get_feature_list,
    ImageGen_get_model_features,
    ImageGen_run_imagegen,
    ImageGen_get_task_status,
    ImageGen_get_chain_schema,
)

__all__ = ["MCP_FUNCTIONS"]
