"""
MCP Module for Lightricks LTX-2.5 Video Generation in comfy-webui.
Exposes MCP_FUNCTIONS from .mcp_tools package for automatic Gradio API registration.
"""

from .mcp_tools import (
    MCP_FUNCTIONS,
    VideoGen_LTX_2_5_get_task_list,
    VideoGen_LTX_2_5_run,
    VideoGen_LTX_2_5_get_task_status,
)

__all__ = [
    "MCP_FUNCTIONS",
    "VideoGen_LTX_2_5_get_task_list",
    "VideoGen_LTX_2_5_run",
    "VideoGen_LTX_2_5_get_task_status",
]
