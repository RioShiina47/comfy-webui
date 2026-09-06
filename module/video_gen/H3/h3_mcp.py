"""
MCP Module for MiniMax-H3 Video Generation in comfy-webui.
Exposes MCP_FUNCTIONS from .mcp_tools package for automatic Gradio API registration.
"""

from .mcp_tools import (
    MCP_FUNCTIONS,
    VideoGen_H3_get_task_list,
    VideoGen_H3_run,
    VideoGen_H3_get_task_status,
)

__all__ = [
    "MCP_FUNCTIONS",
    "VideoGen_H3_get_task_list",
    "VideoGen_H3_run",
    "VideoGen_H3_get_task_status",
]
