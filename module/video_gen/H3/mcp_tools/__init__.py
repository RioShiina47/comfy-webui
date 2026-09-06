"""
MiniMax-H3 MCP Tools Package.
Exposes MCP tool functions for automated registration in comfy-webui.
"""

from .get_task_list import VideoGen_H3_get_task_list
from .run import VideoGen_H3_run
from .get_task_status import VideoGen_H3_get_task_status
from .error_schema import make_error, make_validation_error, make_not_found_error

MCP_FUNCTIONS = [
    VideoGen_H3_get_task_list,
    VideoGen_H3_run,
    VideoGen_H3_get_task_status,
]

__all__ = [
    "VideoGen_H3_get_task_list",
    "VideoGen_H3_run",
    "VideoGen_H3_get_task_status",
    "make_error",
    "make_validation_error",
    "make_not_found_error",
    "MCP_FUNCTIONS",
]
