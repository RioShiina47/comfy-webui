"""
Lightricks LTX-2.5 MCP Tools Package.
Exposes MCP tool functions for automated registration in comfy-webui.
"""

from .get_task_list import VideoGen_LTX_2_5_get_task_list
from .run import VideoGen_LTX_2_5_run
from .get_task_status import VideoGen_LTX_2_5_get_task_status
from .error_schema import make_error, make_validation_error, make_not_found_error

MCP_FUNCTIONS = [
    VideoGen_LTX_2_5_get_task_list,
    VideoGen_LTX_2_5_run,
    VideoGen_LTX_2_5_get_task_status,
]

__all__ = [
    "VideoGen_LTX_2_5_get_task_list",
    "VideoGen_LTX_2_5_run",
    "VideoGen_LTX_2_5_get_task_status",
    "make_error",
    "make_validation_error",
    "make_not_found_error",
    "MCP_FUNCTIONS",
]
