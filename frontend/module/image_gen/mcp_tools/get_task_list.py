"""
MCP Tool: get_task_list
Get a list of all supported image generation task types along with their required/optional parameter lists.
"""

from .common import _TASK_DEFINITIONS


def ImageGen_get_task_list() -> list:
    """Get a list of all supported image generation task types along with their required/optional parameter lists."""
    return _TASK_DEFINITIONS
