"""
MCP Tool: get_task_status
Query the processing progress and final results of an async image generation task.
"""

from .common import _TASKS_DB
from .error_schema import make_validation_error, make_not_found_error


def ImageGen_get_task_status(task_id: str) -> dict:
    """Query the processing progress and final results of an async image generation task."""
    if not task_id:
        return make_validation_error(
            "Parameter 'task_id' is required.",
            missing_fields=["task_id"],
        )

    if task_id not in _TASKS_DB:
        return make_not_found_error("task", task_id)

    return _TASKS_DB[task_id]
