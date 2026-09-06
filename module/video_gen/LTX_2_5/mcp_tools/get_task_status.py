"""
MCP Tool: VideoGen_LTX_2_5_get_task_status
Query the processing progress, status, and result of an async Lightricks LTX-2.5 video generation task.
"""

from .common import _TASKS_DB
from .error_schema import make_validation_error, make_not_found_error


def VideoGen_LTX_2_5_get_task_status(task_id: str) -> dict:
    """
    Query the processing progress and final results of an async Lightricks LTX-2.5 video generation task.

    Args:
        task_id (str): The unique task ID returned when submitting an async task.

    Returns:
        dict: The current task object containing status ("queued", "processing", "completed", "failed"),
              progress (0-100), and result (with generated video URLs) if completed.
    """
    if not task_id:
        return make_validation_error(
            "Parameter 'task_id' is required.",
            missing_fields=["task_id"],
        )

    if task_id not in _TASKS_DB:
        return make_not_found_error("task", task_id)

    return _TASKS_DB[task_id]
