"""
MCP Tool Handlers — Backward-compatible aggregation entry point.
Core logic has been split into individual files (get_*.py and run_imagegen.py).
"""

from .get_task_list import ImageGen_get_task_list
from .get_model_architecture_list import ImageGen_get_model_architecture_list
from .get_model_list import ImageGen_get_model_list
from .get_feature_list import ImageGen_get_feature_list
from .get_model_features import ImageGen_get_model_features
from .get_chain_schema import ImageGen_get_chain_schema
from .run_imagegen import ImageGen_run_imagegen
from .get_task_status import ImageGen_get_task_status
from .common import (
    _TASK_DEFINITIONS,
    _TASKS_DB,
    _load_yaml,
    _execute_imagegen_pipeline,
)

__all__ = [
    "ImageGen_get_task_list",
    "ImageGen_get_model_architecture_list",
    "ImageGen_get_model_list",
    "ImageGen_get_feature_list",
    "ImageGen_get_model_features",
    "ImageGen_get_chain_schema",
    "ImageGen_run_imagegen",
    "ImageGen_get_task_status",
]
