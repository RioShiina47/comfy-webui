"""
MCP Tool: VideoGen_H3_get_task_list
Get a list of all supported MiniMax-H3 video generation task types along with parameter specifications.
"""

from .common import _TASK_DEFINITIONS


def VideoGen_H3_get_task_list() -> list:
    """
    Get a list of all supported MiniMax-H3 video generation task types along with parameter schemas.
    
    Supported task types:
    - t2va: Text-to-Video & Audio
    - i2va: Image-to-Video & Audio (First Frame)
    - flf2va: First & Last Frame-to-Video & Audio
    - ref2va: Multi-modal Reference-to-Video & Audio
    """
    return _TASK_DEFINITIONS
