"""
MCP Tool: VideoGen_LTX_2_5_get_task_list
Get a list of all supported Lightricks LTX-2.5 video generation task types along with parameter specifications.
"""

from .common import _TASK_DEFINITIONS


def VideoGen_LTX_2_5_get_task_list() -> list:
    """
    Get a list of all supported Lightricks LTX-2.5 video generation task types along with parameter schemas.

    Supported task types:
    - t2va: Text-to-Video & Audio
    - i2va: Image-to-Video & Audio (Starting Frame)
    - ta2va: Text & Audio-to-Video & Audio (Audio Driven)
    - ia2va: Image & Audio-to-Video & Audio
    - flf2va: First & Last Frame-to-Video & Audio (Keyframe Interpolation)
    """
    return _TASK_DEFINITIONS
