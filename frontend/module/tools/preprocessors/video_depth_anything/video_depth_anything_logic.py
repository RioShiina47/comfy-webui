import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_video

WORKFLOW_RECIPE_PATH = "video_depth_anything_recipe.yaml"

MODEL_CHOICES = [
    "video_depth_anything_vits.pth",
    "video_depth_anything_vitb.pth",
    "video_depth_anything_vitl.pth"
]
PRECISION_CHOICES = ["fp16", "fp32"]
COLORMAP_CHOICES = ["gray", "viridis", "plasma", "inferno", "magma", "cividis", "turbo"]

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    
    input_video = local_ui_values.get('input_video')
    if input_video is None:
        raise ValueError("Please provide an input video.")
    
    local_ui_values['filename_prefix'] = get_filename_prefix()
    local_ui_values['input_video_filename'] = save_temp_video(input_video)
    
    metadata = get_media_metadata(input_video, is_video=True)
    local_ui_values['fps'] = metadata.get('fps', 30)
    
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None