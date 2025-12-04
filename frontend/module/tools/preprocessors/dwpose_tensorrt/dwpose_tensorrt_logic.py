import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, save_temp_video

WORKFLOW_RECIPE_PATH = "dwpose_tensorrt_recipe.yaml"
PRECISION_CHOICES = ["fp16", "fp32"]

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    is_video = local_ui_values.get('input_type') == "Video"
    
    input_file_obj = local_ui_values.get('input_video') if is_video else local_ui_values.get('input_image')
    if input_file_obj is None:
        raise ValueError(f"Please provide an input {local_ui_values.get('input_type').lower()}.")
    
    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    if is_video:
        local_ui_values['input_video_filename'] = save_temp_video(input_file_obj)
        metadata = get_media_metadata(input_file_obj, is_video=True)
        local_ui_values['fps'] = metadata.get('fps', 30)
    else:
        local_ui_values['input_image_filename'] = save_temp_image(input_file_obj)
    
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None