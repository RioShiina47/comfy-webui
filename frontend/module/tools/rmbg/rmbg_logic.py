import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, save_temp_video

WORKFLOW_RECIPE_PATH = "rmbg_recipe.yaml"

def process_inputs(ui_values):
    local_ui_values = ui_values.copy()
    is_video = local_ui_values.get('input_type') == "Video"
    
    input_file_obj = local_ui_values.get('input_video') if is_video else local_ui_values.get('input_image')
    if input_file_obj is None:
        raise ValueError(f"Please provide an input {local_ui_values.get('input_type').lower()}.")
    
    unique_prefix = get_filename_prefix()
    local_ui_values['filename_prefix_result'] = f"{unique_prefix}_result"
    local_ui_values['filename_prefix_mask'] = f"{unique_prefix}_mask"
    
    if is_video:
        local_ui_values['input_video_filename'] = save_temp_video(input_file_obj)
        metadata = get_media_metadata(input_file_obj, is_video=True)
        local_ui_values['fps'] = metadata.get('fps', 30)
    else:
        local_ui_values['input_image_filename'] = save_temp_image(input_file_obj)
        
    recipe_path = WORKFLOW_RECIPE_PATH
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)

    if is_video:
        workflow[assembler.node_map['rmbg_node']]['inputs']['image'] = [assembler.node_map['get_frames'], 0]
    else:
        workflow[assembler.node_map['rmbg_node']]['inputs']['image'] = [assembler.node_map['load_image'], 0]
        
    return workflow, None