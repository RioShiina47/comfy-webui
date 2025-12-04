import os
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core import node_info_manager
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, save_temp_video

WORKFLOW_RECIPE_PATH = "controlnet_base_recipe.yaml"
MAX_DYNAMIC_CONTROLS = 8

def make_even(n):
    return n if n % 2 == 0 else n + 1

def process_inputs(ui_values):
    is_video = ui_values.get('input_type') == "Video"
    preprocessor_name = ui_values.get('preprocessor_name')
    if not preprocessor_name: raise ValueError("Please select a preprocessor.")

    input_file_obj = ui_values.get('input_video') if is_video else ui_values.get('input_image')
    if input_file_obj is None: raise ValueError("Please provide an input image or video.")

    metadata = get_media_metadata(input_file_obj, is_video=is_video)
    w, h, fps = metadata['width'], metadata['height'], metadata['fps']
    if w == 0 or h == 0: raise ValueError("Could not get the dimensions of the input file.")

    node_info = node_info_manager.get_node_info(preprocessor_name)
    resolution_config = node_info.get("input", {}).get("optional", {}).get("resolution", [None, {}])[1]
    final_resolution = resolution_config.get("default", max(w,h))

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)

    local_ui_values = {}
    if is_video:
        local_ui_values['input_video_filename'] = save_temp_video(input_file_obj)
    else:
        local_ui_values['input_image_filename'] = save_temp_image(input_file_obj)
        
    local_ui_values['filename_prefix'] = get_filename_prefix()
    workflow = assembler.assemble(local_ui_values)
    
    preprocessor_id = assembler._get_unique_id()
    preprocessor_node = assembler._get_node_template_from_api(preprocessor_name)
    
    preprocessor_node['_meta']['title'] = node_info.get("display_name", preprocessor_name)
    preprocessor_node['inputs']['resolution'] = final_resolution
    
    params_in_node = node_info.get("input", {}).get("optional", {})
    
    sliders_params, combos_params, checkboxes_params = [], [], []
    for name, details in params_in_node.items():
        if name in ["resolution", "image"]: continue
        param_type = details[0]
        is_bool_combo = isinstance(param_type, list) and set(s.lower() for s in param_type) == {'enable', 'disable'}
        if isinstance(param_type, str) and param_type.upper() in ["INT", "FLOAT"]: sliders_params.append(name)
        elif isinstance(param_type, list) and not is_bool_combo: combos_params.append(name)
        elif is_bool_combo: checkboxes_params.append(name)
        
    param_sliders_list = ui_values.get('param_sliders_list', [])
    for i, name in enumerate(sliders_params):
        if i < len(param_sliders_list):
            preprocessor_node['inputs'][name] = param_sliders_list[i]
            
    param_combos_list = ui_values.get('param_combos_list', [])
    for i, name in enumerate(combos_params):
        if i < len(param_combos_list):
            preprocessor_node['inputs'][name] = param_combos_list[i]
            
    param_checkboxes_list = ui_values.get('param_checkboxes_list', [])
    for i, name in enumerate(checkboxes_params):
        if i < len(param_checkboxes_list):
            is_enabled = param_checkboxes_list[i]
            preprocessor_node['inputs'][name] = "enable" if is_enabled else "disable"

    workflow[preprocessor_id] = preprocessor_node

    input_node_id = assembler.node_map['get_frames'] if is_video else assembler.node_map['load_image']
    workflow[preprocessor_id]['inputs']['image'] = [input_node_id, 0]

    output_types = node_info.get("output", ["IMAGE"])
    for i, out_type in enumerate(output_types):
        if i >= 2: break 

        current_image_source = [preprocessor_id, i]
        
        if out_type == "MASK":
            mask_converter_id = assembler.node_map['mask_to_image_converter']
            workflow[mask_converter_id]['inputs']['mask'] = current_image_source
            current_image_source = [mask_converter_id, 0]
        
        save_node_params = { 'filename_prefix': f"{local_ui_values['filename_prefix']}_out{i+1}" }

        if is_video:
            resize_id, create_id, save_id = [assembler.node_map[f'{name}_{i+1}'] for name in ['resize_frames', 'create_video', 'save_video']]
            workflow[resize_id]['inputs'].update({'image': current_image_source, 'target_width': make_even(w), 'target_height': make_even(h)})
            workflow[create_id]['inputs'].update({'images': [resize_id, 0], 'fps': fps})
            workflow[save_id]['inputs'].update({'video': [create_id, 0], **save_node_params})
        else:
            save_id = assembler.node_map[f'save_image_{i+1}']
            workflow[save_id]['inputs'].update({'images': current_image_source, **save_node_params})
            
    return workflow, None