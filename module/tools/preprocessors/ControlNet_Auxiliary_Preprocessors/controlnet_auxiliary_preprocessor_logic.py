import os
from core.config import COMFYUI_INPUT_PATH
from core.utils import get_media_metadata
from core import node_info_manager
from core.workflow_assembler import WorkflowAssembler
from core.utils import get_filename_prefix
from core.utils import save_temp_image, save_temp_video

IMAGE_RECIPE_PATH = "controlnet_image_recipe.yaml"
VIDEO_RECIPE_PATH = "controlnet_video_recipe.yaml"
MAX_DYNAMIC_CONTROLS = 8

def make_even(n):
    return n if n % 2 == 0 else n + 1

def process_inputs(ui_values):
    is_video = ui_values.get('input_type') == "Video"
    preprocessor_name = ui_values.get('preprocessor_name')
    if not preprocessor_name: raise ValueError("Please select a preprocessor.")

    input_file_obj = ui_values.get('input_video') if is_video else ui_values.get('input_image')
    if input_file_obj is None: raise ValueError("Please provide an input image or video.")

    node_info = node_info_manager.get_node_info(preprocessor_name)
    resolution_config = node_info.get("input", {}).get("optional", {}).get("resolution", [None, {}])[1]
    metadata = get_media_metadata(input_file_obj, is_video=is_video)
    final_resolution = resolution_config.get("default", max(metadata['width'], metadata['height']))

    module_path = os.path.dirname(os.path.abspath(__file__))
    
    local_ui_values = {}
    if is_video:
        recipe_to_load = VIDEO_RECIPE_PATH
        local_ui_values['input_video_filename'] = save_temp_video(input_file_obj)
    else:
        recipe_to_load = IMAGE_RECIPE_PATH
        local_ui_values['input_image_filename'] = save_temp_image(input_file_obj)
    
    assembler = WorkflowAssembler(recipe_to_load, base_path=module_path)
        
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
    
    nodes_in_recipe = set(assembler.node_map.keys())
    
    for i, out_type in enumerate(output_types):
        output_index = i + 1
        if output_index > 2: break 

        current_image_source = [preprocessor_id, i]
        
        if out_type == "MASK":
            if 'mask_to_image_converter' in nodes_in_recipe:
                mask_converter_id = assembler.node_map['mask_to_image_converter']
                workflow[mask_converter_id]['inputs']['mask'] = current_image_source
                current_image_source = [mask_converter_id, 0]
            else:
                print(f"Warning: Output {i} is a MASK, but 'mask_to_image_converter' not found in recipe. Skipping.")
                continue
        
        save_node_params = { 'filename_prefix': f"{local_ui_values['filename_prefix']}_out{output_index}" }

        if is_video:
            create_node_name = f'create_video_{output_index}'
            save_node_name = f'save_video_{output_index}'
            
            if create_node_name in nodes_in_recipe and save_node_name in nodes_in_recipe:
                create_id = assembler.node_map[create_node_name]
                save_id = assembler.node_map[save_node_name]
                
                workflow[create_id]['inputs']['images'] = current_image_source
                workflow[save_id]['inputs'].update(save_node_params)
            else:
                print(f"Warning: Nodes for output {output_index} ('{create_node_name}', '{save_node_name}') not found in video recipe. Skipping this output.")
        else:
            save_node_name = f'save_image_{output_index}'
            if save_node_name in nodes_in_recipe:
                save_id = assembler.node_map[save_node_name]
                workflow[save_id]['inputs'].update({'images': current_image_source, **save_node_params})
            else:
                 print(f"Warning: Node '{save_node_name}' for output {output_index} not found in image recipe. Skipping this output.")

    all_inputs = set()
    for node in workflow.values():
        for input_val in node['inputs'].values():
            if isinstance(input_val, list) and len(input_val) == 2 and isinstance(input_val[0], str):
                all_inputs.add(input_val[0])
    
    final_output_node_types = ("SaveImage", "SaveVideo")
    
    all_node_ids = list(workflow.keys())
    
    for node_id in all_node_ids:
        if node_id not in workflow:
            continue
            
        node_info = workflow[node_id]
        if node_id not in all_inputs and node_info['class_type'] not in final_output_node_types:
            print(f"Cleaning up unused node: ID {node_id}, Title '{node_info['_meta']['title']}'")
            del workflow[node_id]

    return workflow, None