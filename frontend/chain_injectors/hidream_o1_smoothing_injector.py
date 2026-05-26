from copy import deepcopy

def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    target_node_name = chain_definition.get('target_node')
    if not target_node_name or target_node_name not in assembler.node_map:
        print(f"Warning: Target node '{target_node_name}' not found for HiDream-O1 Smoothing. Skipping.")
        return

    target_node_id = assembler.node_map[target_node_name]
    
    if 'model' not in assembler.workflow[target_node_id]['inputs']:
        print(f"Warning: Target node '{target_node_name}' has no 'model' input. Skipping.")
        return

    current_model_connection = assembler.workflow[target_node_id]['inputs']['model']

    for _ in chain_items:
        template = assembler._get_node_template_from_api("HiDreamO1PatchSeamSmoothing")
        node_data = deepcopy(template)
        
        node_data['inputs']['start_percent'] = 0.8
        node_data['inputs']['end_percent'] = 1.0
        node_data['inputs']['pattern'] = "single_shift"
        node_data['inputs']['passes'] = "ramp_2_4"
        node_data['inputs']['blend'] = "median"
        node_data['inputs']['strength'] = 1.0
        
        node_data['inputs']['model'] = current_model_connection
        
        new_node_id = assembler._get_unique_id()
        assembler.workflow[new_node_id] = node_data
        
        current_model_connection = [new_node_id, 0]

    assembler.workflow[target_node_id]['inputs']['model'] = current_model_connection
    print("HiDream-O1 Patch Seam Smoothing injector applied.")