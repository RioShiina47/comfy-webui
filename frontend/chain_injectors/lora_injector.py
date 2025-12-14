from copy import deepcopy

def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    start_node_name = chain_definition.get('start')
    start_node_id = None
    if start_node_name:
        if start_node_name not in assembler.node_map:
            print(f"Warning: Start node '{start_node_name}' for dynamic LoRA chain not found. Skipping chain.")
            return
        start_node_id = assembler.node_map[start_node_name]
    
    output_map = chain_definition.get('output_map', {})
    current_connections = {}
    for key, type_name in output_map.items():
        if ':' in str(key):
            node_name, idx_str = key.split(':')
            if node_name not in assembler.node_map:
                print(f"Warning: Node '{node_name}' in chain's output_map not found. Skipping.")
                continue
            node_id = assembler.node_map[node_name]
            start_output_idx = int(idx_str)
            current_connections[type_name] = [node_id, start_output_idx]
        elif start_node_id:
            start_output_idx = int(key)
            current_connections[type_name] = [start_node_id, start_output_idx]
        else:
            print(f"Warning: LoRA chain has no 'start' node defined, and an output_map key '{key}' is not in 'node:index' format. Skipping this connection.")


    input_map = chain_definition.get('input_map', {})
    chain_output_map = chain_definition.get('template_output_map', { "0": "model", "1": "clip" })

    for item_data in chain_items:
        template_name = chain_definition['template']
        template = assembler._get_node_template_from_api(template_name)
        node_data = deepcopy(template)
        
        for param_name, value in item_data.items():
            if param_name in node_data['inputs']:
                node_data['inputs'][param_name] = value
        
        for type_name, input_name in input_map.items():
            if type_name in current_connections:
                node_data['inputs'][input_name] = current_connections[type_name]

        new_node_id = assembler._get_unique_id()
        assembler.workflow[new_node_id] = node_data
        
        for idx_str, type_name in chain_output_map.items():
            current_connections[type_name] = [new_node_id, int(idx_str)]

    end_input_map = chain_definition.get('end_input_map', {})
    for type_name, targets in end_input_map.items():
        if type_name in current_connections:
            if not isinstance(targets, list):
                targets = [targets]
            
            for target_str in targets:
                end_node_name, end_input_name = target_str.split(':')
                if end_node_name in assembler.node_map:
                    end_node_id = assembler.node_map[end_node_name]
                    assembler.workflow[end_node_id]['inputs'][end_input_name] = current_connections[type_name]
                else:
                    print(f"Warning: End node '{end_node_name}' for dynamic chain not found. Skipping connection.")