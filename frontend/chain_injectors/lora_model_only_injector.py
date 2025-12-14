from copy import deepcopy

def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    output_map = chain_definition.get('output_map', {})
    
    current_model_connection = None
    for key, type_name in output_map.items():
        if type_name == "model":
            node_name, idx_str = key.split(':')
            if node_name in assembler.node_map:
                node_id = assembler.node_map[node_name]
                current_model_connection = [node_id, int(idx_str)]
                break
    
    if not current_model_connection:
        print(f"Warning: 'output_map' for LoRA (Model Only) chain must define a 'model' type from a valid node. Skipping.")
        return

    template_name = chain_definition.get('template')
    if not template_name:
        print(f"Warning: No 'template' defined for LoRA (Model Only) chain. Skipping.")
        return

    for item_data in chain_items:
        template = assembler._get_node_template_from_api(template_name)
        node_data = deepcopy(template)
        
        for param_name, value in item_data.items():
            if param_name in node_data['inputs']:
                node_data['inputs'][param_name] = value
        
        node_data['inputs']['model'] = current_model_connection

        new_node_id = assembler._get_unique_id()
        assembler.workflow[new_node_id] = node_data
        
        current_model_connection = [new_node_id, 0]

    end_input_map = chain_definition.get('end_input_map', {})
    model_targets = end_input_map.get('model', [])
    if not isinstance(model_targets, list):
        model_targets = [model_targets]

    for target_str in model_targets:
        try:
            end_node_name, end_input_name = target_str.split(':')
            if end_node_name in assembler.node_map:
                end_node_id = assembler.node_map[end_node_name]
                assembler.workflow[end_node_id]['inputs'][end_input_name] = current_model_connection
            else:
                print(f"Warning: End node '{end_node_name}' for dynamic LoRA (Model Only) chain not found. Skipping connection.")
        except ValueError:
            print(f"Warning: Invalid target format '{target_str}' in end_input_map. Skipping.")
            
    if model_targets:
        print(f"LoRA (Model Only) injector applied. Re-routed model through {len(chain_items)} LoRA(s).")