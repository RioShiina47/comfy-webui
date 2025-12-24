from copy import deepcopy

def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    output_map = chain_definition.get('output_map', {})
    current_connections = {}
    for key, type_name in output_map.items():
        if ':' in str(key):
            node_name, idx_str = key.split(':')
            if node_name not in assembler.node_map:
                print(f"Warning: [NewBie LoRA Injector] Node '{node_name}' in chain's output_map not found. Skipping.")
                continue
            node_id = assembler.node_map[node_name]
            start_output_idx = int(idx_str)
            current_connections[type_name] = [node_id, start_output_idx]
        else:
            print(f"Warning: [NewBie LoRA Injector] output_map key '{key}' is not in 'node:index' format. Skipping this connection.")

    template_name = chain_definition.get('template')
    if not template_name:
        print(f"Warning: [NewBie LoRA Injector] No 'template' defined for chain. Skipping.")
        return

    for item_data in chain_items:
        template = assembler._get_node_template_from_api(template_name)
        node_data = deepcopy(template)
        
        node_data['inputs']['lora_name'] = item_data.get('lora_name')
        node_data['inputs']['strength'] = item_data.get('strength_model', 1.0)
        node_data['inputs']['enabled'] = True
        
        if 'model' in current_connections:
            node_data['inputs']['model'] = current_connections['model']
        if 'clip' in current_connections:
            node_data['inputs']['clip'] = current_connections['clip']

        new_node_id = assembler._get_unique_id()
        assembler.workflow[new_node_id] = node_data
        
        current_connections['model'] = [new_node_id, 0]
        current_connections['clip'] = [new_node_id, 1]

    end_input_map = chain_definition.get('end_input_map', {})
    for type_name, targets in end_input_map.items():
        if type_name in current_connections:
            if not isinstance(targets, list):
                targets = [targets]
            
            for target_str in targets:
                try:
                    end_node_name, end_input_name = target_str.split(':')
                    if end_node_name in assembler.node_map:
                        end_node_id = assembler.node_map[end_node_name]
                        assembler.workflow[end_node_id]['inputs'][end_input_name] = current_connections[type_name]
                    else:
                        print(f"Warning: [NewBie LoRA Injector] End node '{end_node_name}' for dynamic chain not found. Skipping connection.")
                except ValueError:
                    print(f"Warning: [NewBie LoRA Injector] Invalid target format '{target_str}' in end_input_map. Skipping.")

    if chain_items:
        print(f"NewBie LoRA injector applied. Re-routed model and clip through {len(chain_items)} LoRA(s).")