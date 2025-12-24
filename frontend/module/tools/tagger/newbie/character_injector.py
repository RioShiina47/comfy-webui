from copy import deepcopy

def inject(assembler, chain_definition, chain_items):
    """
    Dynamically injects NewBieCharacterBuilder nodes and connects them to the XML assembler.
    """
    if not chain_items:
        return

    target_node_name = chain_definition.get('target_node')
    if not target_node_name or target_node_name not in assembler.node_map:
        print(f"Warning: [Character Injector] Target node '{target_node_name}' not found. Skipping.")
        return
        
    target_node_id = assembler.node_map[target_node_name]
    
    for i, item_data in enumerate(chain_items):
        char_num = i + 1
        if char_num > 4:
            break

        template = assembler._get_node_template_from_api("NewBieCharacterBuilder")
        node_data = deepcopy(template)
        
        for param_name, value in item_data.items():
            if param_name in node_data['inputs']:
                node_data['inputs'][param_name] = value
        
        new_node_id = assembler._get_unique_id()
        assembler.workflow[new_node_id] = node_data
        
        target_input_name = f'character_{char_num}'
        if target_input_name in assembler.workflow[target_node_id]['inputs']:
            assembler.workflow[target_node_id]['inputs'][target_input_name] = [new_node_id, 0]
        else:
            print(f"Warning: [Character Injector] Input '{target_input_name}' not found on target node '{target_node_name}'.")

    print(f"Character injector applied. Injected {len(chain_items)} character(s).")