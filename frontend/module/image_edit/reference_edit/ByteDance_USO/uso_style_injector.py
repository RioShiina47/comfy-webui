def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    start_node_name = chain_definition.get('start')
    if not start_node_name or start_node_name not in assembler.node_map:
        print(f"Warning: Start node '{start_node_name}' for USO Style chain not found in recipe. Skipping.")
        return
    start_node_id = assembler.node_map[start_node_name]

    output_map = chain_definition.get('output_map', {})
    current_connections = {}
    for key, type_name in output_map.items():
        if ':' in str(key):
            node_name, idx_str = key.split(':')
            if node_name in assembler.node_map:
                node_id = assembler.node_map[node_name]
                current_connections[type_name] = [node_id, int(idx_str)]
            else:
                print(f"Warning: Node '{node_name}' in USO Style output_map not found. Skipping chain.")
                return
        else:
            current_connections[type_name] = [start_node_id, int(key)]
    
    if "model" not in current_connections:
        print("Warning: Could not find start model for USO Style chain. Skipping.")
        return

    clip_vision_loader_id = assembler._get_unique_id()
    clip_vision_loader_node = assembler._get_node_template_from_api("CLIPVisionLoader")
    clip_vision_loader_node['inputs']['clip_name'] = "sigclip_vision_patch14_384.safetensors"
    assembler.workflow[clip_vision_loader_id] = clip_vision_loader_node
    
    model_patch_loader_id = assembler._get_unique_id()
    model_patch_loader_node = assembler._get_node_template_from_api("ModelPatchLoader")
    model_patch_loader_node['inputs']['name'] = "uso-flux1-projector-v1.safetensors"
    assembler.workflow[model_patch_loader_id] = model_patch_loader_node

    for item_data in chain_items:
        img_loader_id = assembler._get_unique_id()
        img_loader_node = assembler._get_node_template_from_api("LoadImage")
        img_loader_node['inputs']['image'] = item_data['image']
        assembler.workflow[img_loader_id] = img_loader_node

        clip_vision_encode_id = assembler._get_unique_id()
        clip_vision_encode_node = assembler._get_node_template_from_api("CLIPVisionEncode")
        clip_vision_encode_node['inputs']['crop'] = "center"
        clip_vision_encode_node['inputs']['clip_vision'] = [clip_vision_loader_id, 0]
        clip_vision_encode_node['inputs']['image'] = [img_loader_id, 0]
        assembler.workflow[clip_vision_encode_id] = clip_vision_encode_node

        style_ref_id = assembler._get_unique_id()
        style_ref_node = assembler._get_node_template_from_api("USOStyleReference")
        style_ref_node['inputs']['model'] = current_connections['model']
        style_ref_node['inputs']['model_patch'] = [model_patch_loader_id, 0]
        style_ref_node['inputs']['clip_vision_output'] = [clip_vision_encode_id, 0]
        assembler.workflow[style_ref_id] = style_ref_node

        current_connections['model'] = [style_ref_id, 0]

    end_input_map = chain_definition.get('end_input_map', {})
    for type_name, targets in end_input_map.items():
        if type_name in current_connections:
            if not isinstance(targets, list): targets = [targets]
            for target_str in targets:
                end_node_name, end_input_name = target_str.split(':')
                if end_node_name in assembler.node_map:
                    end_node_id = assembler.node_map[end_node_name]
                    assembler.workflow[end_node_id]['inputs'][end_input_name] = current_connections[type_name]
                else:
                    print(f"Warning: End node '{end_node_name}' for dynamic USO Style chain not found.")