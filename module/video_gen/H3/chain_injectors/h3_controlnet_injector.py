def inject(assembler, chain_definition, chain_items):
    """
    Dynamically injects H3 Fun ControlNet into the model pipeline.
    Connects:
      LoadVideo -> GetVideoComponents -> MiniMaxH3FunControlNetApply (control_video)
      ModelPatchLoader (name) -> MiniMaxH3FunControlNetApply (model_patch)
      Video VAE -> MiniMaxH3FunControlNetApply (vae)
      Current Model -> MiniMaxH3FunControlNetApply (model) -> Target Nodes (basic_guider, basic_scheduler)
    """
    if not chain_items:
        return

    def get_template(class_type):
        if hasattr(assembler, '_get_node_template_from_api'):
            return assembler._get_node_template_from_api(class_type)
        elif hasattr(assembler, '_get_node_template'):
            return assembler._get_node_template(class_type)
        raise AttributeError("WorkflowAssembler does not provide a node template retriever.")

    # 1. Determine target nodes to receive the modified model output
    target_nodes_def = chain_definition.get('target_nodes', [
        'basic_guider:model',
        'basic_scheduler:model'
    ])
    
    # Locate current model connection from the first available target node
    current_model_connection = None
    for target_str in target_nodes_def:
        if ':' in target_str:
            node_name, input_name = target_str.split(':')
            node_id = assembler.node_map.get(node_name)
            if node_id and input_name in assembler.workflow.get(node_id, {}).get('inputs', {}):
                current_model_connection = assembler.workflow[node_id]['inputs'][input_name]
                break

    if not current_model_connection:
        # Fallback to unet_loader:0
        unet_node_id = assembler.node_map.get('unet_loader')
        if unet_node_id:
            current_model_connection = [unet_node_id, 0]
        else:
            print("[H3 ControlNet Injector] Warning: Could not determine initial model connection. Skipping.")
            return

    # 2. Resolve VAE connection
    vae_source_str = chain_definition.get('vae_source', 'video_vae_loader:0')
    vae_connection = None
    if vae_source_str and ':' in vae_source_str:
        vae_node_name, vae_idx_str = vae_source_str.split(':')
        vae_node_id = assembler.node_map.get(vae_node_name)
        if vae_node_id:
            vae_connection = [vae_node_id, int(vae_idx_str)]
        else:
            print(f"[H3 ControlNet Injector] Warning: VAE source '{vae_node_name}' not found in node_map.")

    # 3. Iterate through active controlnet items and chain them
    applied_count = 0
    for item_data in chain_items:
        video_file = item_data.get('video') or item_data.get('control_video') or item_data.get('file')
        if not video_file:
            continue

        control_net_name = (
            item_data.get('control_net_name') or 
            item_data.get('filepath') or 
            item_data.get('name') or 
            'minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors'
        )
        strength = float(item_data.get('strength', 1.0))
        start_percent = float(item_data.get('start_percent', 0.0))
        end_percent = float(item_data.get('end_percent', 1.0))

        # Node: LoadVideo
        load_vid_id = assembler._get_unique_id()
        load_vid_node = get_template("LoadVideo")
        load_vid_node['inputs']['file'] = video_file
        if 'video-preview' in load_vid_node['inputs']:
            load_vid_node['inputs']['video-preview'] = ""
        load_vid_node['_meta'] = {"title": "Load Video"}
        assembler.workflow[load_vid_id] = load_vid_node

        # Node: GetVideoComponents
        comp_id = assembler._get_unique_id()
        comp_node = get_template("GetVideoComponents")
        comp_node['inputs']['video'] = [load_vid_id, 0]
        comp_node['_meta'] = {"title": "Extract Control Frames"}
        assembler.workflow[comp_id] = comp_node

        # Node: ModelPatchLoader
        patch_id = assembler._get_unique_id()
        patch_node = get_template("ModelPatchLoader")
        patch_node['inputs']['name'] = control_net_name
        patch_node['_meta'] = {"title": "Load Model Patch"}
        assembler.workflow[patch_id] = patch_node

        # Node: MiniMaxH3FunControlNetApply
        apply_template_name = chain_definition.get('template', 'MiniMaxH3FunControlNetApply')
        apply_id = assembler._get_unique_id()
        apply_node = get_template(apply_template_name)
        apply_node['inputs']['strength'] = strength
        apply_node['inputs']['start_percent'] = start_percent
        apply_node['inputs']['end_percent'] = end_percent
        apply_node['inputs']['model'] = current_model_connection
        apply_node['inputs']['model_patch'] = [patch_id, 0]
        if vae_connection:
            apply_node['inputs']['vae'] = vae_connection
        apply_node['inputs']['control_video'] = [comp_id, 0]
        apply_node['_meta'] = {"title": "Apply MiniMax H3 Fun ControlNet"}
        assembler.workflow[apply_id] = apply_node

        current_model_connection = [apply_id, 0]
        applied_count += 1

    # 4. Reconnect target nodes to the final model connection
    if applied_count > 0:
        for target_str in target_nodes_def:
            if ':' in target_str:
                node_name, input_name = target_str.split(':')
                node_id = assembler.node_map.get(node_name)
                if node_id and node_id in assembler.workflow:
                    assembler.workflow[node_id]['inputs'][input_name] = current_model_connection
        print(f"[H3 ControlNet Injector] Successfully applied {applied_count} H3 ControlNet patch(es).")
