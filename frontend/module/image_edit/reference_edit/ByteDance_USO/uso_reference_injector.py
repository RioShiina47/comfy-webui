def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    start_node_name = chain_definition.get('start')
    if not start_node_name or start_node_name not in assembler.node_map:
        print(f"Warning: Start node '{start_node_name}' for USO chain not found in recipe. Skipping.")
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
                print(f"Warning: Node '{node_name}' in USO output_map not found. Skipping chain.")
                return
        else:
            current_connections[type_name] = [start_node_id, int(key)]

    if "conditioning" not in current_connections:
        print("Warning: Could not find start conditioning for USO chain. Skipping.")
        return

    vae_node_id = assembler.node_map.get('vae_loader')
    if not vae_node_id:
        print("Warning: VAE loader node not found in map for USO chain. Skipping.")
        return

    for item_data in chain_items:
        img_loader_id = assembler._get_unique_id()
        img_loader_node = assembler._get_node_template_from_api("LoadImage")
        img_loader_node['inputs']['image'] = item_data['image']
        assembler.workflow[img_loader_id] = img_loader_node

        image_scaler_id = assembler._get_unique_id()
        image_scaler_node = assembler._get_node_template_from_api("ImageScaleToMaxDimension")
        image_scaler_node['inputs']['largest_size'] = 512
        image_scaler_node['inputs']['upscale_method'] = "area"
        image_scaler_node['inputs']['image'] = [img_loader_id, 0]
        assembler.workflow[image_scaler_id] = image_scaler_node

        vae_encode_id = assembler._get_unique_id()
        vae_encode_node = assembler._get_node_template_from_api("VAEEncode")
        vae_encode_node['inputs']['pixels'] = [image_scaler_id, 0]
        vae_encode_node['inputs']['vae'] = [vae_node_id, 0]
        assembler.workflow[vae_encode_id] = vae_encode_node

        ref_latent_id = assembler._get_unique_id()
        ref_latent_node = assembler._get_node_template_from_api("ReferenceLatent")
        ref_latent_node['inputs']['conditioning'] = current_connections['conditioning']
        ref_latent_node['inputs']['latent'] = [vae_encode_id, 0]
        assembler.workflow[ref_latent_id] = ref_latent_node

        kontext_method_id = assembler._get_unique_id()
        kontext_method_node = assembler._get_node_template_from_api("FluxKontextMultiReferenceLatentMethod")
        kontext_method_node['inputs']['reference_latents_method'] = "uxo/uno"
        kontext_method_node['inputs']['conditioning'] = [ref_latent_id, 0]
        assembler.workflow[kontext_method_id] = kontext_method_node
        
        flux_guidance_id = assembler._get_unique_id()
        flux_guidance_node = assembler._get_node_template_from_api("FluxGuidance")
        flux_guidance_node['inputs']['guidance'] = item_data['guidance']
        flux_guidance_node['inputs']['conditioning'] = [kontext_method_id, 0]
        assembler.workflow[flux_guidance_id] = flux_guidance_node

        current_connections['conditioning'] = [flux_guidance_id, 0]

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
                    print(f"Warning: End node '{end_node_name}' for dynamic USO chain not found.")