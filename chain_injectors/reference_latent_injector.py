def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    guider_node_name = chain_definition.get('guider_node')
    guider_target_inputs = chain_definition.get('guider_target_inputs', [])
    start_connections_map = chain_definition.get('start_connections', {})
    vae_node_name = chain_definition.get('vae_node', 'vae_loader')

    if guider_node_name and guider_node_name in assembler.node_map and guider_target_inputs:
        guider_id = assembler.node_map[guider_node_name]
        if vae_node_name not in assembler.node_map:
            print(f"Warning: VAE node '{vae_node_name}' not found for Guider chain. Skipping.")
            return
        vae_node_id = assembler.node_map[vae_node_name]
        
        print(f"ReferenceLatent injector targeting DualCFGGuider node '{guider_node_name}'.")

        current_connections = {}
        for target_input in guider_target_inputs:
            conn_str = start_connections_map.get(target_input)
            if not conn_str:
                print(f"Warning: No start connection defined for '{target_input}' in Guider chain. Skipping this input.")
                continue
            try:
                node_name, idx_str = conn_str.split(':')
                node_id = assembler.node_map[node_name]
                current_connections[target_input] = [node_id, int(idx_str)]
            except (ValueError, KeyError):
                print(f"Warning: Invalid start connection '{conn_str}' for '{target_input}'. Skipping.")

        encoded_latents = []
        for i, img_filename in enumerate(chain_items):
            load_id = assembler._get_unique_id()
            load_node = assembler._get_node_template_from_api("LoadImage")
            load_node['inputs']['image'] = img_filename
            assembler.workflow[load_id] = load_node
            
            scale_id = assembler._get_unique_id()
            scale_node = assembler._get_node_template_from_api("ImageScaleToTotalPixels")
            scale_node['inputs']['megapixels'] = 1.0
            scale_node['inputs']['upscale_method'] = "lanczos"
            scale_node['inputs']['image'] = [load_id, 0]
            assembler.workflow[scale_id] = scale_node
            
            vae_encode_id = assembler._get_unique_id()
            vae_encode_node = assembler._get_node_template_from_api("VAEEncode")
            vae_encode_node['inputs']['pixels'] = [scale_id, 0]
            vae_encode_node['inputs']['vae'] = [vae_node_id, 0]
            assembler.workflow[vae_encode_id] = vae_encode_node
            encoded_latents.append([vae_encode_id, 0])
        
        for target_input_name, start_connection in current_connections.items():
            current_chain_head = start_connection
            for i, latent_conn in enumerate(encoded_latents):
                ref_latent_id = assembler._get_unique_id()
                ref_latent_node = assembler._get_node_template_from_api("ReferenceLatent")
                ref_latent_node['inputs']['conditioning'] = current_chain_head
                ref_latent_node['inputs']['latent'] = latent_conn
                ref_latent_node['_meta']['title'] = f"{target_input_name} RefLatent {i+1}"
                assembler.workflow[ref_latent_id] = ref_latent_node
                current_chain_head = [ref_latent_id, 0]
            
            assembler.workflow[guider_id]['inputs'][target_input_name] = current_chain_head
            print(f"  - Input '{target_input_name}' of node '{guider_node_name}' re-routed through {len(chain_items)} reference images.")

        return

    flux_guidance_name = chain_definition.get('flux_guidance_node')
    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')

    if ksampler_name not in assembler.node_map:
        print(f"Warning: KSampler node '{ksampler_name}' not found for ReferenceLatent chain. Skipping.")
        return
    if vae_node_name not in assembler.node_map:
        print(f"Warning: VAE loader node '{vae_node_name}' not found for ReferenceLatent chain. Skipping.")
        return

    ksampler_id = assembler.node_map[ksampler_name]
    vae_node_id = assembler.node_map[vae_node_name]

    pos_target_node_id = None
    pos_target_input_name = None
    if flux_guidance_name and flux_guidance_name in assembler.node_map:
        flux_guidance_id = assembler.node_map[flux_guidance_name]
        if 'conditioning' in assembler.workflow[flux_guidance_id]['inputs']:
            pos_target_node_id = flux_guidance_id
            pos_target_input_name = 'conditioning'
            print(f"ReferenceLatent injector targeting FluxGuidance node '{flux_guidance_name}' for positive chain.")
    
    if not pos_target_node_id:
        if 'positive' in assembler.workflow[ksampler_id]['inputs']:
            pos_target_node_id = ksampler_id
            pos_target_input_name = 'positive'
            print(f"ReferenceLatent injector targeting KSampler node '{ksampler_name}' for positive chain.")
        else:
            print(f"Warning: Could not find a valid positive injection point for ReferenceLatent chain. Skipping.")
            return

    current_pos_conditioning = assembler.workflow[pos_target_node_id]['inputs'][pos_target_input_name]

    neg_target_node_id = ksampler_id
    neg_target_input_name = 'negative'
    if 'negative' not in assembler.workflow[neg_target_node_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' has no 'negative' input. Skipping negative ReferenceLatent chain.")
        neg_target_node_id = None
    
    current_neg_conditioning = None
    if neg_target_node_id:
        current_neg_conditioning = assembler.workflow[neg_target_node_id]['inputs'][neg_target_input_name]

    for i, img_filename in enumerate(chain_items):
        load_id = assembler._get_unique_id()
        load_node = assembler._get_node_template_from_api("LoadImage")
        load_node['inputs']['image'] = img_filename
        load_node['_meta']['title'] = f"Load Reference Image {i+1}"
        assembler.workflow[load_id] = load_node
        
        scale_id = assembler._get_unique_id()
        scale_node = assembler._get_node_template_from_api("ImageScaleToTotalPixels")
        scale_node['inputs']['megapixels'] = 1.0
        scale_node['inputs']['upscale_method'] = "lanczos"
        scale_node['inputs']['image'] = [load_id, 0]
        scale_node['_meta']['title'] = f"Scale Reference {i+1}"
        assembler.workflow[scale_id] = scale_node
        
        vae_encode_id = assembler._get_unique_id()
        vae_encode_node = assembler._get_node_template_from_api("VAEEncode")
        vae_encode_node['inputs']['pixels'] = [scale_id, 0]
        vae_encode_node['inputs']['vae'] = [vae_node_id, 0]
        vae_encode_node['_meta']['title'] = f"VAE Encode Reference {i+1}"
        assembler.workflow[vae_encode_id] = vae_encode_node
        
        latent_conn = [vae_encode_id, 0]

        pos_ref_latent_id = assembler._get_unique_id()
        pos_ref_latent_node = assembler._get_node_template_from_api("ReferenceLatent")
        pos_ref_latent_node['inputs']['conditioning'] = current_pos_conditioning
        pos_ref_latent_node['inputs']['latent'] = latent_conn
        pos_ref_latent_node['_meta']['title'] = f"Positive ReferenceLatent {i+1}"
        assembler.workflow[pos_ref_latent_id] = pos_ref_latent_node
        current_pos_conditioning = [pos_ref_latent_id, 0]

        if neg_target_node_id:
            neg_ref_latent_id = assembler._get_unique_id()
            neg_ref_latent_node = assembler._get_node_template_from_api("ReferenceLatent")
            neg_ref_latent_node['inputs']['conditioning'] = current_neg_conditioning
            neg_ref_latent_node['inputs']['latent'] = latent_conn
            neg_ref_latent_node['_meta']['title'] = f"Negative ReferenceLatent {i+1}"
            assembler.workflow[neg_ref_latent_id] = neg_ref_latent_node
            current_neg_conditioning = [neg_ref_latent_id, 0]

    assembler.workflow[pos_target_node_id]['inputs'][pos_target_input_name] = current_pos_conditioning
    if neg_target_node_id:
        assembler.workflow[neg_target_node_id]['inputs'][neg_target_input_name] = current_neg_conditioning
    
    print(f"ReferenceLatent injector applied. Re-routed inputs through {len(chain_items)} reference images.")