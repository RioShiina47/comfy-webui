def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    if ksampler_name not in assembler.node_map:
        print(f"Warning: Target node '{ksampler_name}' for Krea2 ControlNet chain not found. Skipping.")
        return
        
    ksampler_id = assembler.node_map[ksampler_name]

    if 'model' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'model' input. Skipping.")
        return

    vae_source_str = chain_definition.get('vae_source')
    vae_connection = None
    if vae_source_str:
        vae_node_name, vae_idx_str = vae_source_str.split(':')
        if vae_node_name in assembler.node_map:
            vae_connection = [assembler.node_map[vae_node_name], int(vae_idx_str)]
            
    latent_connection = assembler.workflow[ksampler_id]['inputs'].get('latent_image')
    if not latent_connection:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'latent_image' input. Krea2 ControlNet requires it. Skipping.")
        return

    current_model_connection = assembler.workflow[ksampler_id]['inputs']['model']
    
    for item_data in chain_items:
        image_loader_id = assembler._get_unique_id()
        image_loader_node = assembler._get_node_template_from_api("LoadImage")
        image_loader_node['inputs']['image'] = item_data['image']
        assembler.workflow[image_loader_id] = image_loader_node

        image_scaler_id = assembler._get_unique_id()
        image_scaler_node = assembler._get_node_template_from_api("ImageScaleToTotalPixels")
        image_scaler_node['inputs']['image'] = [image_loader_id, 0]
        image_scaler_node['inputs']['upscale_method'] = 'nearest-exact'
        image_scaler_node['inputs']['megapixels'] = 1.0
        image_scaler_node['inputs']['resolution_steps'] = 1
        assembler.workflow[image_scaler_id] = image_scaler_node

        lora_loader_id = assembler._get_unique_id()
        lora_loader_node = assembler._get_node_template_from_api("Krea2ControlLoRALoader")
        lora_loader_node['inputs']['lora_name'] = item_data['control_net_name']
        lora_loader_node['inputs']['strength'] = item_data.get('strength', 1.0)
        lora_loader_node['inputs']['model'] = current_model_connection
        assembler.workflow[lora_loader_id] = lora_loader_node

        img_encode_id = assembler._get_unique_id()
        img_encode_node = assembler._get_node_template_from_api("Krea2ControlImageEncode")
        img_encode_node['inputs']['resize'] = "match_latent_size"
        img_encode_node['inputs']['upscale_method'] = "lanczos"
        img_encode_node['inputs']['crop'] = "center"
        img_encode_node['inputs']['channel_mode'] = "rgb"
        img_encode_node['inputs']['normalize'] = "none"
        img_encode_node['inputs']['invert'] = False
        img_encode_node['inputs']['batch_mode'] = "independent_images"
        img_encode_node['inputs']['control_image'] = [image_scaler_id, 0]
        if vae_connection:
            img_encode_node['inputs']['vae'] = vae_connection
        if latent_connection:
            img_encode_node['inputs']['latent'] = latent_connection
        assembler.workflow[img_encode_id] = img_encode_node

        apply_cn_id = assembler._get_unique_id()
        apply_cn_node = assembler._get_node_template_from_api("Krea2ControlApply")
        apply_cn_node['inputs']['model'] = [lora_loader_id, 0]
        apply_cn_node['inputs']['control_latent'] = [img_encode_id, 0]
        
        assembler.workflow[apply_cn_id] = apply_cn_node

        current_model_connection = [apply_cn_id, 0]

    assembler.workflow[ksampler_id]['inputs']['model'] = current_model_connection
    
    print(f"Krea2 ControlNet injector applied. KSampler model input redirected through {len(chain_items)} Krea2 ControlNet nodes.")
