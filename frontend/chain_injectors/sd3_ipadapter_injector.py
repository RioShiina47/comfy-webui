def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    if ksampler_name not in assembler.node_map:
        print(f"Warning: KSampler node '{ksampler_name}' not found for SD3 IPAdapter chain. Skipping.")
        return
        
    ksampler_id = assembler.node_map[ksampler_name]

    if 'model' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'model' input. Skipping SD3 IPAdapter chain.")
        return
        
    current_model_connection = assembler.workflow[ksampler_id]['inputs']['model']
    
    clip_vision_loader_id = assembler._get_unique_id()
    clip_vision_loader_node = assembler._get_node_template_from_api("CLIPVisionLoader")
    clip_vision_loader_node['inputs']['clip_name'] = "sigclip_vision_patch14_384.safetensors"
    assembler.workflow[clip_vision_loader_id] = clip_vision_loader_node

    ipadapter_loader_id = assembler._get_unique_id()
    ipadapter_loader_node = assembler._get_node_template_from_api("IPAdapterSD3Loader")
    ipadapter_loader_node['inputs']['ipadapter'] = "ip-adapter_sd35l_instantx.bin"
    ipadapter_loader_node['inputs']['provider'] = "cuda"
    assembler.workflow[ipadapter_loader_id] = ipadapter_loader_node

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
        assembler.workflow[image_scaler_id] = image_scaler_node

        clip_vision_encode_id = assembler._get_unique_id()
        clip_vision_encode_node = assembler._get_node_template_from_api("CLIPVisionEncode")
        clip_vision_encode_node['inputs']['crop'] = "center"
        clip_vision_encode_node['inputs']['clip_vision'] = [clip_vision_loader_id, 0]
        clip_vision_encode_node['inputs']['image'] = [image_scaler_id, 0]
        assembler.workflow[clip_vision_encode_id] = clip_vision_encode_node
        
        apply_ipa_id = assembler._get_unique_id()
        apply_ipa_node = assembler._get_node_template_from_api("ApplyIPAdapterSD3")
        
        apply_ipa_node['inputs']['weight'] = item_data['weight']
        apply_ipa_node['inputs']['start_percent'] = item_data['start_percent']
        apply_ipa_node['inputs']['end_percent'] = item_data['end_percent']
        
        apply_ipa_node['inputs']['model'] = current_model_connection
        apply_ipa_node['inputs']['ipadapter'] = [ipadapter_loader_id, 0]
        apply_ipa_node['inputs']['image_embed'] = [clip_vision_encode_id, 0]
        
        assembler.workflow[apply_ipa_id] = apply_ipa_node
        
        current_model_connection = [apply_ipa_id, 0]

    assembler.workflow[ksampler_id]['inputs']['model'] = current_model_connection
    
    print(f"SD3 IPAdapter injector applied. KSampler model input re-routed through {len(chain_items)} IPAdapter(s).")