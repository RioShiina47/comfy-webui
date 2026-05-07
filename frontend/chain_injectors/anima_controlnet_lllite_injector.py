def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    if ksampler_name not in assembler.node_map:
        print(f"Warning: KSampler node '{ksampler_name}' not found for Anima LLLite chain. Skipping.")
        return
        
    ksampler_id = assembler.node_map[ksampler_name]

    if 'model' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'model' input. Skipping.")
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

        apply_cn_id = assembler._get_unique_id()
        apply_cn_node = assembler._get_node_template_from_api("AnimaLLLiteApply")
        
        apply_cn_node['inputs']['lllite_name'] = item_data['control_net_name']
        apply_cn_node['inputs']['strength'] = item_data['strength']
        apply_cn_node['inputs']['start_percent'] = item_data['start_percent']
        apply_cn_node['inputs']['end_percent'] = item_data['end_percent']

        apply_cn_node['inputs']['model'] = current_model_connection
        apply_cn_node['inputs']['image'] = [image_scaler_id, 0]
        
        assembler.workflow[apply_cn_id] = apply_cn_node

        current_model_connection = [apply_cn_id, 0]

    assembler.workflow[ksampler_id]['inputs']['model'] = current_model_connection
    
    print(f"Anima LLLite injector applied. KSampler model input re-routed through {len(chain_items)} LLLite(s).")