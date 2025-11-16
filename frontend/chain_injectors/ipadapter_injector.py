def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    final_settings = {}
    if chain_items and isinstance(chain_items[-1], dict) and chain_items[-1].get('is_final_settings'):
        final_settings = chain_items.pop()

    end_node_name = chain_definition.get('end')
    if not end_node_name or end_node_name not in assembler.node_map:
        print(f"Warning: End node '{end_node_name}' for dynamic IPAdapter chain not found. Skipping chain.")
        return
        
    end_node_id = assembler.node_map[end_node_name]
    
    if 'model' not in assembler.workflow[end_node_id]['inputs']:
        print(f"Warning: 'model' input not found in end node '{end_node_name}' for IPAdapter chain. Skipping.")
        return
    
    current_model_connection = assembler.workflow[end_node_id]['inputs']['model']

    model_type = final_settings.get('model_type', 'sdxl')
    megapixels = 1.05 if model_type == 'sdxl' else 0.39

    pos_embed_outputs = []
    neg_embed_outputs = []

    for i, item_data in enumerate(chain_items):
        loader_type = item_data.get('loader_type', 'Unified')
        
        loader_template_name = "IPAdapterUnifiedLoader"
        if loader_type == 'FaceID':
            loader_template_name = "IPAdapterUnifiedLoaderFaceID"

        image_loader_id = assembler._get_unique_id()
        image_loader_node = assembler._get_node_template_from_api("LoadImage")
        image_loader_node['inputs']['image'] = item_data['image']
        assembler.workflow[image_loader_id] = image_loader_node
        
        image_scaler_id = assembler._get_unique_id()
        image_scaler_node = assembler._get_node_template_from_api("ImageScaleToTotalPixels")
        image_scaler_node['inputs']['image'] = [image_loader_id, 0]
        image_scaler_node['inputs']['upscale_method'] = 'bicubic'
        image_scaler_node['inputs']['megapixels'] = megapixels
        assembler.workflow[image_scaler_id] = image_scaler_node

        ipadapter_loader_id = assembler._get_unique_id()
        ipadapter_loader_node = assembler._get_node_template_from_api(loader_template_name)
        ipadapter_loader_node['inputs']['preset'] = item_data['preset']
        ipadapter_loader_node['inputs']['model'] = current_model_connection
        if loader_type == 'FaceID':
            ipadapter_loader_node['inputs']['lora_strength'] = item_data.get('lora_strength', 0.6)
            ipadapter_loader_node['inputs']['provider'] = item_data.get('provider', 'CPU')
        assembler.workflow[ipadapter_loader_id] = ipadapter_loader_node
        
        encoder_id = assembler._get_unique_id()
        encoder_node = assembler._get_node_template_from_api("IPAdapterEncoder")
        encoder_node['inputs']['weight'] = item_data['weight']
        encoder_node['inputs']['ipadapter'] = [ipadapter_loader_id, 1]
        encoder_node['inputs']['image'] = [image_scaler_id, 0]
        assembler.workflow[encoder_id] = encoder_node
        
        pos_embed_outputs.append([encoder_id, 0])
        neg_embed_outputs.append([encoder_id, 1])

    pos_combiner_id = assembler._get_unique_id()
    pos_combiner_node = assembler._get_node_template_from_api("IPAdapterCombineEmbeds")
    pos_combiner_node['inputs']['method'] = final_settings.get('final_combine_method', 'concat')
    for i, conn in enumerate(pos_embed_outputs):
        pos_combiner_node['inputs'][f'embed{i+1}'] = conn
    assembler.workflow[pos_combiner_id] = pos_combiner_node

    neg_combiner_id = assembler._get_unique_id()
    neg_combiner_node = assembler._get_node_template_from_api("IPAdapterCombineEmbeds")
    neg_combiner_node['inputs']['method'] = final_settings.get('final_combine_method', 'concat')
    for i, conn in enumerate(neg_embed_outputs):
        neg_combiner_node['inputs'][f'embed{i+1}'] = conn
    assembler.workflow[neg_combiner_id] = neg_combiner_node

    final_loader_type = final_settings.get('final_loader_type', 'Unified')
    final_loader_template_name = "IPAdapterUnifiedLoader"
    if final_loader_type == 'FaceID':
        final_loader_template_name = "IPAdapterUnifiedLoaderFaceID"

    final_loader_id = assembler._get_unique_id()
    final_loader_node = assembler._get_node_template_from_api(final_loader_template_name)
    final_loader_node['inputs']['preset'] = final_settings.get('final_preset', 'STANDARD (medium strength)')
    final_loader_node['inputs']['model'] = current_model_connection
    if final_loader_type == 'FaceID':
        final_loader_node['inputs']['lora_strength'] = final_settings.get('final_lora_strength', 0.6)
        final_loader_node['inputs']['provider'] = final_settings.get('final_provider', 'CPU')
    assembler.workflow[final_loader_id] = final_loader_node
    
    final_embeds_applier_id = assembler._get_unique_id()
    final_embeds_applier_node = assembler._get_node_template_from_api("IPAdapterEmbeds")
    final_embeds_applier_node['inputs']['weight'] = final_settings.get('final_weight', 1.0)
    final_embeds_applier_node['inputs']['weight_type'] = final_settings.get('final_weight_type', 'linear')
    final_embeds_applier_node['inputs']['embeds_scaling'] = final_settings.get('final_embeds_scaling', 'V only')
    final_embeds_applier_node['inputs']['model'] = [final_loader_id, 0]
    final_embeds_applier_node['inputs']['ipadapter'] = [final_loader_id, 1]
    final_embeds_applier_node['inputs']['pos_embed'] = [pos_combiner_id, 0]
    final_embeds_applier_node['inputs']['neg_embed'] = [neg_combiner_id, 0]
    assembler.workflow[final_embeds_applier_id] = final_embeds_applier_node

    assembler.workflow[end_node_id]['inputs']['model'] = [final_embeds_applier_id, 0]
    print(f"IPAdapter chain injected. KSampler ('{end_node_name}') model input re-routed.")