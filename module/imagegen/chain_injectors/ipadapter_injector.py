def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    final_settings = {}
    if chain_items and isinstance(chain_items[-1], dict) and chain_items[-1].get('is_final_settings'):
        final_settings = chain_items.pop()

    if not chain_items:
        return

    end_node_name = chain_definition.get('end')
    if not end_node_name or end_node_name not in assembler.node_map:
        print(f"Warning: Target node '{end_node_name}' for IPAdapter chain not found. Skipping chain injection.")
        return
        
    end_node_id = assembler.node_map[end_node_name]
    
    if 'model' not in assembler.workflow[end_node_id]['inputs']:
        print(f"Warning: Target node '{end_node_name}' is missing 'model' input. Skipping IPAdapter chain.")
        return
    
    current_model_connection = assembler.workflow[end_node_id]['inputs']['model']
    
    model_type = final_settings.get('model_type', 'sdxl')
    megapixels = 1.05 if model_type == 'sdxl' else 0.39

    first_preset = chain_items[0].get('preset', '')
    is_faceid_chain = 'FACEID' in first_preset.upper()

    if is_faceid_chain:
        for i, item_data in enumerate(chain_items):
            image_loader_id = assembler._get_unique_id()
            image_loader_node = assembler._get_node_template_from_api("LoadImage")
            image_loader_node['inputs']['image'] = item_data['image']
            assembler.workflow[image_loader_id] = image_loader_node
            
            image_scaler_id = assembler._get_unique_id()
            image_scaler_node = assembler._get_node_template_from_api("ImageScaleToTotalPixels")
            image_scaler_node['inputs']['image'] = [image_loader_id, 0]
            image_scaler_node['inputs']['megapixels'] = megapixels
            image_scaler_node['inputs']['upscale_method'] = "lanczos"
            assembler.workflow[image_scaler_id] = image_scaler_node

            ipadapter_loader_id = assembler._get_unique_id()
            ipadapter_loader_node = assembler._get_node_template_from_api("IPAdapterUnifiedLoaderFaceID")
            ipadapter_loader_node['inputs']['model'] = current_model_connection
            ipadapter_loader_node['inputs']['preset'] = item_data['preset']
            ipadapter_loader_node['inputs']['lora_strength'] = item_data.get('lora_strength', 0.6)
            ipadapter_loader_node['inputs']['provider'] = "CUDA"
            assembler.workflow[ipadapter_loader_id] = ipadapter_loader_node

            apply_id = assembler._get_unique_id()
            apply_node = assembler._get_node_template_from_api("IPAdapterFaceID")
            apply_node['inputs']['model'] = [ipadapter_loader_id, 0]
            apply_node['inputs']['ipadapter'] = [ipadapter_loader_id, 1]
            apply_node['inputs']['image'] = [image_scaler_id, 0]
            apply_node['inputs']['weight'] = item_data['weight']
            apply_node['inputs']['weight_faceidv2'] = final_settings.get('final_lora_strength', 0.6)
            apply_node['inputs']['weight_type'] = "linear"
            apply_node['inputs']['combine_embeds'] = final_settings.get('final_combine_method', 'concat')
            apply_node['inputs']['start_at'] = item_data.get('start_percent', 0.0)
            apply_node['inputs']['end_at'] = item_data.get('end_percent', 1.0)
            apply_node['inputs']['embeds_scaling'] = final_settings.get('final_embeds_scaling', 'V only')
            
            assembler.workflow[apply_id] = apply_node
            current_model_connection = [apply_id, 0]

        assembler.workflow[end_node_id]['inputs']['model'] = current_model_connection
        print(f"IPAdapter FaceID injector applied (Direct Apply). Redirected '{end_node_name}' model input through {len(chain_items)} FaceID node(s).")
        return

    else:
        pos_embed_outputs = []
        neg_embed_outputs = []

        for i, item_data in enumerate(chain_items):
            loader_type = 'FaceID' if 'FACEID' in item_data.get('preset', '') else 'Unified'
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
            image_scaler_node['inputs']['megapixels'] = megapixels
            image_scaler_node['inputs']['upscale_method'] = "lanczos"
            assembler.workflow[image_scaler_id] = image_scaler_node

            ipadapter_loader_id = assembler._get_unique_id()
            ipadapter_loader_node = assembler._get_node_template_from_api(loader_template_name)
            ipadapter_loader_node['inputs']['model'] = current_model_connection
            ipadapter_loader_node['inputs']['preset'] = item_data['preset']
            if loader_type == 'FaceID':
                 ipadapter_loader_node['inputs']['lora_strength'] = item_data.get('lora_strength', 0.6)
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
        
        final_loader_type = 'FaceID' if 'FACEID' in final_settings.get('final_preset', '') else 'Unified'
        final_loader_template_name = "IPAdapterUnifiedLoader"
        if final_loader_type == 'FaceID':
            final_loader_template_name = "IPAdapterUnifiedLoaderFaceID"

        final_loader_id = assembler._get_unique_id()
        final_loader_node = assembler._get_node_template_from_api(final_loader_template_name)
        final_loader_node['inputs']['model'] = current_model_connection
        final_loader_node['inputs']['preset'] = final_settings.get('final_preset', 'STANDARD (medium strength)')
        if final_loader_type == 'FaceID':
            final_loader_node['inputs']['lora_strength'] = final_settings.get('final_lora_strength', 0.6)
        assembler.workflow[final_loader_id] = final_loader_node

        apply_embeds_id = assembler._get_unique_id()
        apply_embeds_node = assembler._get_node_template_from_api("IPAdapterEmbeds")
        apply_embeds_node['inputs']['weight'] = final_settings.get('final_weight', 1.0)
        apply_embeds_node['inputs']['weight_type'] = final_settings.get('final_weight_type', 'linear')
        apply_embeds_node['inputs']['embeds_scaling'] = final_settings.get('final_embeds_scaling', 'V only')
        apply_embeds_node['inputs']['model'] = [final_loader_id, 0]
        apply_embeds_node['inputs']['ipadapter'] = [final_loader_id, 1]
        apply_embeds_node['inputs']['pos_embed'] = [pos_combiner_id, 0]
        apply_embeds_node['inputs']['neg_embed'] = [neg_combiner_id, 0]
        assembler.workflow[apply_embeds_id] = apply_embeds_node

        assembler.workflow[end_node_id]['inputs']['model'] = [apply_embeds_id, 0]
        print(f"IPAdapter Unified injector applied. Redirected '{end_node_name}' model input through {len(chain_items)} reference image(s).")