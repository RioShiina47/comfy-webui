def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    guider_node_name = chain_definition.get('guider_node')
    guider_target_inputs = chain_definition.get('guider_target_inputs', [])

    if guider_node_name and guider_node_name in assembler.node_map and guider_target_inputs:
        guider_id = assembler.node_map[guider_node_name]
        print(f"Style injector targeting DualCFGGuider node '{guider_node_name}'.")

        style_model_loader_id = assembler._get_unique_id()
        style_model_loader_node = assembler._get_node_template_from_api("StyleModelLoader")
        style_model_loader_node['inputs']['style_model_name'] = "flux1-redux-dev.safetensors"
        assembler.workflow[style_model_loader_id] = style_model_loader_node

        clip_vision_loader_id = assembler._get_unique_id()
        clip_vision_loader_node = assembler._get_node_template_from_api("CLIPVisionLoader")
        clip_vision_loader_node['inputs']['clip_name'] = "sigclip_vision_patch14_384.safetensors"
        assembler.workflow[clip_vision_loader_id] = clip_vision_loader_node

        for target_input_name in guider_target_inputs:
            if target_input_name in assembler.workflow[guider_id]['inputs']:
                original_connection = assembler.workflow[guider_id]['inputs'][target_input_name]
                current_conditioning = original_connection

                for item_data in chain_items:
                    image = item_data.get('image')
                    strength = item_data.get('strength', 1.0)
                    if not image or strength is None:
                        continue

                    load_image_id = assembler._get_unique_id()
                    clip_vision_encode_id = assembler._get_unique_id()
                    style_apply_id = assembler._get_unique_id()

                    load_image_node = assembler._get_node_template_from_api("LoadImage")
                    clip_vision_encode_node = assembler._get_node_template_from_api("CLIPVisionEncode")
                    style_apply_node = assembler._get_node_template_from_api("StyleModelApply")
                    
                    load_image_node['inputs']['image'] = image
                    
                    clip_vision_encode_node['inputs']['crop'] = "center"
                    clip_vision_encode_node['inputs']['clip_vision'] = [clip_vision_loader_id, 0]
                    clip_vision_encode_node['inputs']['image'] = [load_image_id, 0]

                    style_apply_node['inputs']['strength'] = strength
                    style_apply_node['inputs']['strength_type'] = "multiply"
                    style_apply_node['inputs']['conditioning'] = current_conditioning
                    style_apply_node['inputs']['style_model'] = [style_model_loader_id, 0]
                    style_apply_node['inputs']['clip_vision_output'] = [clip_vision_encode_id, 0]
                    
                    assembler.workflow[load_image_id] = load_image_node
                    assembler.workflow[clip_vision_encode_id] = clip_vision_encode_node
                    assembler.workflow[style_apply_id] = style_apply_node

                    current_conditioning = [style_apply_id, 0]

                assembler.workflow[guider_id]['inputs'][target_input_name] = current_conditioning
                print(f"  - Input '{target_input_name}' of node '{guider_node_name}' re-routed through {len(chain_items)} style images.")
        
        return

    flux_guidance_name = chain_definition.get('flux_guidance_node')
    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')

    target_node_id = None
    target_input_name = None
    
    if flux_guidance_name and flux_guidance_name in assembler.node_map:
        flux_guidance_id = assembler.node_map[flux_guidance_name]
        if 'conditioning' in assembler.workflow[flux_guidance_id]['inputs']:
            target_node_id = flux_guidance_id
            target_input_name = 'conditioning'
            print(f"Style injector targeting FluxGuidance node '{flux_guidance_name}'.")

    if not target_node_id:
        if ksampler_name in assembler.node_map:
            ksampler_id = assembler.node_map[ksampler_name]
            if 'positive' in assembler.workflow[ksampler_id]['inputs']:
                target_node_id = ksampler_id
                target_input_name = 'positive'
                print(f"Style injector targeting KSampler node '{ksampler_name}'.")
        else:
            print(f"Warning: KSampler node '{ksampler_name}' not found for Style chain. Skipping.")
            return
            
    if not target_node_id:
        print(f"Warning: Could not find a valid injection point for Style chain (checked '{flux_guidance_name}' and '{ksampler_name}'). Skipping.")
        return

    current_conditioning = assembler.workflow[target_node_id]['inputs'][target_input_name]

    style_model_loader_id = assembler._get_unique_id()
    style_model_loader_node = assembler._get_node_template_from_api("StyleModelLoader")
    style_model_loader_node['inputs']['style_model_name'] = "flux1-redux-dev.safetensors"
    assembler.workflow[style_model_loader_id] = style_model_loader_node

    clip_vision_loader_id = assembler._get_unique_id()
    clip_vision_loader_node = assembler._get_node_template_from_api("CLIPVisionLoader")
    clip_vision_loader_node['inputs']['clip_name'] = "sigclip_vision_patch14_384.safetensors"
    assembler.workflow[clip_vision_loader_id] = clip_vision_loader_node

    for item_data in chain_items:
        image = item_data.get('image')
        strength = item_data.get('strength', 1.0)
        if not image or strength is None:
            continue

        load_image_id = assembler._get_unique_id()
        clip_vision_encode_id = assembler._get_unique_id()
        style_apply_id = assembler._get_unique_id()

        load_image_node = assembler._get_node_template_from_api("LoadImage")
        clip_vision_encode_node = assembler._get_node_template_from_api("CLIPVisionEncode")
        style_apply_node = assembler._get_node_template_from_api("StyleModelApply")
        
        load_image_node['inputs']['image'] = image
        
        clip_vision_encode_node['inputs']['crop'] = "center"
        clip_vision_encode_node['inputs']['clip_vision'] = [clip_vision_loader_id, 0]
        clip_vision_encode_node['inputs']['image'] = [load_image_id, 0]

        style_apply_node['inputs']['strength'] = strength
        style_apply_node['inputs']['strength_type'] = "multiply"
        style_apply_node['inputs']['conditioning'] = current_conditioning
        style_apply_node['inputs']['style_model'] = [style_model_loader_id, 0]
        style_apply_node['inputs']['clip_vision_output'] = [clip_vision_encode_id, 0]
        
        assembler.workflow[load_image_id] = load_image_node
        assembler.workflow[clip_vision_encode_id] = clip_vision_encode_node
        assembler.workflow[style_apply_id] = style_apply_node

        current_conditioning = [style_apply_id, 0]

    assembler.workflow[target_node_id]['inputs'][target_input_name] = current_conditioning
    print(f"Style injector successfully applied. Input '{target_input_name}' of node '{target_node_id}' re-routed through {len(chain_items)} style images.")