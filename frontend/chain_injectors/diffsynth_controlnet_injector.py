def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    model_sampler_name = chain_definition.get('model_sampler_node')
    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    
    target_node_id = None
    target_input_name = 'model'
    
    if model_sampler_name and model_sampler_name in assembler.node_map:
        model_sampler_id = assembler.node_map[model_sampler_name]
        if target_input_name in assembler.workflow[model_sampler_id]['inputs']:
            target_node_id = model_sampler_id
            print(f"ControlNet Model Patch injector targeting ModelSamplingAuraFlow node '{model_sampler_name}'.")

    if not target_node_id:
        if ksampler_name in assembler.node_map:
            ksampler_id = assembler.node_map[ksampler_name]
            if target_input_name in assembler.workflow[ksampler_id]['inputs']:
                target_node_id = ksampler_id
                print(f"ControlNet Model Patch injector targeting KSampler node '{ksampler_name}'.")
        else:
            print(f"Warning: Neither ModelSamplingAuraFlow node '{model_sampler_name}' nor KSampler node '{ksampler_name}' found for ControlNet patch chain. Skipping.")
            return
            
    if not target_node_id:
        print(f"Warning: Could not find a valid 'model' input on target nodes. Skipping ControlNet patch chain.")
        return

    current_model_connection = assembler.workflow[target_node_id]['inputs'][target_input_name]
    
    vae_source_str = chain_definition.get('vae_source')
    vae_connection = None
    if vae_source_str:
        try:
            vae_node_name, vae_idx_str = vae_source_str.split(':')
            if vae_node_name in assembler.node_map:
                vae_connection = [assembler.node_map[vae_node_name], int(vae_idx_str)]
            else:
                print(f"Warning: VAE source node '{vae_node_name}' not found for ControlNet patch chain. VAE will not be connected.")
        except ValueError:
            print(f"Warning: Invalid 'vae_source' format '{vae_source_str}' for ControlNet patch chain. Expected 'node_name:index'. VAE will not be connected.")
    else:
        print(f"Warning: 'vae_source' not defined for ControlNet patch chain definition. VAE may not be connected.")

    for item_data in chain_items:
        patch_loader_id = assembler._get_unique_id()
        patch_loader_node = assembler._get_node_template_from_api("ModelPatchLoader")
        patch_loader_node['inputs']['name'] = item_data['control_net_name']
        assembler.workflow[patch_loader_id] = patch_loader_node

        image_loader_id = assembler._get_unique_id()
        image_loader_node = assembler._get_node_template_from_api("LoadImage")
        image_loader_node['inputs']['image'] = item_data['image']
        assembler.workflow[image_loader_id] = image_loader_node

        apply_cn_id = assembler._get_unique_id()
        apply_cn_node = assembler._get_node_template_from_api(chain_definition['template'])
        
        apply_cn_node['inputs']['strength'] = item_data.get('strength', 1.0)
        apply_cn_node['inputs']['model'] = current_model_connection
        apply_cn_node['inputs']['model_patch'] = [patch_loader_id, 0]
        apply_cn_node['inputs']['image'] = [image_loader_id, 0]
        
        if 'vae' in apply_cn_node['inputs'] and vae_connection:
            apply_cn_node['inputs']['vae'] = vae_connection
        
        assembler.workflow[apply_cn_id] = apply_cn_node

        current_model_connection = [apply_cn_id, 0]

    assembler.workflow[target_node_id]['inputs'][target_input_name] = current_model_connection
    
    print(f"ControlNet Model Patch injector applied. Target 'model' input re-routed through {len(chain_items)} patch(es).")