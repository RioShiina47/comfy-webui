def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    if ksampler_name not in assembler.node_map:
        print(f"Warning: KSampler node '{ksampler_name}' not found for ControlNet chain. Skipping.")
        return
        
    ksampler_id = assembler.node_map[ksampler_name]

    if 'positive' not in assembler.workflow[ksampler_id]['inputs'] or \
       'negative' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'positive' or 'negative' inputs. Skipping ControlNet chain.")
        return
        
    current_positive_connection = assembler.workflow[ksampler_id]['inputs']['positive']
    current_negative_connection = assembler.workflow[ksampler_id]['inputs']['negative']
    
    vae_source_str = chain_definition.get('vae_source')
    vae_connection = None
    if vae_source_str:
        try:
            vae_node_name, vae_idx_str = vae_source_str.split(':')
            if vae_node_name in assembler.node_map:
                vae_connection = [assembler.node_map[vae_node_name], int(vae_idx_str)]
            else:
                print(f"Warning: VAE source node '{vae_node_name}' not found for ControlNet chain. VAE will not be connected.")
        except ValueError:
            print(f"Warning: Invalid 'vae_source' format '{vae_source_str}' for ControlNet chain. Expected 'node_name:index'. VAE will not be connected.")
    else:
        print(f"Warning: 'vae_source' not defined for ControlNet chain definition. VAE may not be connected if required by the ControlNet node.")

    for item_data in chain_items:
        cn_loader_id = assembler._get_unique_id()
        cn_loader_node = assembler._get_node_template_from_api("ControlNetLoader")
        cn_loader_node['inputs']['control_net_name'] = item_data['control_net_name']
        assembler.workflow[cn_loader_id] = cn_loader_node

        image_loader_id = assembler._get_unique_id()
        image_loader_node = assembler._get_node_template_from_api("LoadImage")
        image_loader_node['inputs']['image'] = item_data['image']
        assembler.workflow[image_loader_id] = image_loader_node

        apply_cn_id = assembler._get_unique_id()
        apply_cn_node = assembler._get_node_template_from_api(chain_definition['template'])
        
        apply_cn_node['inputs']['strength'] = item_data['strength']
        apply_cn_node['inputs']['start_percent'] = item_data['start_percent']
        apply_cn_node['inputs']['end_percent'] = item_data['end_percent']

        apply_cn_node['inputs']['positive'] = current_positive_connection
        apply_cn_node['inputs']['negative'] = current_negative_connection
        apply_cn_node['inputs']['control_net'] = [cn_loader_id, 0]
        apply_cn_node['inputs']['image'] = [image_loader_id, 0]
        
        if 'vae' in apply_cn_node['inputs'] and vae_connection:
            apply_cn_node['inputs']['vae'] = vae_connection
        
        assembler.workflow[apply_cn_id] = apply_cn_node

        current_positive_connection = [apply_cn_id, 0]
        current_negative_connection = [apply_cn_id, 1]

    assembler.workflow[ksampler_id]['inputs']['positive'] = current_positive_connection
    assembler.workflow[ksampler_id]['inputs']['negative'] = current_negative_connection
    
    print(f"ControlNet injector applied. KSampler inputs re-routed through {len(chain_items)} ControlNet(s).")