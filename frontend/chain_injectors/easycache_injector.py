def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    if ksampler_name not in assembler.node_map:
        print(f"Warning: [EasyCache Injector] KSampler node '{ksampler_name}' not found. Skipping.")
        return
        
    ksampler_id = assembler.node_map[ksampler_name]

    if 'model' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: [EasyCache Injector] KSampler node '{ksampler_name}' is missing 'model' input. Skipping.")
        return
        
    current_model_connection = assembler.workflow[ksampler_id]['inputs']['model']
    
    easycache_id = assembler._get_unique_id()
    easycache_node = assembler._get_node_template_from_api("EasyCache")
    
    easycache_node['inputs']['reuse_threshold'] = chain_definition.get('reuse_threshold', 0.2)
    easycache_node['inputs']['start_percent'] = chain_definition.get('start_percent', 0.15)
    easycache_node['inputs']['end_percent'] = chain_definition.get('end_percent', 0.95)
    easycache_node['inputs']['verbose'] = chain_definition.get('verbose', False)
    
    easycache_node['inputs']['model'] = current_model_connection
    assembler.workflow[easycache_id] = easycache_node
    
    assembler.workflow[ksampler_id]['inputs']['model'] = [easycache_id, 0]
    
    print(f"EasyCache injector applied. KSampler model input re-routed through EasyCache node.")