def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    vae_name = chain_items[0] if isinstance(chain_items, list) else chain_items
    if not vae_name or vae_name == "None":
        return

    targets = chain_definition.get('targets', [])
    if not targets:
        return

    vae_loader_id = assembler._get_unique_id()
    vae_loader_node = assembler._get_node_template_from_api("VAELoader")
    vae_loader_node['inputs']['vae_name'] = vae_name
    assembler.workflow[vae_loader_id] = vae_loader_node

    injected_count = 0
    for target_str in targets:
        try:
            node_name, input_name = target_str.split(':')
            if node_name in assembler.node_map:
                node_id = assembler.node_map[node_name]
                assembler.workflow[node_id]['inputs'][input_name] = [vae_loader_id, 0]
                injected_count += 1
        except ValueError:
            print(f"Warning: Invalid VAE injector target format '{target_str}'. Expected 'node_name:input_name'.")
    
    if injected_count > 0:
        print(f"VAE injector applied. Rerouted {injected_count} connection(s) to new VAELoader ({vae_name}).")