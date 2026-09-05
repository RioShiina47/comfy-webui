def inject(assembler, chain_definition, chain_items):
    """
    Dynamically injects uploaded reference audios into MiniMaxH3ReferenceToVideo node.
    chain_items is a list of saved audio filenames.
    """
    if not chain_items or not isinstance(chain_items, list):
        return

    target_node_name = chain_definition.get('target_node', 'minimax_h3')
    target_node_id = assembler.node_map.get(target_node_name)
    if not target_node_id:
        print(f"[H3 RefAudio Injector] Error: Target node '{target_node_name}' not found in node_map.")
        return

    minimax_h3_node = assembler.workflow[target_node_id]

    valid_audios = [aud for aud in chain_items if aud]
    for idx, audio_file in enumerate(valid_audios):
        load_id = assembler._get_unique_id()
        load_node = assembler._get_node_template_from_api("LoadAudio")
        load_node['inputs']['audio'] = audio_file
        assembler.workflow[load_id] = load_node

        param_name = f"ref_audios.ref_audio_{idx}"
        minimax_h3_node['inputs'][param_name] = [load_id, 0]

    print(f"[H3 RefAudio Injector] Successfully injected {len(valid_audios)} reference audio(s).")
