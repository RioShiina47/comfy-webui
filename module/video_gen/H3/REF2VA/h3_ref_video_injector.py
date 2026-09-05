def inject(assembler, chain_definition, chain_items):
    """
    Dynamically injects uploaded reference videos into MiniMaxH3ReferenceToVideo node.
    chain_items is a list of saved video filenames.
    """
    if not chain_items or not isinstance(chain_items, list):
        return

    target_node_name = chain_definition.get('target_node', 'minimax_h3')
    target_node_id = assembler.node_map.get(target_node_name)
    if not target_node_id:
        print(f"[H3 RefVideo Injector] Error: Target node '{target_node_name}' not found in node_map.")
        return

    minimax_h3_node = assembler.workflow[target_node_id]

    valid_videos = [vid for vid in chain_items if vid]
    for idx, video_file in enumerate(valid_videos):
        load_id = assembler._get_unique_id()
        load_node = assembler._get_node_template_from_api("LoadVideo")
        load_node['inputs']['file'] = video_file
        assembler.workflow[load_id] = load_node

        comp_id = assembler._get_unique_id()
        comp_node = assembler._get_node_template_from_api("GetVideoComponents")
        comp_node['inputs']['video'] = [load_id, 0]
        assembler.workflow[comp_id] = comp_node

        minimax_h3_node['inputs'][f"ref_videos.ref_video_{idx}"] = [comp_id, 0]
        minimax_h3_node['inputs'][f"ref_video_audios.ref_video_audio_{idx}"] = [comp_id, 1]

    print(f"[H3 RefVideo Injector] Successfully injected {len(valid_videos)} reference video(s).")
