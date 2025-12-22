from copy import deepcopy

def inject(assembler, chain_definition, chain_items):
    """
    Injects a chain of track generation and concatenation nodes.
    """
    if not chain_items:
        return

    start_node_name = chain_definition.get('start_node')
    end_node_name = chain_definition.get('end_node')

    if not start_node_name or start_node_name not in assembler.node_map:
        print(f"Warning: [WanMove Injector] Start node '{start_node_name}' not found. Skipping chain.")
        return
    if not end_node_name or end_node_name not in assembler.node_map:
        print(f"Warning: [WanMove Injector] End node '{end_node_name}' not found. Skipping chain.")
        return
        
    start_node_id = assembler.node_map[start_node_name]
    end_node_id = assembler.node_map[end_node_name]
    
    current_track_connection = [start_node_id, 0]
    
    base_track_gen_node = assembler.workflow[start_node_id]

    for item_data in chain_items:
        new_track_gen_id = assembler._get_unique_id()
        new_track_gen_node = deepcopy(base_track_gen_node)
        
        for key, value in item_data.items():
            if key in new_track_gen_node['inputs']:
                new_track_gen_node['inputs'][key] = value
        
        assembler.workflow[new_track_gen_id] = new_track_gen_node
        
        concat_id = assembler._get_unique_id()
        concat_node = assembler._get_node_template_from_api("WanMoveConcatTrack")
        concat_node['inputs']['tracks_1'] = current_track_connection
        concat_node['inputs']['tracks_2'] = [new_track_gen_id, 0]
        assembler.workflow[concat_id] = concat_node
        
        current_track_connection = [concat_id, 0]

    assembler.workflow[end_node_id]['inputs']['tracks'] = current_track_connection
    print(f"[WanMove Injector] Injected and concatenated {len(chain_items)} additional track segments.")