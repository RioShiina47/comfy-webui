FEATURE_NAME = 'h3_guide'
CHAIN_TYPE = 'dynamic_h3_guide_chains'

def inject(assembler, chain_definition, chain_items):
    """
    Dynamically injects MiniMaxH3AddGuide nodes into the conditioning pipeline.
    Connects:
      Initial:
        positive: minimax_h3:0
        latent: minimax_h3:1
        vae: video_vae_loader:0
        audio_vae: audio_vae_loader:0
      Each guide item:
        - Creates LoadImage (if item has image) or LoadVideo+GetVideoComponents (if video) or LoadAudio (if audio)
        - Creates MiniMaxH3AddGuide with:
            frame_idx: int(item['frame_idx'])
            positive: current_positive
            latent: latent_connection
            vae: vae_connection
            audio_vae: audio_vae_connection
            image: [load_img_id, 0] or [comp_id, 0] (if present)
            audio: [load_aud_id, 0] or [comp_id, 1] (if present)
        - Updates current_positive = [guide_id, 0]
      Finally:
        Connects final current_positive to target_nodes (e.g. basic_guider:conditioning)
    """
    if not chain_items or not isinstance(chain_items, list):
        return

    def get_template(class_type):
        if hasattr(assembler, '_get_node_template_from_api'):
            return assembler._get_node_template_from_api(class_type)
        elif hasattr(assembler, '_get_node_template'):
            return assembler._get_node_template(class_type)
        raise AttributeError("WorkflowAssembler does not provide a node template retriever.")

    source_node_name = chain_definition.get('source_node', 'minimax_h3')
    source_node_id = assembler.node_map.get(source_node_name)
    if not source_node_id:
        print(f"[H3 Guide Injector] Error: Source node '{source_node_name}' not found in node_map.")
        return

    # 1. Resolve initial positive and latent connections from source node
    current_positive = [source_node_id, int(chain_definition.get('source_output', 0))]
    latent_conn = [source_node_id, 1]

    # 2. Resolve VAE connections
    vae_source_str = chain_definition.get('vae_source', 'video_vae_loader:0')
    vae_conn = None
    if vae_source_str and ':' in vae_source_str:
        vae_node_name, vae_idx_str = vae_source_str.split(':')
        vae_node_id = assembler.node_map.get(vae_node_name)
        if vae_node_id:
            vae_conn = [vae_node_id, int(vae_idx_str)]
        else:
            print(f"[H3 Guide Injector] Warning: VAE source '{vae_node_name}' not found in node_map.")

    audio_vae_source_str = chain_definition.get('audio_vae_source', 'audio_vae_loader:0')
    audio_vae_conn = None
    if audio_vae_source_str and ':' in audio_vae_source_str:
        a_vae_name, a_vae_idx_str = audio_vae_source_str.split(':')
        a_vae_id = assembler.node_map.get(a_vae_name)
        if a_vae_id:
            audio_vae_conn = [a_vae_id, int(a_vae_idx_str)]
        else:
            print(f"[H3 Guide Injector] Warning: Audio VAE source '{a_vae_name}' not found in node_map.")

    # 3. Filter valid guide items and sort by frame_idx
    valid_items = []
    for item in chain_items:
        if not isinstance(item, dict):
            continue
        has_media = bool(item.get('image') or item.get('video') or item.get('audio'))
        if has_media and item.get('frame_idx') is not None:
            valid_items.append(item)

    if not valid_items:
        return

    valid_items.sort(key=lambda x: int(x['frame_idx']))

    target_nodes_def = chain_definition.get('target_nodes', ['basic_guider:conditioning'])
    guide_template_name = chain_definition.get('template', 'MiniMaxH3AddGuide')

    applied_count = 0
    for idx, item in enumerate(valid_items):
        frame_idx = int(item['frame_idx'])
        image_conn = None
        audio_conn = None

        # Handle Image
        if item.get('image'):
            load_img_id = assembler._get_unique_id()
            load_img_node = get_template("LoadImage")
            load_img_node['inputs']['image'] = item['image']
            load_img_node['_meta'] = {"title": f"Load Guide Image {idx+1}"}
            assembler.workflow[load_img_id] = load_img_node
            image_conn = [load_img_id, 0]

        # Handle Video
        if item.get('video'):
            load_vid_id = assembler._get_unique_id()
            load_vid_node = get_template("LoadVideo")
            load_vid_node['inputs']['file'] = item['video']
            if 'video-preview' in load_vid_node['inputs']:
                load_vid_node['inputs']['video-preview'] = ""
            load_vid_node['_meta'] = {"title": f"Load Guide Video {idx+1}"}
            assembler.workflow[load_vid_id] = load_vid_node

            comp_id = assembler._get_unique_id()
            comp_node = get_template("GetVideoComponents")
            comp_node['inputs']['video'] = [load_vid_id, 0]
            comp_node['_meta'] = {"title": f"Extract Guide Components {idx+1}"}
            assembler.workflow[comp_id] = comp_node

            image_conn = [comp_id, 0]
            if not item.get('audio'):
                audio_conn = [comp_id, 1]

        # Handle Audio
        if item.get('audio'):
            load_aud_id = assembler._get_unique_id()
            load_aud_node = get_template("LoadAudio")
            load_aud_node['inputs']['audio'] = item['audio']
            load_aud_node['_meta'] = {"title": f"Load Guide Audio {idx+1}"}
            assembler.workflow[load_aud_id] = load_aud_node
            audio_conn = [load_aud_id, 0]

        # Create MiniMaxH3AddGuide
        guide_id = assembler._get_unique_id()
        guide_node = get_template(guide_template_name)
        guide_node['inputs']['frame_idx'] = frame_idx
        guide_node['inputs']['positive'] = current_positive
        guide_node['inputs']['latent'] = latent_conn
        if vae_conn:
            guide_node['inputs']['vae'] = vae_conn
        if audio_vae_conn:
            guide_node['inputs']['audio_vae'] = audio_vae_conn
        if image_conn:
            guide_node['inputs']['image'] = image_conn
        if audio_conn:
            guide_node['inputs']['audio'] = audio_conn
        guide_node['_meta'] = {"title": f"Add Guide for MiniMax H3 (frame {frame_idx})"}
        assembler.workflow[guide_id] = guide_node

        # Update chain positive output
        current_positive = [guide_id, 0]
        applied_count += 1

    # 4. Reconnect target conditioning nodes to the end of the guide chain
    if applied_count > 0:
        for target_str in target_nodes_def:
            if ':' in target_str:
                node_name, input_name = target_str.split(':')
                node_id = assembler.node_map.get(node_name)
                if node_id and node_id in assembler.workflow:
                    assembler.workflow[node_id]['inputs'][input_name] = current_positive

        print(f"[H3 Guide Injector] Successfully chained {applied_count} MiniMax H3 guide(s).")
