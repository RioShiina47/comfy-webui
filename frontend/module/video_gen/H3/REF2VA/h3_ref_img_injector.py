def inject(assembler, chain_definition, chain_items):
    """
    Dynamically injects uploaded reference images into MiniMaxH3ReferenceToVideo node.
    chain_items is a list of saved image filenames.
    """
    if not chain_items or not isinstance(chain_items, list):
        return

    target_node_name = chain_definition.get('target_node', 'minimax_h3')
    target_node_id = assembler.node_map.get(target_node_name)
    if not target_node_id:
        print(f"[H3 RefImg Injector] Error: Target node '{target_node_name}' not found in node_map.")
        return

    minimax_h3_node = assembler.workflow[target_node_id]

    valid_images = [img for img in chain_items if img]
    for idx, img_file in enumerate(valid_images):
        load_id = assembler._get_unique_id()
        load_node = assembler._get_node_template_from_api("LoadImage")
        load_node['inputs']['image'] = img_file
        assembler.workflow[load_id] = load_node

        scale_id = assembler._get_unique_id()
        scale_node = assembler._get_node_template_from_api("ImageScaleToTotalPixels")
        scale_node['inputs']['image'] = [load_id, 0]
        scale_node['inputs']['upscale_method'] = "nearest-exact"
        scale_node['inputs']['megapixels'] = 1
        scale_node['inputs']['resolution_steps'] = 1
        assembler.workflow[scale_id] = scale_node

        param_name = f"ref_images.ref_image_{idx}"
        minimax_h3_node['inputs'][param_name] = [scale_id, 0]

    print(f"[H3 RefImg Injector] Successfully injected {len(valid_images)} reference image(s).")
