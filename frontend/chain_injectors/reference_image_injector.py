import os

def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    valid_images = []
    for item in chain_items:
        if not item:
            continue
        img_path = item
        if isinstance(item, dict):
            img_path = item.get('image') or item.get('filename') or item.get('path')
        if img_path:
            valid_images.append(img_path)

    if not valid_images:
        return

    text_encode_name = chain_definition.get('text_encode_node')
    text_encode_id = None
    if text_encode_name and text_encode_name in assembler.node_map:
        text_encode_id = assembler.node_map[text_encode_name]
    else:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'TextEncodeMageFlowEdit':
                text_encode_id = node_id
                break

    if not text_encode_id or text_encode_id not in assembler.workflow:
        print("Warning: TextEncodeMageFlowEdit node not found for Reference Image chain. Skipping.")
        return

    vae_node_name = chain_definition.get('vae_node', 'vae_loader')
    vae_node_id = assembler.node_map.get(vae_node_name)
    if not vae_node_id:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'VAELoader':
                vae_node_id = node_id
                break

    if vae_node_id:
        assembler.workflow[text_encode_id]['inputs']['vae'] = [vae_node_id, 0]

    for i, img_filename in enumerate(valid_images):
        load_id = assembler._get_unique_id()
        load_node = assembler._get_node_template_from_api("LoadImage")
        load_node['inputs']['image'] = img_filename
        load_node['_meta']['title'] = f"Load Reference Image {i+1}"
        assembler.workflow[load_id] = load_node

        scale_id = assembler._get_unique_id()
        scale_node = assembler._get_node_template_from_api("ImageScaleToTotalPixels")
        scale_node['inputs']['megapixels'] = 1.0
        scale_node['inputs']['upscale_method'] = "nearest-exact"
        scale_node['inputs']['resolution_steps'] = 1
        scale_node['inputs']['image'] = [load_id, 0]
        scale_node['_meta']['title'] = f"Scale Reference {i+1}"
        assembler.workflow[scale_id] = scale_node

        input_key = f"images.image_{i+1}"
        assembler.workflow[text_encode_id]['inputs'][input_key] = [scale_id, 0]

    print(f"Reference Image injector applied. Injected {len(valid_images)} reference images to TextEncodeMageFlowEdit node '{text_encode_id}'.")
