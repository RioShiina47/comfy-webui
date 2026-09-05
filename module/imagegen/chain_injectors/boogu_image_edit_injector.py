def create_node(assembler, class_type, title):
    try:
        node = assembler._get_node_template(class_type)
    except Exception:
        node = {
            "inputs": {},
            "class_type": class_type,
            "_meta": {"title": title}
        }
    node['_meta']['title'] = title
    return node

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

    valid_images = valid_images[:10]

    boogu_prompt_name = chain_definition.get('boogu_prompt_node', 'boogu_prompt')
    vae_loader_name = chain_definition.get('vae_loader_node', 'vae_loader')

    boogu_prompt_id = assembler.node_map.get(boogu_prompt_name)
    if not boogu_prompt_id or boogu_prompt_id not in assembler.workflow:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'TextEncodeBooguEdit':
                boogu_prompt_id = node_id
                break

    if not boogu_prompt_id:
        print(f"Warning: Target node '{boogu_prompt_name}' (TextEncodeBooguEdit) for Boogu Edit chain not found. Skipping.")
        return

    vae_id = assembler.node_map.get(vae_loader_name)
    if not vae_id:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'VAELoader':
                vae_id = node_id
                break

    if vae_id:
        assembler.workflow[boogu_prompt_id]['inputs']['vae'] = [vae_id, 0]

    for i, img_filename in enumerate(valid_images):
        load_id = assembler._get_unique_id()
        load_node = create_node(assembler, "LoadImage", f"Load Reference Image {i+1}")
        load_node['inputs']['image'] = img_filename
        assembler.workflow[load_id] = load_node

        scale_id = assembler._get_unique_id()
        scale_node = create_node(assembler, "ImageScaleToTotalPixels", f"Scale Reference {i+1}")
        scale_node['inputs']['upscale_method'] = "nearest-exact"
        scale_node['inputs']['megapixels'] = 1
        scale_node['inputs']['resolution_steps'] = 1
        scale_node['inputs']['image'] = [load_id, 0]
        assembler.workflow[scale_id] = scale_node

        image_key = f"images.image_{i+1}"
        assembler.workflow[boogu_prompt_id]['inputs'][image_key] = [scale_id, 0]

    print(f"Boogu Edit injector applied with {len(valid_images)} reference image(s). Connected VAE dynamically.")
