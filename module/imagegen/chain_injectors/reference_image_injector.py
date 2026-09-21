def create_node(assembler, class_type, title):
    try:
        node = assembler._get_node_template(class_type)
    except Exception:
        node = {
            "inputs": {},
            "class_type": class_type,
            "_meta": {"title": title}
        }
    if "_meta" not in node:
        node["_meta"] = {}
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

    max_images = chain_definition.get('max_images', 10)
    if isinstance(max_images, int) and max_images > 0:
        valid_images = valid_images[:max_images]

    # Target prompt/text_encode node resolution
    target_node_name = (
        chain_definition.get('text_encode_node')
        or chain_definition.get('prompt_node')
        or chain_definition.get('boogu_prompt_node')
        or 'prompt'
    )
    target_node_id = assembler.node_map.get(target_node_name)

    if not target_node_id or target_node_id not in assembler.workflow:
        supported_types = ('TextEncodeMageFlowEdit', 'TextEncodeQwenImage21', 'TextEncodeBooguEdit')
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') in supported_types:
                target_node_id = node_id
                break

    if not target_node_id or target_node_id not in assembler.workflow:
        print(f"Warning: Target text encode node '{target_node_name}' for Reference Image chain not found. Skipping.")
        return

    # VAE loader node resolution
    vae_node_name = chain_definition.get('vae_node') or chain_definition.get('vae_loader_node', 'vae_loader')
    vae_node_id = assembler.node_map.get(vae_node_name)
    if not vae_node_id or vae_node_id not in assembler.workflow:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'VAELoader':
                vae_node_id = node_id
                break

    if vae_node_id:
        assembler.workflow[target_node_id]['inputs']['vae'] = [vae_node_id, 0]

    upscale_method = chain_definition.get('upscale_method', 'nearest-exact')
    megapixels = chain_definition.get('megapixels', 1.0)
    resolution_steps = chain_definition.get('resolution_steps', 1)

    for i, img_filename in enumerate(valid_images):
        load_id = assembler._get_unique_id()
        load_node = create_node(assembler, "LoadImage", f"Load Reference Image {i+1}")
        load_node['inputs']['image'] = img_filename
        assembler.workflow[load_id] = load_node

        scale_id = assembler._get_unique_id()
        scale_node = create_node(assembler, "ImageScaleToTotalPixels", f"Scale Reference {i+1}")
        scale_node['inputs']['upscale_method'] = upscale_method
        scale_node['inputs']['megapixels'] = megapixels
        scale_node['inputs']['resolution_steps'] = resolution_steps
        scale_node['inputs']['image'] = [load_id, 0]
        assembler.workflow[scale_id] = scale_node

        input_key = f"images.image_{i+1}"
        assembler.workflow[target_node_id]['inputs'][input_key] = [scale_id, 0]

    node_class = assembler.workflow[target_node_id].get('class_type', 'Unknown')
    print(f"Reference Image injector applied: {len(valid_images)} image(s) injected into '{target_node_id}' ({node_class}), max_images={max_images}.")
