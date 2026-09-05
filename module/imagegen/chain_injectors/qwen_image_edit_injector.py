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

    valid_images = valid_images[:3]

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    pos_prompt_name = chain_definition.get('pos_prompt_node', 'pos_prompt')
    neg_prompt_name = chain_definition.get('neg_prompt_node', 'neg_prompt')
    vae_loader_name = chain_definition.get('vae_loader_node', 'vae_loader')
    model_sampler_name = chain_definition.get('model_sampler_node', 'model_sampler')

    if ksampler_name not in assembler.node_map:
        print(f"Warning: Target node '{ksampler_name}' for Qwen-Image Edit chain not found. Skipping.")
        return

    ksampler_id = assembler.node_map[ksampler_name]
    pos_prompt_id = assembler.node_map.get(pos_prompt_name)
    neg_prompt_id = assembler.node_map.get(neg_prompt_name)

    if not pos_prompt_id or not neg_prompt_id:
        print("Warning: Positive or negative prompt node not found for Qwen-Image Edit chain. Skipping.")
        return

    vae_id = assembler.node_map.get(vae_loader_name)
    if not vae_id:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'VAELoader':
                vae_id = node_id
                break

    if vae_id:
        assembler.workflow[pos_prompt_id]['inputs']['vae'] = [vae_id, 0]
        assembler.workflow[neg_prompt_id]['inputs']['vae'] = [vae_id, 0]

    for i, img_filename in enumerate(valid_images):
        load_id = assembler._get_unique_id()
        load_node = create_node(assembler, "LoadImage", f"Load Reference Image {i+1}")
        load_node['inputs']['image'] = img_filename
        assembler.workflow[load_id] = load_node

        scale_id = assembler._get_unique_id()
        scale_node = create_node(assembler, "ImageScaleToTotalPixels", f"Scale Reference {i+1}")
        scale_node['inputs']['upscale_method'] = "lanczos"
        scale_node['inputs']['megapixels'] = 1
        scale_node['inputs']['resolution_steps'] = 1
        scale_node['inputs']['image'] = [load_id, 0]
        assembler.workflow[scale_id] = scale_node

        image_key = f"image{i+1}"
        assembler.workflow[pos_prompt_id]['inputs'][image_key] = [scale_id, 0]
        assembler.workflow[neg_prompt_id]['inputs'][image_key] = [scale_id, 0]

    pos_ref_id = assembler._get_unique_id()
    pos_ref_node = create_node(assembler, "FluxKontextMultiReferenceLatentMethod", "Edit Model Reference Method")
    pos_ref_node['inputs']['reference_latents_method'] = "index_timestep_zero"
    pos_ref_node['inputs']['conditioning'] = [pos_prompt_id, 0]
    assembler.workflow[pos_ref_id] = pos_ref_node

    neg_ref_id = assembler._get_unique_id()
    neg_ref_node = create_node(assembler, "FluxKontextMultiReferenceLatentMethod", "Edit Model Reference Method")
    neg_ref_node['inputs']['reference_latents_method'] = "index_timestep_zero"
    neg_ref_node['inputs']['conditioning'] = [neg_prompt_id, 0]
    assembler.workflow[neg_ref_id] = neg_ref_node

    assembler.workflow[ksampler_id]['inputs']['positive'] = [pos_ref_id, 0]
    assembler.workflow[ksampler_id]['inputs']['negative'] = [neg_ref_id, 0]

    model_sampler_id = assembler.node_map.get(model_sampler_name)
    if not model_sampler_id:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'ModelSamplingAuraFlow':
                model_sampler_id = node_id
                break

    if model_sampler_id and model_sampler_id in assembler.workflow:
        assembler.workflow[model_sampler_id]['inputs']['shift'] = 3

    current_model_connection = assembler.workflow[ksampler_id]['inputs']['model']
    cfg_norm_id = assembler._get_unique_id()
    cfg_norm_node = create_node(assembler, "CFGNorm", "CFGNorm")
    cfg_norm_node['inputs']['strength'] = 1
    cfg_norm_node['inputs']['pre_cfg'] = False
    cfg_norm_node['inputs']['model'] = current_model_connection
    assembler.workflow[cfg_norm_id] = cfg_norm_node
    assembler.workflow[ksampler_id]['inputs']['model'] = [cfg_norm_id, 0]

    print(f"Qwen-Image Edit injector applied with {len(valid_images)} reference image(s). Connected VAE dynamically.")
