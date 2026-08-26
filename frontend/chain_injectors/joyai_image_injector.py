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

    valid_images = valid_images[:6]

    pos_prompt_name = chain_definition.get('pos_prompt_node', 'pos_prompt')
    neg_prompt_name = chain_definition.get('neg_prompt_node', 'neg_prompt')
    vae_node_name = chain_definition.get('vae_node', 'vae_loader')

    if pos_prompt_name not in assembler.node_map:
        print(f"Warning: Positive prompt node '{pos_prompt_name}' not found for JoyAI Reference chain. Skipping.")
        return

    if vae_node_name not in assembler.node_map:
        print(f"Warning: VAE loader node '{vae_node_name}' not found for JoyAI Reference chain. Skipping.")
        return

    pos_prompt_id = assembler.node_map[pos_prompt_name]
    neg_prompt_id = assembler.node_map.get(neg_prompt_name)
    vae_node_id = assembler.node_map[vae_node_name]

    assembler.workflow[pos_prompt_id]['inputs']['vae'] = [vae_node_id, 0]
    if neg_prompt_id and neg_prompt_id in assembler.workflow:
        assembler.workflow[neg_prompt_id]['inputs']['vae'] = [vae_node_id, 0]

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

        input_key = f"images.image{i}"
        assembler.workflow[pos_prompt_id]['inputs'][input_key] = [scale_id, 0]
        if neg_prompt_id and neg_prompt_id in assembler.workflow:
            assembler.workflow[neg_prompt_id]['inputs'][input_key] = [scale_id, 0]

    print(f"JoyAI Reference injector applied. Injected {len(valid_images)} reference images to JoyAI text encoding nodes.")
