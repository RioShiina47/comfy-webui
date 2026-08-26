def create_node(assembler, class_type, title=None):
    try:
        node = assembler._get_node_template_from_api(class_type)
    except Exception:
        try:
            node = assembler._get_node_template(class_type)
        except Exception:
            node = {
                "inputs": {},
                "class_type": class_type,
                "_meta": {"title": title or class_type}
            }
    if title:
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

    valid_images = valid_images[:2]

    lora_filename = "krea2_identity_edit_v1_2.safetensors"

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    pos_prompt_name = chain_definition.get('pos_prompt_node', 'pos_prompt')
    neg_prompt_name = chain_definition.get('neg_prompt_node', 'neg_prompt')
    clip_loader_name = chain_definition.get('clip_loader_node', 'clip_loader')
    vae_loader_name = chain_definition.get('vae_loader_node', 'vae_loader')

    if ksampler_name not in assembler.node_map:
        print(f"Warning: Target node '{ksampler_name}' for Krea2 Identity Edit chain not found. Skipping.")
        return

    ksampler_id = assembler.node_map[ksampler_name]

    if 'model' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'model' input. Skipping.")
        return

    latent_connection = assembler.workflow[ksampler_id]['inputs'].get('latent_image')
    if not latent_connection:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'latent_image' input. Skipping.")
        return

    current_model_connection = assembler.workflow[ksampler_id]['inputs']['model']

    vae_connection = None
    if vae_loader_name in assembler.node_map:
        vae_connection = [assembler.node_map[vae_loader_name], 0]

    clip_connection = None
    if clip_loader_name in assembler.node_map:
        clip_connection = [assembler.node_map[clip_loader_name], 0]
    elif pos_prompt_name in assembler.node_map:
        pos_id = assembler.node_map[pos_prompt_name]
        clip_connection = assembler.workflow[pos_id]['inputs'].get('clip')

    lora_loader_id = assembler._get_unique_id()
    lora_loader_node = create_node(assembler, "LoraLoaderModelOnly", "Load LoRA (Krea2 Identity Edit)")
    lora_loader_node['inputs']['lora_name'] = lora_filename
    lora_loader_node['inputs']['strength_model'] = 1.0
    lora_loader_node['inputs']['model'] = current_model_connection
    assembler.workflow[lora_loader_id] = lora_loader_node

    image_ids = []
    vae_encode_ids = []

    for i, img_filename in enumerate(valid_images):
        load_id = assembler._get_unique_id()
        load_node = create_node(assembler, "LoadImage", f"Load Image (Ref {i+1})")
        load_node['inputs']['image'] = img_filename
        assembler.workflow[load_id] = load_node
        image_ids.append(load_id)

        vae_enc_id = assembler._get_unique_id()
        vae_enc_node = create_node(assembler, "VAEEncode", f"VAE Encode (Ref {i+1})")
        vae_enc_node['inputs']['pixels'] = [load_id, 0]
        if vae_connection:
            vae_enc_node['inputs']['vae'] = vae_connection
        assembler.workflow[vae_enc_id] = vae_enc_node
        vae_encode_ids.append(vae_enc_id)

    patch_id = assembler._get_unique_id()
    patch_node = create_node(assembler, "Krea2EditModelPatch", "Krea2 Edit (source patch)")
    patch_node['inputs']['ref_boost'] = 4
    patch_node['inputs']['ref_boost_a'] = 1
    patch_node['inputs']['fit_mode'] = "fit"
    patch_node['inputs']['model'] = [lora_loader_id, 0]
    patch_node['inputs']['source_latent'] = [vae_encode_ids[0], 0]
    if vae_connection:
        patch_node['inputs']['vae'] = vae_connection
    patch_node['inputs']['source_image'] = [image_ids[0], 0]
    patch_node['inputs']['target_latent'] = latent_connection

    if len(valid_images) > 1:
        patch_node['inputs']['source_latent_b'] = [vae_encode_ids[1], 0]
        patch_node['inputs']['source_image_b'] = [image_ids[1], 0]

    assembler.workflow[patch_id] = patch_node

    assembler.workflow[ksampler_id]['inputs']['model'] = [patch_id, 0]

    pos_prompt_id = assembler.node_map.get(pos_prompt_name)
    neg_prompt_id = assembler.node_map.get(neg_prompt_name)

    pos_text = ""
    if pos_prompt_id and pos_prompt_id in assembler.workflow:
        pos_text = assembler.workflow[pos_prompt_id]['inputs'].get('text', '')
    elif hasattr(assembler, 'ui_values') and isinstance(assembler.ui_values, dict):
        pos_text = assembler.ui_values.get('positive_prompt') or assembler.ui_values.get('prompt') or ''

    if not pos_text:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict):
                cls = node.get('class_type', '')
                if cls in ['Krea2EditGroundedEncode', 'TextEncodeQwenImageEditPlus', 'CLIPTextEncode']:
                    t = node.get('inputs', {}).get('prompt') or node.get('inputs', {}).get('text')
                    if t:
                        pos_text = t
                        break

    neg_text = ""
    if neg_prompt_id and neg_prompt_id in assembler.workflow:
        neg_text = assembler.workflow[neg_prompt_id]['inputs'].get('text', '')
    elif hasattr(assembler, 'ui_values') and isinstance(assembler.ui_values, dict):
        neg_text = assembler.ui_values.get('negative_prompt') or assembler.ui_values.get('neg_prompt') or ''

    pos_grounded_id = assembler._get_unique_id()
    pos_grounded_node = create_node(assembler, "Krea2EditGroundedEncode", "Krea2 Edit (grounded encode positive)")
    pos_grounded_node['inputs']['prompt'] = pos_text
    pos_grounded_node['inputs']['grounding_px'] = 768
    pos_grounded_node['inputs']['system_prompt'] = ""
    if clip_connection:
        pos_grounded_node['inputs']['clip'] = clip_connection
    pos_grounded_node['inputs']['image'] = [image_ids[0], 0]
    if len(valid_images) > 1:
        pos_grounded_node['inputs']['image_b'] = [image_ids[1], 0]
    assembler.workflow[pos_grounded_id] = pos_grounded_node

    assembler.workflow[ksampler_id]['inputs']['positive'] = [pos_grounded_id, 0]

    neg_grounded_id = assembler._get_unique_id()
    neg_grounded_node = create_node(assembler, "Krea2EditGroundedEncode", "Krea2 Edit (grounded encode negative)")
    neg_grounded_node['inputs']['prompt'] = neg_text
    neg_grounded_node['inputs']['grounding_px'] = 768
    neg_grounded_node['inputs']['system_prompt'] = ""
    if clip_connection:
        neg_grounded_node['inputs']['clip'] = clip_connection
    neg_grounded_node['inputs']['image'] = [image_ids[0], 0]
    if len(valid_images) > 1:
        neg_grounded_node['inputs']['image_b'] = [image_ids[1], 0]
    assembler.workflow[neg_grounded_id] = neg_grounded_node

    assembler.workflow[ksampler_id]['inputs']['negative'] = [neg_grounded_id, 0]

    if pos_prompt_id and pos_prompt_id in assembler.workflow:
        del assembler.workflow[pos_prompt_id]

    if neg_prompt_id and neg_prompt_id in assembler.workflow:
        del assembler.workflow[neg_prompt_id]

    print(f"Krea2 Identity Edit injector applied with {len(valid_images)} reference image(s). Original CLIPTextEncode nodes removed.")
