import os
try:
    from utils.app_utils import ensure_file_downloaded
except ImportError:
    ensure_file_downloaded = None

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

    valid_images = valid_images[:3]

    lora_filename = "krea2_style_reference.safetensors"
    if ensure_file_downloaded:
        try:
            ensure_file_downloaded(lora_filename)
        except Exception as e:
            print(f"Warning: Failed to ensure '{lora_filename}' downloaded: {e}")

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    pos_prompt_name = chain_definition.get('pos_prompt_node', 'pos_prompt')
    neg_prompt_name = chain_definition.get('neg_prompt_node', 'neg_prompt')
    clip_loader_name = chain_definition.get('clip_loader_node', 'clip_loader')
    vae_loader_name = chain_definition.get('vae_loader_node', 'vae_loader')

    if ksampler_name not in assembler.node_map:
        print(f"Warning: Target node '{ksampler_name}' for Krea2 Style Reference Edit chain not found. Skipping.")
        return

    ksampler_id = assembler.node_map[ksampler_name]

    if 'model' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'model' input. Skipping.")
        return

    current_model_connection = assembler.workflow[ksampler_id]['inputs']['model']

    vae_connection = None
    if vae_loader_name in assembler.node_map:
        vae_connection = [assembler.node_map[vae_loader_name], 0]
    else:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'VAELoader':
                vae_connection = [node_id, 0]
                break

    clip_connection = None
    if clip_loader_name in assembler.node_map:
        clip_connection = [assembler.node_map[clip_loader_name], 0]
    elif pos_prompt_name in assembler.node_map:
        pos_id = assembler.node_map[pos_prompt_name]
        clip_connection = assembler.workflow[pos_id]['inputs'].get('clip')

    scaled_image_ids = []
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
        scaled_image_ids.append(scale_id)

    lora_loader_id = assembler._get_unique_id()
    lora_loader_node = create_node(assembler, "LoraLoaderModelOnly", "Load LoRA (Krea2 Style Reference)")
    lora_loader_node['inputs']['lora_name'] = lora_filename
    lora_loader_node['inputs']['strength_model'] = 1.0
    lora_loader_node['inputs']['model'] = current_model_connection
    assembler.workflow[lora_loader_id] = lora_loader_node

    assembler.workflow[ksampler_id]['inputs']['model'] = [lora_loader_id, 0]

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

    pos_encode_id = assembler._get_unique_id()
    pos_encode_node = create_node(assembler, "TextEncodeQwenImageEditPlus", "TextEncodeQwenImageEditPlus (Positive)")
    pos_encode_node['inputs']['prompt'] = pos_text
    if clip_connection:
        pos_encode_node['inputs']['clip'] = clip_connection
    if vae_connection:
        pos_encode_node['inputs']['vae'] = vae_connection
    for idx, s_id in enumerate(scaled_image_ids):
        pos_encode_node['inputs'][f"image{idx+1}"] = [s_id, 0]
    assembler.workflow[pos_encode_id] = pos_encode_node

    neg_encode_id = assembler._get_unique_id()
    neg_encode_node = create_node(assembler, "TextEncodeQwenImageEditPlus", "TextEncodeQwenImageEditPlus (Negative)")
    neg_encode_node['inputs']['prompt'] = neg_text
    if clip_connection:
        neg_encode_node['inputs']['clip'] = clip_connection
    if vae_connection:
        neg_encode_node['inputs']['vae'] = vae_connection
    for idx, s_id in enumerate(scaled_image_ids):
        neg_encode_node['inputs'][f"image{idx+1}"] = [s_id, 0]
    assembler.workflow[neg_encode_id] = neg_encode_node

    pos_ref_id = assembler._get_unique_id()
    pos_ref_node = create_node(assembler, "FluxKontextMultiReferenceLatentMethod", "Edit Model Reference Method")
    pos_ref_node['inputs']['reference_latents_method'] = "index_timestep_zero"
    pos_ref_node['inputs']['conditioning'] = [pos_encode_id, 0]
    assembler.workflow[pos_ref_id] = pos_ref_node

    neg_ref_id = assembler._get_unique_id()
    neg_ref_node = create_node(assembler, "FluxKontextMultiReferenceLatentMethod", "Edit Model Reference Method")
    neg_ref_node['inputs']['reference_latents_method'] = "index_timestep_zero"
    neg_ref_node['inputs']['conditioning'] = [neg_encode_id, 0]
    assembler.workflow[neg_ref_id] = neg_ref_node

    assembler.workflow[ksampler_id]['inputs']['positive'] = [pos_ref_id, 0]
    assembler.workflow[ksampler_id]['inputs']['negative'] = [neg_ref_id, 0]

    if pos_prompt_id and pos_prompt_id in assembler.workflow:
        del assembler.workflow[pos_prompt_id]

    if neg_prompt_id and neg_prompt_id in assembler.workflow:
        del assembler.workflow[neg_prompt_id]

    print(f"Krea2 Style Reference Edit injector applied with {len(valid_images)} reference image(s). Original CLIPTextEncode nodes replaced.")
