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
        node.setdefault('_meta', {})['title'] = title
    return node


def inject(assembler, chain_definition, chain_items):
    """
    Injects MOSS-Image Edit (Reference Image Edit) into the workflow:
    1. Loads the officially packaged Edit LoRA (moss_image_edit_lora_rank_64_bf16.safetensors)
       onto the model path.
    2. Loads and scales up to 10 reference images with ImageScaleToTotalPixels.
    3. Replaces positive/negative prompt encoding with native TextEncodeMOSSImageEdit nodes,
       which attach reference image latents and text tokens.
    4. Connects patched model and conditioned positive/negative outputs to KSampler.
    """
    if not chain_items:
        return

    # Normalize chain_items: dict or list of dicts/strings
    edit_format = "Moss-Image-Edit"
    strength = 1.0
    valid_images = []

    if isinstance(chain_items, dict):
        chain_items = [chain_items]

    if isinstance(chain_items, list):
        for item in chain_items:
            if not item:
                continue
            if isinstance(item, dict):
                raw_imgs = item.get('images')
                if isinstance(raw_imgs, list):
                    for img in raw_imgs:
                        p = img.get('image') or img.get('filename') or img.get('path') if isinstance(img, dict) else img
                        if p and isinstance(p, str) and p.strip():
                            valid_images.append(p.strip())
                elif isinstance(raw_imgs, str) and raw_imgs.strip():
                    valid_images.append(raw_imgs.strip())

                p = item.get('image') or item.get('filename') or item.get('path')
                if p and isinstance(p, str) and p.strip():
                    valid_images.append(p.strip())

                if 'edit_format' in item:
                    edit_format = item['edit_format']
                if 'strength' in item:
                    try:
                        strength = float(item['strength'])
                    except (ValueError, TypeError):
                        pass
            elif isinstance(item, str) and item.strip():
                valid_images.append(item.strip())

    if not valid_images:
        return

    valid_images = valid_images[:10]

    lora_filename = "moss_image_edit_lora_rank_64_bf16.safetensors"
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
        print(f"Warning: Target node '{ksampler_name}' for MOSS-Image Edit chain not found. Skipping.")
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

    if not vae_connection:
        print("Warning: VAE connection not found for MOSS-Image Edit. Skipping.")
        return

    pos_prompt_id = assembler.node_map.get(pos_prompt_name)
    neg_prompt_id = assembler.node_map.get(neg_prompt_name)

    clip_connection = None
    if clip_loader_name in assembler.node_map:
        clip_connection = [assembler.node_map[clip_loader_name], 0]
    elif pos_prompt_id and 'clip' in assembler.workflow.get(pos_prompt_id, {}).get('inputs', {}):
        clip_connection = assembler.workflow[pos_prompt_id]['inputs']['clip']
    else:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'CLIPLoader':
                clip_connection = [node_id, 0]
                break

    if not clip_connection:
        print("Warning: CLIP connection not found for MOSS-Image Edit. Skipping.")
        return

    # 1. Inject LoRA loader on model path if strength > 0
    if strength > 0.0:
        lora_loader_id = assembler._get_unique_id()
        lora_loader_node = create_node(assembler, "LoraLoaderModelOnly", "Load LoRA (MOSS-Image Edit)")
        lora_loader_node['inputs']['lora_name'] = lora_filename
        lora_loader_node['inputs']['strength_model'] = strength
        lora_loader_node['inputs']['model'] = current_model_connection
        assembler.workflow[lora_loader_id] = lora_loader_node
        current_model_connection = [lora_loader_id, 0]

    megapixels = float(chain_definition.get('megapixels', 1.0))

    # 2. Process Reference Images
    scale_image_ids = []
    for i, img_filename in enumerate(valid_images):
        load_id = assembler._get_unique_id()
        load_node = create_node(assembler, "LoadImage", f"Load Reference Image {i+1} (MOSS Edit)")
        load_node['inputs']['image'] = img_filename
        assembler.workflow[load_id] = load_node

        scale_id = assembler._get_unique_id()
        scale_node = create_node(assembler, "ImageScaleToTotalPixels", f"Scale Reference {i+1} (MOSS Edit)")
        scale_node['inputs']['upscale_method'] = "lanczos"
        scale_node['inputs']['megapixels'] = megapixels
        scale_node['inputs']['resolution_steps'] = 1
        scale_node['inputs']['image'] = [load_id, 0]
        assembler.workflow[scale_id] = scale_node
        scale_image_ids.append(scale_id)

    # 3. Extract Prompt Texts
    pos_text = ""
    if pos_prompt_id and pos_prompt_id in assembler.workflow:
        pos_text = assembler.workflow[pos_prompt_id]['inputs'].get('text', '') or assembler.workflow[pos_prompt_id]['inputs'].get('prompt', '')
    elif hasattr(assembler, 'ui_values') and isinstance(assembler.ui_values, dict):
        pos_text = assembler.ui_values.get('positive_prompt') or assembler.ui_values.get('prompt') or ''

    neg_text = ""
    if neg_prompt_id and neg_prompt_id in assembler.workflow:
        neg_text = assembler.workflow[neg_prompt_id]['inputs'].get('text', '') or assembler.workflow[neg_prompt_id]['inputs'].get('prompt', '')
    elif hasattr(assembler, 'ui_values') and isinstance(assembler.ui_values, dict):
        neg_text = assembler.ui_values.get('negative_prompt') or assembler.ui_values.get('neg_prompt') or ''

    # 4. Positive TextEncodeMOSSImageEdit
    pos_edit_id = assembler._get_unique_id()
    pos_edit_node = create_node(assembler, "TextEncodeMOSSImageEdit", "Text Encode MOSS-Image Edit")
    pos_edit_node['inputs']['clip'] = clip_connection
    pos_edit_node['inputs']['prompt'] = pos_text
    pos_edit_node['inputs']['vae'] = vae_connection
    pos_edit_node['inputs']['edit_format'] = edit_format
    pos_edit_node['inputs']['image_strength'] = 1.0
    for i, sc_id in enumerate(scale_image_ids):
        pos_edit_node['inputs'][f"images.image_{i}"] = [sc_id, 0]
    pos_edit_node['inputs'] = {k: v for k, v in pos_edit_node['inputs'].items() if v is not None}
    assembler.workflow[pos_edit_id] = pos_edit_node

    # 5. Negative TextEncodeMOSSImageEdit
    neg_edit_id = assembler._get_unique_id()
    neg_edit_node = create_node(assembler, "TextEncodeMOSSImageEdit", "Text Encode MOSS-Image Edit")
    neg_edit_node['inputs']['clip'] = clip_connection
    neg_edit_node['inputs']['prompt'] = neg_text
    neg_edit_node['inputs']['vae'] = vae_connection
    neg_edit_node['inputs']['edit_format'] = edit_format
    neg_edit_node['inputs']['image_strength'] = 1.0
    for i, sc_id in enumerate(scale_image_ids):
        neg_edit_node['inputs'][f"images.image_{i}"] = [sc_id, 0]
    neg_edit_node['inputs'] = {k: v for k, v in neg_edit_node['inputs'].items() if v is not None}
    assembler.workflow[neg_edit_id] = neg_edit_node

    # 6. Reconnect model and redirect conditioning references in workflow
    assembler.workflow[ksampler_id]['inputs']['model'] = current_model_connection

    # Redirect all existing connections that referenced pos_prompt_id / neg_prompt_id
    if pos_prompt_id:
        for node in assembler.workflow.values():
            if isinstance(node, dict) and 'inputs' in node:
                for k, v in node['inputs'].items():
                    if isinstance(v, list) and len(v) == 2 and v[0] == pos_prompt_id:
                        node['inputs'][k] = [pos_edit_id, 0]

    if neg_prompt_id:
        for node in assembler.workflow.values():
            if isinstance(node, dict) and 'inputs' in node:
                for k, v in node['inputs'].items():
                    if isinstance(v, list) and len(v) == 2 and v[0] == neg_prompt_id:
                        node['inputs'][k] = [neg_edit_id, 0]

    # Clean up original CLIPTextEncode nodes if they are in workflow
    if pos_prompt_id and pos_prompt_id in assembler.workflow:
        del assembler.workflow[pos_prompt_id]
    if neg_prompt_id and neg_prompt_id in assembler.workflow:
        del assembler.workflow[neg_prompt_id]


    print(f"MOSS-Image Edit injector applied with {len(valid_images)} reference image(s) (format: {edit_format}, strength: {strength}).")
