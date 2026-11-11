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


def _get_autogrow_image_slot_names(assembler, target_node_id, input_name='images'):
    """
    Resolve the real per-image slot key template for an Autogrow image input.

    ComfyUI exposes two flavours of Autogrow template:
      * TemplatePrefix(prefix="image_") -> names are 0-based: image_0, image_1, ...
      * TemplateNames(names=["image_1", ...]) -> names start at image_1

    Returns a list of slot names indexed from 0 (i.e. index i is the name for the i-th image),
    or None when the node schema cannot be introspected.
    """
    target_node = assembler.workflow.get(target_node_id)
    if not isinstance(target_node, dict):
        return None

    class_type = target_node.get('class_type')
    if not class_type:
        return None

    node_info = None
    try:
        from comfy_webui_core import node_info_manager
        node_info = node_info_manager.get_node_info(class_type)
    except Exception:
        node_info = None

    if not isinstance(node_info, dict):
        return None

    spec = None
    inputs_cfg = node_info.get('input', {}) or {}
    for category in ('required', 'optional'):
        candidate = (inputs_cfg.get(category) or {}).get(input_name)
        if candidate:
            spec = candidate
            break

    if not isinstance(spec, (list, tuple)) or len(spec) < 2 or not isinstance(spec[1], dict):
        return None

    template = spec[1].get('template')
    if not isinstance(template, dict):
        return None

    names = template.get('names')
    if isinstance(names, (list, tuple)) and names:
        return [str(n) for n in names]

    prefix = template.get('prefix')
    if isinstance(prefix, str):
        max_count = template.get('max', 100)
        if not isinstance(max_count, int) or max_count <= 0:
            max_count = 100
        return [f"{prefix}{i}" for i in range(max_count)]

    return None


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
        supported_types = ('TextEncodeMageFlowEdit', 'TextEncodeQwenImage21', 'TextEncodeBooguEdit', 'TextEncodeMingImageEdit', 'TextEncodeFibo2')
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

    # Derive the node's real autogrow slot naming (0-based prefix vs 1-based names)
    # instead of assuming image_1..image_N, which silently breaks 0-based nodes.
    input_name = chain_definition.get('input_name', 'images')
    slot_names = _get_autogrow_image_slot_names(assembler, target_node_id, input_name=input_name)
    index_base = chain_definition.get('slot_index_base')

    for i, img_filename in enumerate(valid_images):
        if slot_names is not None:
            if i >= len(slot_names):
                print(f"Warning: Reference Image chain reached the node's slot limit ({len(slot_names)}). Stopping.")
                break
            slot_key = f"{input_name}.{slot_names[i]}"
        else:
            # Fallback when node schema is unavailable: honor explicit chain config or use 1-based legacy default.
            base = 1 if index_base is None else int(index_base)
            slot_key = f"{input_name}.image_{i + base}"

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

        assembler.workflow[target_node_id]['inputs'][slot_key] = [scale_id, 0]

    node_class = assembler.workflow[target_node_id].get('class_type', 'Unknown')
    print(f"Reference Image injector applied: {len(valid_images)} image(s) injected into '{target_node_id}' ({node_class}), max_images={max_images}, slots={slot_key if valid_images else '-'}.")
