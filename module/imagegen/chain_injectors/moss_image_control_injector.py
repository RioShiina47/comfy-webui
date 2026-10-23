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
    Injects MOSS-Image Color/Depth Control into the workflow:
    1. Loads the darask0/MOSS-Image-color-depth-control LoRA adapter (moss-control.safetensors)
       which widens the video_embedder from 128 to 384 channels.
    2. Loads and scales the color map and/or depth map.
    3. Adds the MOSSImageControl node to encode maps into latents, concatenate them into
       concat_latent_image conditioning, and set timestep schedule (start/end percent).
    4. Connects the patched model and conditioned positive/negative outputs to KSampler.
    """
    if not chain_items:
        return

    # Normalize chain_items: may be a single dict or list of dicts
    if isinstance(chain_items, dict):
        item_data = chain_items
    elif isinstance(chain_items, list) and len(chain_items) > 0:
        item_data = chain_items[0]
        if not isinstance(item_data, dict):
            return
    else:
        return

    color_map = item_data.get('color_map')
    depth_map = item_data.get('depth_map')

    if not color_map and not depth_map:
        return

    lora_filename = "moss-control.safetensors"
    if ensure_file_downloaded:
        try:
            ensure_file_downloaded(lora_filename)
        except Exception as e:
            print(f"Warning: Failed to ensure '{lora_filename}' downloaded: {e}")

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    vae_loader_name = chain_definition.get('vae_loader_node', 'vae_loader')

    if ksampler_name not in assembler.node_map:
        print(f"Warning: Target node '{ksampler_name}' for MOSS-Image Control chain not found. Skipping.")
        return

    ksampler_id = assembler.node_map[ksampler_name]

    if 'model' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'model' input. Skipping.")
        return

    if 'positive' not in assembler.workflow[ksampler_id]['inputs'] or \
       'negative' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'positive' or 'negative' inputs. Skipping.")
        return

    current_model_connection = assembler.workflow[ksampler_id]['inputs']['model']
    current_pos_connection = assembler.workflow[ksampler_id]['inputs']['positive']
    current_neg_connection = assembler.workflow[ksampler_id]['inputs']['negative']

    vae_connection = None
    if vae_loader_name in assembler.node_map:
        vae_connection = [assembler.node_map[vae_loader_name], 0]
    else:
        for node_id, node in assembler.workflow.items():
            if isinstance(node, dict) and node.get('class_type') == 'VAELoader':
                vae_connection = [node_id, 0]
                break

    if not vae_connection:
        print("Warning: VAE connection not found for MOSS-Image Control. Skipping.")
        return

    megapixels = float(chain_definition.get('megapixels', 1.0))

    # 1. Inject LoRA loader onto the model path
    lora_loader_id = assembler._get_unique_id()
    lora_loader_node = create_node(assembler, "LoraLoaderModelOnly", "Load LoRA (MOSS-Image Control)")
    lora_loader_node['inputs']['lora_name'] = lora_filename
    lora_loader_node['inputs']['strength_model'] = 1.0
    lora_loader_node['inputs']['model'] = current_model_connection
    assembler.workflow[lora_loader_id] = lora_loader_node
    current_model_connection = [lora_loader_id, 0]

    # 2. Process Color Map if provided
    scale_color_id = None
    if color_map:
        load_color_id = assembler._get_unique_id()
        load_color_node = create_node(assembler, "LoadImage", "Load MOSS Color Map")
        load_color_node['inputs']['image'] = color_map
        assembler.workflow[load_color_id] = load_color_node

        scale_color_id = assembler._get_unique_id()
        scale_color_node = create_node(assembler, "ImageScaleToTotalPixels", "Scale MOSS Color Map")
        scale_color_node['inputs']['upscale_method'] = "nearest-exact"
        scale_color_node['inputs']['megapixels'] = megapixels
        scale_color_node['inputs']['resolution_steps'] = 1
        scale_color_node['inputs']['image'] = [load_color_id, 0]
        assembler.workflow[scale_color_id] = scale_color_node

    # 3. Process Depth Map if provided
    scale_depth_id = None
    if depth_map:
        load_depth_id = assembler._get_unique_id()
        load_depth_node = create_node(assembler, "LoadImage", "Load MOSS Depth Map")
        load_depth_node['inputs']['image'] = depth_map
        assembler.workflow[load_depth_id] = load_depth_node

        scale_depth_id = assembler._get_unique_id()
        scale_depth_node = create_node(assembler, "ImageScaleToTotalPixels", "Scale MOSS Depth Map")
        scale_depth_node['inputs']['upscale_method'] = "nearest-exact"
        scale_depth_node['inputs']['megapixels'] = megapixels
        scale_depth_node['inputs']['resolution_steps'] = 1
        scale_depth_node['inputs']['image'] = [load_depth_id, 0]
        assembler.workflow[scale_depth_id] = scale_depth_node

    # 4. Create MOSSImageControl node
    moss_control_id = assembler._get_unique_id()
    moss_control_node = create_node(assembler, "MOSSImageControl", "MOSS-Image Color/Depth Control")
    moss_control_node['inputs']['positive'] = current_pos_connection
    moss_control_node['inputs']['negative'] = current_neg_connection
    moss_control_node['inputs']['vae'] = vae_connection
    moss_control_node['inputs']['strength'] = float(item_data.get('strength', 1.0))
    moss_control_node['inputs']['start_percent'] = float(item_data.get('start_percent', 0.0))
    moss_control_node['inputs']['end_percent'] = float(item_data.get('end_percent', 0.5))

    if scale_color_id:
        moss_control_node['inputs']['color_map'] = [scale_color_id, 0]
    if scale_depth_id:
        moss_control_node['inputs']['depth_map'] = [scale_depth_id, 0]

    assembler.workflow[moss_control_id] = moss_control_node

    # 5. Connect back to KSampler
    assembler.workflow[ksampler_id]['inputs']['model'] = current_model_connection
    assembler.workflow[ksampler_id]['inputs']['positive'] = [moss_control_id, 0]
    assembler.workflow[ksampler_id]['inputs']['negative'] = [moss_control_id, 1]

    print(f"MOSS-Image Color/Depth Control injector applied (color_map: {bool(color_map)}, depth_map: {bool(depth_map)}).")
