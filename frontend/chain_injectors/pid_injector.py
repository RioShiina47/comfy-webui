import random
from module.image_gen.shared.config_loader import load_model_config, load_pid_config

def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    pid_config = {}
    try:
        pid_config = load_pid_config() or {}
    except Exception as e:
        print(f"Error loading PiD config: {e}")

    pid_items = pid_config.get("PiD", [])
    architectures_settings = {}
    default_settings = {"unet_name": "pid_flux1_1024_to_4096_4step_mxfp8.safetensors", "latent_format": "flux"}
    
    for item in pid_items:
        unet_name = item.get("filepath")
        latent_format = item.get("latent_format")
        archs = item.get("architectures", [])
        for arch in archs:
            architectures_settings[arch] = {
                "unet_name": unet_name,
                "latent_format": latent_format
            }
            if arch == "flux1":
                default_settings = {
                    "unet_name": unet_name,
                    "latent_format": latent_format
                }


    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    if ksampler_name not in assembler.node_map:
        print(f"Warning: [PiD Injector] KSampler node '{ksampler_name}' not found. Skipping.")
        return
    
    original_ksampler_id = assembler.node_map[ksampler_name]
    
    original_vae_loader_id = assembler.node_map.get('vae_loader')
    original_vae_decode_id = assembler.node_map.get('vae_decode')
    original_pos_prompt_id = assembler.node_map.get('pos_prompt')
    original_neg_prompt_id = assembler.node_map.get('neg_prompt')
    
    if not original_vae_loader_id:
        for node_id, node_data in assembler.workflow.items():
            if node_data.get('class_type') == 'VAELoader':
                original_vae_loader_id = node_id
                break
                
    if not original_vae_decode_id:
        for node_id, node_data in assembler.workflow.items():
            if node_data.get('class_type') == 'VAEDecode':
                original_vae_decode_id = node_id
                break

    if not original_pos_prompt_id or not original_neg_prompt_id:
        for node_id, node_data in assembler.workflow.items():
            if node_data.get('class_type') == 'CLIPTextEncode':
                title = node_data.get('_meta', {}).get('title', '')
                if 'Positive' in title:
                    original_pos_prompt_id = node_id
                elif 'Negative' in title:
                    original_neg_prompt_id = node_id
                    
    pos_text = ""
    if original_pos_prompt_id and original_pos_prompt_id in assembler.workflow:
        pos_text = assembler.workflow[original_pos_prompt_id]['inputs'].get('text', '')

    neg_text = ""
    if original_neg_prompt_id and original_neg_prompt_id in assembler.workflow:
        neg_text = assembler.workflow[original_neg_prompt_id]['inputs'].get('text', '')

    clip_loader_id = assembler._get_unique_id()
    clip_loader_node = assembler._get_node_template_from_api("CLIPLoader")
    clip_loader_node['inputs']['clip_name'] = "gemma_2_2b_it_elm_fp8_scaled.safetensors"
    clip_loader_node['inputs']['type'] = "pixeldit"
    clip_loader_node['inputs']['device'] = "default"
    assembler.workflow[clip_loader_id] = clip_loader_node

    pos_text_encode_id = assembler._get_unique_id()
    pos_text_encode_node = assembler._get_node_template_from_api("CLIPTextEncode")
    pos_text_encode_node['inputs']['text'] = pos_text
    pos_text_encode_node['inputs']['clip'] = [clip_loader_id, 0]
    assembler.workflow[pos_text_encode_id] = pos_text_encode_node

    neg_text_encode_id = assembler._get_unique_id()
    neg_text_encode_node = assembler._get_node_template_from_api("CLIPTextEncode")
    neg_text_encode_node['inputs']['text'] = neg_text
    neg_text_encode_node['inputs']['clip'] = [clip_loader_id, 0]
    assembler.workflow[neg_text_encode_id] = neg_text_encode_node

    active_model_file = None
    for node_id, node_data in assembler.workflow.items():
        class_type = node_data.get('class_type')
        if class_type == 'UNETLoader':
            active_model_file = node_data.get('inputs', {}).get('unet_name')
            if active_model_file:
                break
        elif class_type == 'CheckpointLoaderSimple':
            active_model_file = node_data.get('inputs', {}).get('ckpt_name')
            if active_model_file:
                break

    architecture = None
    if active_model_file:
        try:
            model_config = load_model_config()
            checkpoints = model_config.get("Checkpoints", {})
            for arch_name, arch_data in checkpoints.items():
                models_list = arch_data.get("models", [])
                for model_entry in models_list:
                    if model_entry.get('path') == active_model_file:
                        architecture = arch_name
                        break
                    components_dict = model_entry.get('components', {})
                    if active_model_file in components_dict.values():
                        architecture = arch_name
                        break
                if architecture:
                    break
        except Exception as e:
            print(f"Error looking up model architecture in PiD injector: {e}")

        if architecture:
            architecture = architecture.lower().replace(" ", "-").replace(".", "")
        else:
            file_lower = active_model_file.lower().replace("-", "").replace("_", "").replace(".", "")
            for arch in sorted(architectures_settings.keys(), key=len, reverse=True):
                candidates = [arch]
                if "-image" in arch:
                    candidates.append(arch.replace("-image", ""))
                if "-i1" in arch:
                    candidates.append(arch.replace("-i1", ""))
                if "-kv" in arch:
                    candidates.append(arch.replace("-kv", ""))
                
                matched = False
                for cand in candidates:
                    if cand.replace("-", "").replace(".", "") in file_lower:
                        architecture = arch
                        matched = True
                        break
                if matched:
                    break

    unet_name = default_settings.get("unet_name")
    latent_format = default_settings.get("latent_format")

    if architecture in architectures_settings:
        arch_config = architectures_settings[architecture]
        unet_name = arch_config.get("unet_name", unet_name)
        latent_format = arch_config.get("latent_format", latent_format)
    else:
        print(f"[PiD Injector] Warning: Model architecture '{architecture}' (file: '{active_model_file}') not explicitly mapped. Using default settings.")

    pid_pos_id = assembler._get_unique_id()
    pid_pos_node = assembler._get_node_template_from_api("PiDConditioning")
    pid_pos_node['inputs']['latent_format'] = latent_format
    pid_pos_node['inputs']['degrade_sigma'] = 0
    pid_pos_node['inputs']['positive'] = [pos_text_encode_id, 0]
    pid_pos_node['inputs']['latent'] = [original_ksampler_id, 0]
    assembler.workflow[pid_pos_id] = pid_pos_node

    pid_neg_id = assembler._get_unique_id()
    pid_neg_node = assembler._get_node_template_from_api("PiDConditioning")
    pid_neg_node['inputs']['latent_format'] = latent_format
    pid_neg_node['inputs']['degrade_sigma'] = 0
    pid_neg_node['inputs']['positive'] = [neg_text_encode_id, 0]
    pid_neg_node['inputs']['latent'] = [original_ksampler_id, 0]
    assembler.workflow[pid_neg_id] = pid_neg_node

    pid_unet_loader_id = assembler._get_unique_id()
    pid_unet_loader_node = assembler._get_node_template_from_api("UNETLoader")
    pid_unet_loader_node['inputs']['unet_name'] = unet_name
    pid_unet_loader_node['inputs']['weight_dtype'] = "default"
    assembler.workflow[pid_unet_loader_id] = pid_unet_loader_node

    orig_width = 1024
    orig_height = 1024
    original_latent_source_id = assembler.node_map.get('latent_source')
    if original_latent_source_id in assembler.workflow:
        node_inputs = assembler.workflow[original_latent_source_id].get('inputs', {})
        if 'width' in node_inputs and 'height' in node_inputs:
            orig_width = node_inputs['width']
            orig_height = node_inputs['height']
        else:
            for node_data in assembler.workflow.values():
                inputs = node_data.get('inputs', {})
                if 'width' in inputs and 'height' in inputs and isinstance(inputs['width'], (int, float)) and isinstance(inputs['height'], (int, float)):
                    if 256 <= inputs['width'] <= 4096 and 256 <= inputs['height'] <= 4096:
                        orig_width = inputs['width']
                        orig_height = inputs['height']
                        break
    else:
        for node_data in assembler.workflow.values():
            inputs = node_data.get('inputs', {})
            if 'width' in inputs and 'height' in inputs and isinstance(inputs['width'], (int, float)) and isinstance(inputs['height'], (int, float)):
                if 256 <= inputs['width'] <= 4096 and 256 <= inputs['height'] <= 4096:
                    orig_width = inputs['width']
                    orig_height = inputs['height']
                    break

    empty_latent_id = assembler._get_unique_id()
    empty_latent_node = assembler._get_node_template_from_api("EmptyChromaRadianceLatentImage")
    empty_latent_node['inputs']['width'] = int(orig_width) * 4
    empty_latent_node['inputs']['height'] = int(orig_height) * 4
    empty_latent_node['inputs']['batch_size'] = 1
    
    if original_latent_source_id in assembler.workflow:
        orig_batch_size = assembler.workflow[original_latent_source_id]['inputs'].get('batch_size') or assembler.workflow[original_latent_source_id]['inputs'].get('amount')
        if orig_batch_size:
            empty_latent_node['inputs']['batch_size'] = orig_batch_size
            
    assembler.workflow[empty_latent_id] = empty_latent_node

    orig_seed = 0
    if original_ksampler_id in assembler.workflow:
        orig_seed = assembler.workflow[original_ksampler_id]['inputs'].get('seed', 0)
        if orig_seed == -1:
            orig_seed = random.randint(0, 2**32 - 1)
        else:
            orig_seed = (orig_seed + 1) % (2**32)

    new_ksampler_id = assembler._get_unique_id()
    new_ksampler_node = assembler._get_node_template_from_api("KSampler")
    new_ksampler_node['inputs']['seed'] = orig_seed
    new_ksampler_node['inputs']['steps'] = 4
    new_ksampler_node['inputs']['cfg'] = 1
    new_ksampler_node['inputs']['sampler_name'] = "lcm"
    new_ksampler_node['inputs']['scheduler'] = "simple"
    new_ksampler_node['inputs']['denoise'] = 1.0
    new_ksampler_node['inputs']['model'] = [pid_unet_loader_id, 0]
    new_ksampler_node['inputs']['positive'] = [pid_pos_id, 0]
    new_ksampler_node['inputs']['negative'] = [pid_neg_id, 0]
    new_ksampler_node['inputs']['latent_image'] = [empty_latent_id, 0]
    assembler.workflow[new_ksampler_id] = new_ksampler_node

    pid_vae_loader_id = assembler._get_unique_id()
    pid_vae_loader_node = assembler._get_node_template_from_api("VAELoader")
    pid_vae_loader_node['inputs']['vae_name'] = "pixel_space"
    assembler.workflow[pid_vae_loader_id] = pid_vae_loader_node

    pid_vae_decode_id = assembler._get_unique_id()
    pid_vae_decode_node = assembler._get_node_template_from_api("VAEDecode")
    pid_vae_decode_node['inputs']['samples'] = [new_ksampler_id, 0]
    pid_vae_decode_node['inputs']['vae'] = [pid_vae_loader_id, 0]
    assembler.workflow[pid_vae_decode_id] = pid_vae_decode_node

    if original_vae_decode_id:
        for node_id, node_data in assembler.workflow.items():
            if 'inputs' in node_data:
                for input_name, input_val in list(node_data['inputs'].items()):
                    if isinstance(input_val, list) and len(input_val) == 2:
                        if input_val[0] == original_vae_decode_id:
                            node_data['inputs'][input_name] = [pid_vae_decode_id, 0]

    if original_vae_loader_id in assembler.workflow:
        del assembler.workflow[original_vae_loader_id]
    if original_vae_decode_id in assembler.workflow:
        del assembler.workflow[original_vae_decode_id]

    print("[PiD Injector] Successfully injected PiD pipeline and replaced VAE decode/loader.")