def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')
    if ksampler_name not in assembler.node_map:
        print(f"Warning: Target node '{ksampler_name}' for ControlNet chain not found. Skipping chain injection.")
        return
        
    ksampler_id = assembler.node_map[ksampler_name]

    vae_source_str = chain_definition.get('vae_source')
    vae_connection = None
    if vae_source_str and ':' in vae_source_str:
        vae_node_name, vae_idx_str = vae_source_str.split(':')
        if vae_node_name in assembler.node_map:
            vae_connection = [assembler.node_map[vae_node_name], int(vae_idx_str)]
        else:
            print(f"Warning: VAE source node '{vae_node_name}' for ControlNet chain not found.")

    template_name = chain_definition.get('template', 'ControlNetApplyAdvanced')

    # Resolution scaling: SD1.5 defaults to 0.75 MP, all other architectures default to 1.0 MP
    model_type = getattr(assembler, 'dynamic_values', {}).get('model_type', '')
    if isinstance(model_type, str) and model_type.lower() in ('sd15', 'sd1.5', 'sd-1.5'):
        default_megapixels = 0.75
    else:
        default_megapixels = 1.0
    megapixels = float(chain_definition.get('megapixels', default_megapixels))

    # 1. Fun ControlNet (Qwen-Image-2.1)
    if template_name in ("ZImageFunControlnet", "QwenImage21FunControlNetApply"):
        if 'model' not in assembler.workflow[ksampler_id]['inputs']:
            print(f"Warning: KSampler node '{ksampler_name}' is missing 'model' input. Skipping Fun ControlNet.")
            return

        current_model_connection = assembler.workflow[ksampler_id]['inputs']['model']

        for item_data in chain_items:
            patch_loader_id = assembler._get_unique_id()
            patch_loader_node = assembler._get_node_template("ModelPatchLoader")
            patch_loader_node['inputs']['name'] = item_data['control_net_name']
            assembler.workflow[patch_loader_id] = patch_loader_node

            image_loader_id = assembler._get_unique_id()
            image_loader_node = assembler._get_node_template("LoadImage")
            image_loader_node['inputs']['image'] = item_data['image']
            assembler.workflow[image_loader_id] = image_loader_node

            image_scaler_id = assembler._get_unique_id()
            image_scaler_node = assembler._get_node_template("ImageScaleToTotalPixels")
            image_scaler_node['inputs']['image'] = [image_loader_id, 0]
            image_scaler_node['inputs']['upscale_method'] = 'nearest-exact'
            image_scaler_node['inputs']['megapixels'] = megapixels
            assembler.workflow[image_scaler_id] = image_scaler_node

            apply_cn_id = assembler._get_unique_id()
            apply_cn_node = assembler._get_node_template(template_name)
            apply_cn_node['inputs']['strength'] = item_data.get('strength', 1.0)
            if 'start_percent' in apply_cn_node.get('inputs', {}):
                apply_cn_node['inputs']['start_percent'] = item_data.get('start_percent', 0.0)
            if 'end_percent' in apply_cn_node.get('inputs', {}):
                apply_cn_node['inputs']['end_percent'] = item_data.get('end_percent', 1.0)
            apply_cn_node['inputs']['model'] = current_model_connection
            apply_cn_node['inputs']['model_patch'] = [patch_loader_id, 0]
            if 'image' in apply_cn_node.get('inputs', {}):
                apply_cn_node['inputs']['image'] = [image_scaler_id, 0]
            elif 'control_image' in apply_cn_node.get('inputs', {}):
                apply_cn_node['inputs']['control_image'] = [image_scaler_id, 0]
            if vae_connection and 'vae' in apply_cn_node.get('inputs', {}):
                apply_cn_node['inputs']['vae'] = vae_connection

            assembler.workflow[apply_cn_id] = apply_cn_node
            current_model_connection = [apply_cn_id, 0]

        assembler.workflow[ksampler_id]['inputs']['model'] = current_model_connection
        print(f"Fun ControlNet injector applied. KSampler model input redirected through {len(chain_items)} node(s).")
        return

    # 2. Anima LLLite ControlNet
    if template_name == "AnimaLLLiteApply":
        if 'model' not in assembler.workflow[ksampler_id]['inputs']:
            print(f"Warning: KSampler node '{ksampler_name}' is missing 'model' input. Skipping Anima LLLite.")
            return
            
        current_model_connection = assembler.workflow[ksampler_id]['inputs']['model']

        for item_data in chain_items:
            image_loader_id = assembler._get_unique_id()
            image_loader_node = assembler._get_node_template("LoadImage")
            image_loader_node['inputs']['image'] = item_data['image']
            assembler.workflow[image_loader_id] = image_loader_node

            image_scaler_id = assembler._get_unique_id()
            image_scaler_node = assembler._get_node_template("ImageScaleToTotalPixels")
            image_scaler_node['inputs']['image'] = [image_loader_id, 0]
            image_scaler_node['inputs']['upscale_method'] = 'nearest-exact'
            image_scaler_node['inputs']['megapixels'] = megapixels
            assembler.workflow[image_scaler_id] = image_scaler_node

            patch_loader_id = assembler._get_unique_id()
            patch_loader_node = assembler._get_node_template("ModelPatchLoader")
            patch_loader_node['inputs']['name'] = item_data['control_net_name']
            assembler.workflow[patch_loader_id] = patch_loader_node

            apply_cn_id = assembler._get_unique_id()
            apply_cn_node = assembler._get_node_template("AnimaLLLiteApply")
            apply_cn_node['inputs']['strength'] = item_data.get('strength', 1.0)
            apply_cn_node['inputs']['start_percent'] = item_data.get('start_percent', 0.0)
            apply_cn_node['inputs']['end_percent'] = item_data.get('end_percent', 1.0)
            apply_cn_node['inputs']['model'] = current_model_connection
            apply_cn_node['inputs']['model_patch'] = [patch_loader_id, 0]
            apply_cn_node['inputs']['image'] = [image_scaler_id, 0]

            assembler.workflow[apply_cn_id] = apply_cn_node
            current_model_connection = [apply_cn_id, 0]

        assembler.workflow[ksampler_id]['inputs']['model'] = current_model_connection
        print(f"Anima LLLite ControlNet injector applied. KSampler model input redirected through {len(chain_items)} node(s).")
        return

    # 3. DiffSynth ControlNet (Z-Image)
    if template_name in ("QwenImageDiffsynthControlnet", "ZImage_ControlNet"):
        model_sampler_name = chain_definition.get('model_sampler_node')
        target_node_id = None
        target_input_name = 'model'

        if model_sampler_name and model_sampler_name in assembler.node_map:
            model_sampler_id = assembler.node_map[model_sampler_name]
            if target_input_name in assembler.workflow[model_sampler_id]['inputs']:
                target_node_id = model_sampler_id

        if not target_node_id:
            if target_input_name in assembler.workflow[ksampler_id]['inputs']:
                target_node_id = ksampler_id
            else:
                print(f"Warning: Neither model_sampler nor ksampler found with 'model' input. Skipping DiffSynth ControlNet.")
                return

        current_model_connection = assembler.workflow[target_node_id]['inputs'][target_input_name]

        for item_data in chain_items:
            patch_loader_id = assembler._get_unique_id()
            patch_loader_node = assembler._get_node_template("ModelPatchLoader")
            patch_loader_node['inputs']['name'] = item_data['control_net_name']
            assembler.workflow[patch_loader_id] = patch_loader_node

            image_loader_id = assembler._get_unique_id()
            image_loader_node = assembler._get_node_template("LoadImage")
            image_loader_node['inputs']['image'] = item_data['image']
            assembler.workflow[image_loader_id] = image_loader_node

            image_scaler_id = assembler._get_unique_id()
            image_scaler_node = assembler._get_node_template("ImageScaleToTotalPixels")
            image_scaler_node['inputs']['image'] = [image_loader_id, 0]
            image_scaler_node['inputs']['upscale_method'] = 'nearest-exact'
            image_scaler_node['inputs']['megapixels'] = megapixels
            assembler.workflow[image_scaler_id] = image_scaler_node

            apply_cn_id = assembler._get_unique_id()
            apply_cn_node = assembler._get_node_template(template_name)
            apply_cn_node['inputs']['strength'] = item_data.get('strength', 1.0)
            apply_cn_node['inputs']['model'] = current_model_connection
            apply_cn_node['inputs']['model_patch'] = [patch_loader_id, 0]
            apply_cn_node['inputs']['image'] = [image_scaler_id, 0]
            if vae_connection and 'vae' in apply_cn_node.get('inputs', {}):
                apply_cn_node['inputs']['vae'] = vae_connection

            assembler.workflow[apply_cn_id] = apply_cn_node
            current_model_connection = [apply_cn_id, 0]

        assembler.workflow[target_node_id]['inputs'][target_input_name] = current_model_connection
        print(f"DiffSynth ControlNet injector applied. Target 'model' input redirected through {len(chain_items)} node(s).")
        return

    # 4. Krea-2 ControlNet
    if template_name == "Krea2ControlApply":
        if 'model' not in assembler.workflow[ksampler_id]['inputs']:
            print(f"Warning: KSampler node '{ksampler_name}' is missing 'model' input. Skipping Krea2 ControlNet.")
            return

        latent_connection = assembler.workflow[ksampler_id]['inputs'].get('latent_image')
        if not latent_connection:
            print(f"Warning: KSampler node '{ksampler_name}' is missing 'latent_image' input. Krea2 ControlNet requires it. Skipping.")
            return

        current_model_connection = assembler.workflow[ksampler_id]['inputs']['model']

        for item_data in chain_items:
            image_loader_id = assembler._get_unique_id()
            image_loader_node = assembler._get_node_template("LoadImage")
            image_loader_node['inputs']['image'] = item_data['image']
            assembler.workflow[image_loader_id] = image_loader_node

            image_scaler_id = assembler._get_unique_id()
            image_scaler_node = assembler._get_node_template("ImageScaleToTotalPixels")
            image_scaler_node['inputs']['image'] = [image_loader_id, 0]
            image_scaler_node['inputs']['upscale_method'] = 'nearest-exact'
            image_scaler_node['inputs']['megapixels'] = megapixels
            image_scaler_node['inputs']['resolution_steps'] = 1
            assembler.workflow[image_scaler_id] = image_scaler_node

            lora_loader_id = assembler._get_unique_id()
            lora_loader_node = assembler._get_node_template("Krea2ControlLoRALoader")
            lora_loader_node['inputs']['lora_name'] = item_data['control_net_name']
            lora_loader_node['inputs']['strength'] = item_data.get('strength', 1.0)
            lora_loader_node['inputs']['model'] = current_model_connection
            assembler.workflow[lora_loader_id] = lora_loader_node

            img_encode_id = assembler._get_unique_id()
            img_encode_node = assembler._get_node_template("Krea2ControlImageEncode")
            img_encode_node['inputs']['resize'] = "match_latent_size"
            img_encode_node['inputs']['upscale_method'] = "lanczos"
            img_encode_node['inputs']['crop'] = "center"
            img_encode_node['inputs']['channel_mode'] = "rgb"
            img_encode_node['inputs']['normalize'] = "none"
            img_encode_node['inputs']['invert'] = False
            img_encode_node['inputs']['batch_mode'] = "independent_images"
            img_encode_node['inputs']['control_image'] = [image_scaler_id, 0]
            if vae_connection:
                img_encode_node['inputs']['vae'] = vae_connection
            img_encode_node['inputs']['latent'] = latent_connection
            assembler.workflow[img_encode_id] = img_encode_node

            apply_cn_id = assembler._get_unique_id()
            apply_cn_node = assembler._get_node_template("Krea2ControlApply")
            apply_cn_node['inputs']['model'] = [lora_loader_id, 0]
            apply_cn_node['inputs']['control_latent'] = [img_encode_id, 0]

            assembler.workflow[apply_cn_id] = apply_cn_node
            current_model_connection = [apply_cn_id, 0]

        assembler.workflow[ksampler_id]['inputs']['model'] = current_model_connection
        print(f"Krea2 ControlNet injector applied. KSampler model input redirected through {len(chain_items)} node(s).")
        return

    # 5. Default: Classical Conditioning ControlNet (SD1.5, SDXL, SD3.5, FLUX.1, Qwen-Image)
    if 'positive' not in assembler.workflow[ksampler_id]['inputs'] or \
       'negative' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' is missing 'positive' or 'negative' inputs. Skipping ControlNet chain.")
        return

    if not vae_connection:
        print("Warning: 'vae_source' definition missing or invalid in the recipe for the ControlNet chain. Skipping.")
        return

    current_positive_connection = assembler.workflow[ksampler_id]['inputs']['positive']
    current_negative_connection = assembler.workflow[ksampler_id]['inputs']['negative']

    for item_data in chain_items:
        cn_loader_id = assembler._get_unique_id()
        cn_loader_node = assembler._get_node_template("ControlNetLoader")
        cn_loader_node['inputs']['control_net_name'] = item_data['control_net_name']
        assembler.workflow[cn_loader_id] = cn_loader_node

        image_loader_id = assembler._get_unique_id()
        image_loader_node = assembler._get_node_template("LoadImage")
        image_loader_node['inputs']['image'] = item_data['image']
        assembler.workflow[image_loader_id] = image_loader_node

        image_scaler_id = assembler._get_unique_id()
        image_scaler_node = assembler._get_node_template("ImageScaleToTotalPixels")
        image_scaler_node['inputs']['image'] = [image_loader_id, 0]
        image_scaler_node['inputs']['upscale_method'] = 'nearest-exact'
        image_scaler_node['inputs']['megapixels'] = megapixels
        assembler.workflow[image_scaler_id] = image_scaler_node

        apply_cn_id = assembler._get_unique_id()
        apply_cn_node = assembler._get_node_template(template_name)
        apply_cn_node['inputs']['strength'] = item_data.get('strength', 1.0)
        apply_cn_node['inputs']['positive'] = current_positive_connection
        apply_cn_node['inputs']['negative'] = current_negative_connection
        apply_cn_node['inputs']['control_net'] = [cn_loader_id, 0]
        apply_cn_node['inputs']['image'] = [image_scaler_id, 0]
        apply_cn_node['inputs']['vae'] = vae_connection

        assembler.workflow[apply_cn_id] = apply_cn_node
        current_positive_connection = [apply_cn_id, 0]
        current_negative_connection = [apply_cn_id, 1]

    assembler.workflow[ksampler_id]['inputs']['positive'] = current_positive_connection
    assembler.workflow[ksampler_id]['inputs']['negative'] = current_negative_connection
    print(f"ControlNet injector applied. KSampler inputs redirected through {len(chain_items)} ControlNet node(s).")