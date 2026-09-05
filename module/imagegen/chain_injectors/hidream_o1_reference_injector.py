def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    ksampler_name = chain_definition.get('ksampler_node', 'ksampler')

    if ksampler_name not in assembler.node_map:
        print(f"Warning: KSampler node '{ksampler_name}' not found for HiDream-O1 Reference chain. Skipping.")
        return

    ksampler_id = assembler.node_map[ksampler_name]

    if 'positive' not in assembler.workflow[ksampler_id]['inputs'] or 'negative' not in assembler.workflow[ksampler_id]['inputs']:
        print(f"Warning: KSampler node '{ksampler_name}' missing positive/negative inputs. Skipping.")
        return

    current_pos_conditioning = assembler.workflow[ksampler_id]['inputs']['positive']
    current_neg_conditioning = assembler.workflow[ksampler_id]['inputs']['negative']

    ref_images_id = assembler._get_unique_id()
    ref_images_node = assembler._get_node_template_from_api("HiDreamO1ReferenceImages")
    
    ref_images_node['inputs']['positive'] = current_pos_conditioning
    ref_images_node['inputs']['negative'] = current_neg_conditioning

    for i, img_filename in enumerate(chain_items):
        if i >= 10:
            break
            
        load_id = assembler._get_unique_id()
        load_node = assembler._get_node_template_from_api("LoadImage")
        load_node['inputs']['image'] = img_filename
        load_node['_meta']['title'] = f"Load Reference Image {i+1}"
        assembler.workflow[load_id] = load_node
        
        scale_id = assembler._get_unique_id()
        scale_node = assembler._get_node_template_from_api("ImageScaleToTotalPixels")
        scale_node['inputs']['megapixels'] = 1.0
        scale_node['inputs']['upscale_method'] = "lanczos"
        scale_node['inputs']['image'] = [load_id, 0]
        scale_node['_meta']['title'] = f"Scale Reference {i+1}"
        assembler.workflow[scale_id] = scale_node
        
        ref_images_node['inputs'][f'images.image_{i+1}'] = [scale_id, 0]

    assembler.workflow[ref_images_id] = ref_images_node

    assembler.workflow[ksampler_id]['inputs']['positive'] = [ref_images_id, 0]
    assembler.workflow[ksampler_id]['inputs']['negative'] = [ref_images_id, 1]
    
    print(f"HiDream-O1 Reference injector applied. Re-routed inputs through {min(len(chain_items), 10)} reference images.")