def inject(assembler, chain_definition, chain_items):
    if not chain_items:
        return

    guider_node_name = chain_definition.get('guider_node')
    if not guider_node_name or guider_node_name not in assembler.node_map:
        print(f"Warning: Guider node '{guider_node_name}' for Flux2 Reference chain not found. Skipping.")
        return
        
    guider_id = assembler.node_map[guider_node_name]

    if 'conditioning' not in assembler.workflow[guider_id]['inputs']:
        print(f"Warning: Guider node '{guider_node_name}' is missing 'conditioning' input. Skipping Flux2 Reference chain.")
        return

    vae_node_name = chain_definition.get('vae_node', 'vae_loader')
    if vae_node_name not in assembler.node_map:
        print(f"Warning: VAE loader node '{vae_node_name}' not found for Flux2 Reference chain. Skipping.")
        return
    vae_node_id = assembler.node_map[vae_node_name]
        
    current_conditioning_connection = assembler.workflow[guider_id]['inputs']['conditioning']
    
    for i, img_filename in enumerate(chain_items[:10]):
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
        
        vae_encode_id = assembler._get_unique_id()
        vae_encode_node = assembler._get_node_template_from_api("VAEEncode")
        vae_encode_node['inputs']['pixels'] = [scale_id, 0]
        vae_encode_node['inputs']['vae'] = [vae_node_id, 0]
        vae_encode_node['_meta']['title'] = f"VAE Encode Reference {i+1}"
        assembler.workflow[vae_encode_id] = vae_encode_node

        ref_latent_id = assembler._get_unique_id()
        ref_latent_node = assembler._get_node_template_from_api("ReferenceLatent")
        ref_latent_node['inputs']['conditioning'] = current_conditioning_connection
        ref_latent_node['inputs']['latent'] = [vae_encode_id, 0]
        ref_latent_node['_meta']['title'] = f"ReferenceLatent {i+1}"
        assembler.workflow[ref_latent_id] = ref_latent_node
        current_conditioning_connection = [ref_latent_id, 0]

    assembler.workflow[guider_id]['inputs']['conditioning'] = current_conditioning_connection
    
    print(f"Flux2 Reference injector applied. Guider conditioning re-routed through {len(chain_items[:10])} reference images.")