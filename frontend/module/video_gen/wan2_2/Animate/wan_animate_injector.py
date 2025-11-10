import math
from copy import deepcopy

MAX_FRAMES_PER_CHUNK = 77

def inject(assembler, chain_definition, chain_items):
    if not isinstance(chain_items, dict):
        return
        
    total_frames = chain_items.get('video_length', 0)
    if total_frames <= 0:
        return
        
    num_chunks = math.ceil(total_frames / MAX_FRAMES_PER_CHUNK)
    
    create_video_node_name = chain_definition.get('create_video_node')
    create_video_id = assembler.node_map.get(create_video_node_name)
    if not create_video_id:
        print(f"Warning: [Animate Injector] Create Video node '{create_video_node_name}' not found. Skipping.")
        return
    assembler.workflow[create_video_id]['inputs'].pop('images', None)

    template_preprocessor_id = assembler.node_map[chain_definition['template_preprocessor']]
    template_sampler_id = assembler.node_map[chain_definition['template_sampler']]
    
    template_preprocessor_node = assembler.workflow[template_preprocessor_id]
    template_sampler_node = assembler.workflow[template_sampler_id]

    previous_animate_preprocessor_id = None
    previous_vae_decode_id = None
    all_image_from_batch_outputs = []

    for i in range(num_chunks):
        preprocessor_id = assembler._get_unique_id()
        sampler_id = assembler._get_unique_id()
        trim_latent_id = assembler._get_unique_id()
        vae_decode_id = assembler._get_unique_id()
        image_from_batch_id = assembler._get_unique_id()

        preprocessor_node = deepcopy(template_preprocessor_node)
        sampler_node = deepcopy(template_sampler_node)
        trim_latent_node = assembler._get_node_template_from_api("TrimVideoLatent")
        vae_decode_node = assembler._get_node_template_from_api("VAEDecode")
        image_from_batch_node = assembler._get_node_template_from_api("ImageFromBatch")

        remaining_frames = total_frames - (i * MAX_FRAMES_PER_CHUNK)
        current_chunk_length = min(remaining_frames, MAX_FRAMES_PER_CHUNK)
        preprocessor_node['inputs']['length'] = current_chunk_length

        is_char_replacement_mode = chain_definition.get('mode') == 'char_replacement'

        if is_char_replacement_mode:
            preprocessor_node['inputs']['character_mask'] = [assembler.node_map['blockify_mask'], 0]
            preprocessor_node['inputs']['background_video'] = [assembler.node_map['draw_mask_on_image'], 0]
        else:
            preprocessor_node['inputs'].pop('character_mask', None)
            preprocessor_node['inputs'].pop('background_video', None)

        if i == 0:
            preprocessor_node['inputs']['video_frame_offset'] = 0
        else:
            preprocessor_node['inputs']['video_frame_offset'] = [previous_animate_preprocessor_id, 5]
            preprocessor_node['inputs']['continue_motion'] = [previous_vae_decode_id, 0]

        sampler_node['inputs']['seed'] = int(chain_items.get('seed', 0)) + i
        sampler_node['inputs']['positive'] = [preprocessor_id, 0]
        sampler_node['inputs']['negative'] = [preprocessor_id, 1]
        sampler_node['inputs']['latent_image'] = [preprocessor_id, 2]

        trim_latent_node['inputs']['samples'] = [sampler_id, 0]
        trim_latent_node['inputs']['trim_amount'] = [preprocessor_id, 3]

        vae_decode_node['inputs']['samples'] = [trim_latent_id, 0]
        vae_decode_node['inputs']['vae'] = preprocessor_node['inputs']['vae']

        image_from_batch_node['inputs']['image'] = [vae_decode_id, 0]
        image_from_batch_node['inputs']['batch_index'] = [preprocessor_id, 4]
        image_from_batch_node['inputs']['length'] = 4096

        assembler.workflow[preprocessor_id] = preprocessor_node
        assembler.workflow[sampler_id] = sampler_node
        assembler.workflow[trim_latent_id] = trim_latent_node
        assembler.workflow[vae_decode_id] = vae_decode_node
        assembler.workflow[image_from_batch_id] = image_from_batch_node

        previous_animate_preprocessor_id = preprocessor_id
        previous_vae_decode_id = vae_decode_id
        all_image_from_batch_outputs.append([image_from_batch_id, 0])

    if len(all_image_from_batch_outputs) > 1:
        last_batch_output = all_image_from_batch_outputs[0]
        for i in range(1, len(all_image_from_batch_outputs)):
            batch_combiner_id = assembler._get_unique_id()
            batch_combiner_node = assembler._get_node_template_from_api("ImageBatch")
            batch_combiner_node['inputs']['image1'] = last_batch_output
            batch_combiner_node['inputs']['image2'] = all_image_from_batch_outputs[i]
            assembler.workflow[batch_combiner_id] = batch_combiner_node
            last_batch_output = [batch_combiner_id, 0]
        final_image_source = last_batch_output
    else:
        final_image_source = all_image_from_batch_outputs[0]

    assembler.workflow[create_video_id]['inputs']['images'] = final_image_source

    nodes_to_remove = [
        chain_definition['template_preprocessor'],
        chain_definition['template_sampler'],
        chain_definition['template_trim_latent'],
        chain_definition['template_vae_decode'],
        chain_definition['template_image_from_batch']
    ]
    for node_name in nodes_to_remove:
        node_id_to_remove = assembler.node_map.get(node_name)
        if node_id_to_remove and node_id_to_remove in assembler.workflow:
            del assembler.workflow[node_id_to_remove]
    
    print(f"[Animate Injector] Injected {num_chunks} animate chunks for a total of {total_frames} frames.")