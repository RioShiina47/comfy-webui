import math
from copy import deepcopy

MAX_FRAMES_PER_CHUNK = 81

def inject(assembler, chain_definition, chain_items):
    if not isinstance(chain_items, dict):
        print("[InfiniteTalk Multi Injector] chain_items is not a dict, skipping.")
        return

    num_chunks = chain_items.get('num_chunks', 1)
    base_seed = chain_items.get('seed', 0)

    if num_chunks <= 1:
        print("[InfiniteTalk Multi Injector] Only one chunk needed, no extension required.")
        return

    create_video_id = assembler.node_map.get(chain_definition['create_video_node'])
    if not create_video_id:
        print(f"[InfiniteTalk Multi Injector] Error: Create Video node '{chain_definition['create_video_node']}' not found. Skipping.")
        return
    assembler.workflow[create_video_id]['inputs'].pop('images', None)

    preprocessor_id = assembler.node_map.get(chain_definition['preprocessor_node'])
    sampler_id = assembler.node_map.get(chain_definition['sampler_node'])
    decoder_id = assembler.node_map.get(chain_definition['decoder_node'])
    
    if not all([preprocessor_id, sampler_id, decoder_id]):
        print("[InfiniteTalk Multi Injector] Error: Could not find preprocessor, sampler, or decoder nodes in the recipe map. Skipping.")
        return
    
    base_preprocessor_node = assembler.workflow[preprocessor_id]
    base_sampler_node = assembler.workflow[sampler_id]

    previous_decoded_frames_output = [decoder_id, 0]
    all_final_image_outputs = [previous_decoded_frames_output]

    num_extensions = num_chunks - 1
    for i in range(num_extensions):
        new_preprocessor_id = assembler._get_unique_id()
        new_sampler_id = assembler._get_unique_id()
        new_decoder_id = assembler._get_unique_id()
        new_img_from_batch_id = assembler._get_unique_id()

        preprocessor_node = deepcopy(base_preprocessor_node)
        sampler_node = deepcopy(base_sampler_node)
        decoder_node = deepcopy(assembler.workflow[decoder_id])
        img_from_batch_node = assembler._get_node_template_from_api("ImageFromBatch")

        preprocessor_node['inputs']['previous_frames'] = previous_decoded_frames_output
        sampler_node['inputs']['seed'] = base_seed + i + 1
        
        sampler_node['inputs']['model'] = [new_preprocessor_id, 0]
        sampler_node['inputs']['positive'] = [new_preprocessor_id, 1]
        sampler_node['inputs']['negative'] = [new_preprocessor_id, 2]
        sampler_node['inputs']['latent_image'] = [new_preprocessor_id, 3]
        
        decoder_node['inputs']['samples'] = [new_sampler_id, 0]

        img_from_batch_node['inputs']['image'] = [new_decoder_id, 0]
        img_from_batch_node['inputs']['batch_index'] = [new_preprocessor_id, 4]
        img_from_batch_node['inputs']['length'] = 4096

        assembler.workflow[new_preprocessor_id] = preprocessor_node
        assembler.workflow[new_sampler_id] = sampler_node
        assembler.workflow[new_decoder_id] = decoder_node
        assembler.workflow[new_img_from_batch_id] = img_from_batch_node

        previous_decoded_frames_output = [new_img_from_batch_id, 0]
        all_final_image_outputs.append(previous_decoded_frames_output)

    if len(all_final_image_outputs) > 1:
        last_batch_output = all_final_image_outputs[0]
        for i in range(1, len(all_final_image_outputs)):
            batch_combiner_id = assembler._get_unique_id()
            batch_combiner_node = assembler._get_node_template_from_api("ImageBatch")
            batch_combiner_node['inputs']['image1'] = last_batch_output
            batch_combiner_node['inputs']['image2'] = all_final_image_outputs[i]
            assembler.workflow[batch_combiner_id] = batch_combiner_node
            last_batch_output = [batch_combiner_id, 0]
        final_image_source = last_batch_output
    else:
        final_image_source = all_final_image_outputs[0]

    assembler.workflow[create_video_id]['inputs']['images'] = final_image_source
    
    print(f"[InfiniteTalk Multi Injector] Injected {num_extensions} extension chunks, for a total of {num_chunks} chunks.")