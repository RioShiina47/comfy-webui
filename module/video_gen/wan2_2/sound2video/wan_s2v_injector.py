import math
from copy import deepcopy

def inject(assembler, chain_definition, chain_items):
    if not isinstance(chain_items, dict):
        print("[S2V Injector] chain_items is not a dict, skipping.")
        return

    num_chunks = chain_items.get('num_chunks', 1)
    base_seed = chain_items.get('seed', 0)
    
    start_sampler_id = assembler.node_map.get(chain_definition['start_node'])
    end_decode_id = assembler.node_map.get(chain_definition['end_node'])
    
    img_from_batch_id = assembler.node_map.get('image_from_batch')
    
    if not all([start_sampler_id, end_decode_id, img_from_batch_id]):
        print("[S2V Injector] Error: Could not find start_node, end_node, or 'image_from_batch' node in the recipe map. Skipping.")
        return
        
    if num_chunks <= 1:
        assembler.workflow[img_from_batch_id]['inputs']['batch_index'] = 1
        print("[S2V Injector] Only one chunk needed, no extension required.")
        return

    assembler.workflow[end_decode_id]['inputs'].pop('samples', None)

    all_sampler_outputs = [[start_sampler_id, 0]]
    last_sampler_output = [start_sampler_id, 0]
    
    base_sampler_node = assembler.workflow[start_sampler_id]
    base_s2v_node = assembler.workflow[assembler.node_map['s2v_preprocessor']]

    num_extensions = num_chunks - 1
    for i in range(num_extensions):
        extend_id = assembler._get_unique_id()
        sampler_id = assembler._get_unique_id()

        extend_node = assembler._get_node_template_from_api("WanSoundImageToVideoExtend")
        sampler_node = deepcopy(base_sampler_node)

        extend_node['inputs']['video_latent'] = last_sampler_output
        extend_node['inputs']['length'] = base_s2v_node['inputs']['length']
        extend_node['inputs']['positive'] = base_s2v_node['inputs']['positive']
        extend_node['inputs']['negative'] = base_s2v_node['inputs']['negative']
        extend_node['inputs']['vae'] = base_s2v_node['inputs']['vae']
        extend_node['inputs']['audio_encoder_output'] = base_s2v_node['inputs']['audio_encoder_output']
        extend_node['inputs']['ref_image'] = base_s2v_node['inputs']['ref_image']
        
        sampler_node['inputs']['seed'] = base_seed + i + 1
        sampler_node['inputs']['positive'] = [extend_id, 0]
        sampler_node['inputs']['negative'] = [extend_id, 1]
        sampler_node['inputs']['latent_image'] = [extend_id, 2]

        assembler.workflow[extend_id] = extend_node
        assembler.workflow[sampler_id] = sampler_node
        
        last_sampler_output = [sampler_id, 0]
        all_sampler_outputs.append(last_sampler_output)

    if len(all_sampler_outputs) > 1:
        last_concat_output = all_sampler_outputs[0]
        for i in range(1, len(all_sampler_outputs)):
            concat_id = assembler._get_unique_id()
            concat_node = assembler._get_node_template_from_api("LatentConcat")
            concat_node['inputs']['dim'] = "t"
            concat_node['inputs']['samples1'] = last_concat_output
            concat_node['inputs']['samples2'] = all_sampler_outputs[i]
            assembler.workflow[concat_id] = concat_node
            last_concat_output = [concat_id, 0]
        final_latent_source = last_concat_output
    else:
        final_latent_source = all_sampler_outputs[0]

    assembler.workflow[end_decode_id]['inputs']['samples'] = final_latent_source
    
    assembler.workflow[img_from_batch_id]['inputs']['batch_index'] = num_chunks
    
    print(f"[S2V Injector] Injected {num_extensions} extension chunks, for a total of {num_chunks} chunks.")