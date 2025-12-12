import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

def process_inputs_logic(params: dict, seed_override=None):
    local_params = params.copy()
    
    main_img = local_params.get('input_image')
    if main_img is None:
        raise ValueError("Please upload a main Input Image to edit.")

    all_images = [main_img]
    num_refs = int(local_params.get('ref_count_state', 0))
    ref_images_list = local_params.get('ref_image_inputs', [])
    
    for i in range(num_refs):
        if i < len(ref_images_list) and ref_images_list[i] is not None:
            all_images.append(ref_images_list[i])
    
    image_filenames = [save_temp_image(img) for i, img in enumerate(all_images)]
    local_params['image_stitch_chain'] = image_filenames
    local_params['input_image'] = None

    lora_filename = local_params.get('apply_lora')
    if lora_filename and lora_filename != "None":
        print(f"Applying LoRA: {lora_filename}")
        local_params['lora_chain'] = [{
            "lora_name": lora_filename,
            "strength_model": 1.0
        }]
    else:
        local_params['lora_chain'] = []

    seed = seed_override if seed_override is not None else int(local_params.get('seed', -1))
    local_params['seed'] = handle_seed(seed)

    local_params['filename_prefix'] = get_filename_prefix()

    recipe_path = "qwen-image-edit_recipe.yaml"
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_params)
    
    return final_workflow, None