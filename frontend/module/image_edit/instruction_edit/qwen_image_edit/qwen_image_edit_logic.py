import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed
from core.input_processors import process_lora_inputs

ASPECT_RATIO_PRESETS = {
    "1:1 (Square)": (1328, 1328), 
    "16:9 (Landscape)": (1664, 928), 
    "9:16 (Portrait)": (928, 1664),
    "4:3 (Classic)": (1472, 1104), 
    "3:4 (Classic Portrait)": (1104, 1472),
    "3:2 (Photography)": (1536, 1024),
    "2:3 (Photography Portrait)": (1024, 1536)
}

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

    lora_chain = process_lora_inputs(local_params, prefix="qwen_edit")
    if lora_chain:
        print(f"Applying {len(lora_chain)} LoRA(s).")
    
    local_params['lora_chain'] = lora_chain

    seed = seed_override if seed_override is not None else int(local_params.get('seed', -1))
    local_params['seed'] = handle_seed(seed)

    local_params['filename_prefix'] = get_filename_prefix()

    model_selection = local_params.get('model_version', 'Qwen-Image-Edit-2511')
    if model_selection == 'Qwen-Image-Edit-2511':
        recipe_path = "qwen-image-edit_2511_recipe.yaml"
    elif model_selection == 'Qwen-Image-Edit-2509':
        recipe_path = "qwen-image-edit_2509_recipe.yaml"
    else:
        recipe_path = "qwen-image-edit_recipe.yaml"

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_params)
    
    return final_workflow, None

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    selected_ratio = local_ui_values.get('aspect_ratio', "1:1 (Square)")
    width, height = ASPECT_RATIO_PRESETS[selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    return process_inputs_logic(local_ui_values, seed_override)