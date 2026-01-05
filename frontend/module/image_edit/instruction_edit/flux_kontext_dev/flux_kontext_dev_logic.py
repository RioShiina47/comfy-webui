import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed
from core.input_processors import process_lora_inputs

WORKFLOW_RECIPE_PATH = "flux-kontext-dev_recipe.yaml"
PREFIX = "flux_kontext_dev"

ASPECT_RATIO_PRESETS = {
    "1:1 (Square)": (1024, 1024), 
    "16:9 (Landscape)": (1344, 768), 
    "9:16 (Portrait)": (768, 1344),
    "4:3 (Classic)": (1152, 896), 
    "3:4 (Classic Portrait)": (896, 1152),
    "3:2 (Photography)": (1216, 832),
    "2:3 (Photography Portrait)": (832, 1216)
}

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    key = lambda name: f"{PREFIX}_{name}"
    
    main_img = local_ui_values.get(key('input_image'))
    if main_img is None:
        raise ValueError("Please upload an image to edit.")

    all_images = [main_img]
    num_refs = int(local_ui_values.get(key('ref_count_state'), 0))
    ref_images_list = local_ui_values.get(key('ref_image_inputs'), [])
    
    for i in range(num_refs):
        if i < len(ref_images_list) and ref_images_list[i] is not None:
            all_images.append(ref_images_list[i])
    
    image_filenames = [save_temp_image(img) for i, img in enumerate(all_images)]
    
    lora_chain = process_lora_inputs(local_ui_values, prefix=PREFIX)
    if lora_chain:
        print(f"Applying {len(lora_chain)} LoRA(s).")
    
    selected_ratio = local_ui_values.get(key('aspect_ratio'), "1:1 (Square)")
    width, height = ASPECT_RATIO_PRESETS.get(selected_ratio, (1024, 1024))

    params_for_assembler = {
        'width': width,
        'height': height,
        'positive_prompt': local_ui_values.get(key('positive_prompt')),
        'negative_prompt': local_ui_values.get(key('negative_prompt')),
        'batch_size': local_ui_values.get(key('batch_size')),
        'image_stitch_chain': image_filenames,
        'lora_chain': lora_chain,
        'filename_prefix': get_filename_prefix(),
    }

    seed = seed_override if seed_override is not None else local_ui_values.get(key('seed'), -1)
    params_for_assembler['seed'] = handle_seed(seed)

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(params_for_assembler)
    
    return final_workflow, None