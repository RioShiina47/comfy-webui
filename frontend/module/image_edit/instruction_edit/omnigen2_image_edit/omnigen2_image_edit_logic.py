import os
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "omnigen2-image-edit_recipe.yaml"

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
    
    main_img = local_ui_values.get('input_image')
    if main_img is None:
        raise ValueError("Please upload an image to edit.")

    all_images = [main_img]
    num_refs = int(local_ui_values.get('ref_count_state', 0))
    ref_images_list = local_ui_values.get('ref_image_inputs', [])
    
    for i in range(num_refs):
        if i < len(ref_images_list) and ref_images_list[i] is not None:
            all_images.append(ref_images_list[i])
    
    image_filenames = [save_temp_image(img) for i, img in enumerate(all_images)]
    local_ui_values['image_stitch_chain'] = image_filenames
    local_ui_values['input_image'] = None

    selected_ratio = local_ui_values.get('aspect_ratio', "1:1 (Square)")
    width, height = ASPECT_RATIO_PRESETS.get(selected_ratio, (1024, 1024))
    local_ui_values['width'] = width
    local_ui_values['height'] = height

    seed = seed_override if seed_override is not None else local_ui_values.get('seed', -1)
    local_ui_values['seed'] = handle_seed(seed)
        
    local_ui_values['filename_prefix'] = get_filename_prefix()

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None