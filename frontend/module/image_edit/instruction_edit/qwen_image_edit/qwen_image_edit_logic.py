import random
import os
from PIL import Image
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix

def save_temp_image(img: Image.Image, name: str) -> str:
    if not isinstance(img, Image.Image):
        return None
    filename = f"temp_qwen_edit_{name}_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    print(f"Saved temporary image to {filepath}")
    return filename

def process_inputs_logic(params: dict, seed_override=None):
    local_params = params.copy()
    
    main_img = local_params.get('input_image')
    if main_img is None:
        raise gr.Error("Please upload a main Input Image to edit.")

    all_images = [main_img]
    num_refs = int(local_params.get('ref_count_state', 0))
    ref_images_list = local_params.get('ref_image_inputs', [])
    
    for i in range(num_refs):
        if i < len(ref_images_list) and ref_images_list[i] is not None:
            all_images.append(ref_images_list[i])
    
    image_filenames = [save_temp_image(img, f"ref_{i}") for i, img in enumerate(all_images)]
    local_params['image_stitch_chain'] = image_filenames
    local_params['input_image'] = None

    seed = int(local_params.get('seed', -1))
    if seed_override is not None:
        local_params['seed'] = seed_override
    elif seed == -1:
        local_params['seed'] = random.randint(0, 999999999999999)

    local_params['filename_prefix'] = get_filename_prefix()

    recipe_path = "qwen-image-edit_recipe.yaml"
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_params)
    
    return final_workflow, None