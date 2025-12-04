import os

from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix
from core.utils import save_temp_image, handle_seed

WORKFLOW_RECIPE_PATH = "bytedance_uso_recipe.yaml"
PREFIX = "uso"

def process_uso_reference_inputs(vals):
    references = []
    ref_images = vals.get('reference_images', [])
    if not ref_images:
        return []
    
    ref_guidances = vals.get('reference_guidances', [])

    for i in range(len(ref_images)):
        image_pil = ref_images[i]
        guidance = ref_guidances[i] if i < len(ref_guidances) else 3.5

        if image_pil is not None:
            image_filename = save_temp_image(image_pil)
            references.append({
                "image": image_filename,
                "guidance": guidance,
            })
    return references

def process_uso_style_reference_inputs(vals):
    references = []
    style_ref_images = vals.get('style_reference_images', [])
    if not style_ref_images:
        return []
    
    for i in range(len(style_ref_images)):
        image_pil = style_ref_images[i]
        if image_pil is not None:
            image_filename = save_temp_image(image_pil)
            references.append({"image": image_filename})
    return references

def process_inputs(ui_values, seed_override=None):
    vals = {k.replace(f'{PREFIX}_', ''): v for k, v in ui_values.items() if isinstance(k, str) and k.startswith(PREFIX)}

    if not vals.get('positive_prompt'):
        raise ValueError("Prompt is required.")

    seed = seed_override if seed_override is not None else vals.get('seed', -1)
    vals['seed'] = handle_seed(seed)
    
    vals['filename_prefix'] = ui_values.get(f'{PREFIX}_filename_prefix', get_filename_prefix())

    vals['uso_reference_chain'] = process_uso_reference_inputs(vals)
    vals['uso_style_reference_chain'] = process_uso_style_reference_inputs(vals)

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(WORKFLOW_RECIPE_PATH, base_path=module_path)
    workflow = assembler.assemble(vals)
    return workflow, {"extra_pnginfo": {"workflow": ""}}