import gradio as gr
import os
import random
from PIL import Image
import numpy as np

from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from .shared.utils import (
    get_model_path,
    save_temp_image_from_pil,
    create_mask_from_layer,
    get_latent_type_for_model
)
from .shared.input_processors import (
    process_lora_inputs,
    process_controlnet_inputs,
    process_ipadapter_inputs,
    process_embedding_inputs,
    process_style_inputs,
    process_conditioning_inputs
)
from .shared.vae_utils import process_vae_override_input

def process_inputs(task_type: str, ui_values: dict, seed_override=None):
    """
    Generic input processor for all image generation tasks based on sd_unified_recipe.yaml.
    """
    prefix = task_type
    
    vals = {k.replace(f'{prefix}_', ''): v for k, v in ui_values.items() if isinstance(k, str) and k.startswith(prefix)}

    display_name = vals.get('model_name')
    if not display_name:
        raise gr.Error("Please select a base model.")

    model_type = vals.get('model_type_state', 'sdxl')

    if task_type in ['img2img', 'hires_fix']:
        if vals.get('input_image') is None:
            raise gr.Error(f"An input image is required for {task_type}.")
        vals['input_image'] = save_temp_image_from_pil(vals.get('input_image'), f"{task_type}_input")
    
    elif task_type == 'inpaint':
        img_dict = vals.get('input_image_dict')
        mask_img = create_mask_from_layer(img_dict)
        if not img_dict or img_dict.get('background') is None or mask_img is None:
            raise gr.Error("Inpainting requires an input image and a drawn mask.")
        
        background_img = img_dict['background'].convert("RGBA")
        mask_alpha_np = np.array(mask_img.split()[-1])
        inverted_alpha_np = 255 - mask_alpha_np
        inverted_alpha_pil = Image.fromarray(inverted_alpha_np, mode='L')
        background_img.putalpha(inverted_alpha_pil)
        vals['input_image'] = save_temp_image_from_pil(background_img, "inpaint_composite")

    elif task_type == 'outpaint':
        if vals.get('input_image') is None:
            raise gr.Error("Outpainting requires an input image.")
        vals['input_image'] = save_temp_image_from_pil(vals.get('input_image'), "outpaint_input")
        vals['megapixels'] = 0.25
        vals['grow_mask_by'] = vals.get('feathering')

    model_info = get_model_path(display_name)
    if not model_info:
        raise gr.Error(f"Could not find information for model '{display_name}'. Please check model_list.yaml.")

    if isinstance(model_info, dict):
        vals.update({
            'unet_name': model_info.get('unet'),
            'vae_name': model_info.get('vae'),
            'clip_name': model_info.get('clip'),
            'clip1_name': model_info.get('clip1'),
            'clip2_name': model_info.get('clip2'),
            'clip3_name': model_info.get('clip3'),
            'clip4_name': model_info.get('clip4'),
            'lora_name': model_info.get('lora'),
        })
    else:
        vals['model_name'] = model_info

    vae_override = process_vae_override_input(vals)
    if vae_override:
        vals['vae_name'] = vae_override

    if task_type == 'txt2img':
        vals['latent_type'] = get_latent_type_for_model(display_name)
        if vals['latent_type'] == 'latent':
            vals['latent_generator_template'] = 'EmptyLatentImage'
    
    recipe_path = "workflow_recipes/sd_unified_recipe.yaml"
    dynamic_values = {'task_type': task_type, 'model_type': model_type}
    if 'latent_type' in vals:
        dynamic_values['latent_type'] = vals['latent_type']

    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, dynamic_values, base_path=module_path)

    if 'clip_skip' in vals and vals['clip_skip'] is not None and model_type == 'sd15':
        vals['clip_skip'] = int(vals['clip_skip']) * -1

    embedding_files = process_embedding_inputs(vals)
    embedding_prompt_text = " ".join([f"embedding:{f.replace(os.path.sep, '/')}" for f in embedding_files])
    
    if embedding_prompt_text:
        if vals.get('positive_prompt'):
            vals['positive_prompt'] = f"{vals['positive_prompt']}, {embedding_prompt_text}"
        else:
            vals['positive_prompt'] = embedding_prompt_text

    vals['lora_chain'] = process_lora_inputs(vals)
    vals['controlnet_chain'] = process_controlnet_inputs(vals)
    vals['ipadapter_chain'] = process_ipadapter_inputs(vals)
    vals['style_chain'] = process_style_inputs(vals)
    vals['conditioning_chain'] = process_conditioning_inputs(vals)

    seed = int(vals.get('seed', -1))
    vals['seed'] = seed_override if seed_override is not None else (random.randint(0, 2**32 - 1) if seed == -1 else seed)
    
    workflow = assembler.assemble(vals)
    
    return workflow, {"extra_pnginfo": {"workflow": ""}}