import gradio as gr
import os
from core.config import CIVITAI_API_KEY
from core.download_utils import get_lora_path, get_embedding_path
from .utils import save_temp_image_from_pil
from .config_loader import load_ipadapter_presets

def process_lora_inputs(vals):
    loras = []
    lora_sources = vals.get('loras_sources', [])
    if not lora_sources:
        return []
        
    lora_ids = vals.get('loras_ids', [])
    lora_scales = vals.get('loras_scales', [])
    
    for i in range(len(lora_sources)):
        scale = lora_scales[i] if i < len(lora_scales) else 1.0
        if scale is not None and scale != 0:
            name = None
            src = lora_sources[i] if i < len(lora_sources) else None
            id_val = lora_ids[i] if i < len(lora_ids) else None

            if src == "File" and id_val:
                name = id_val
            elif src in ["Civitai", "Custom URL"] and id_val:
                path, status_msg = get_lora_path(src, id_val, CIVITAI_API_KEY)
                if path is None:
                    raise gr.Error(f"LoRA '{id_val}' failed to download: {status_msg}")
                name = path
            
            if name:
                loras.append({"lora_name": name, "strength_model": scale, "strength_clip": scale})
    return loras

def process_embedding_inputs(vals):
    embeddings = []
    embedding_sources = vals.get('embeddings_sources', [])
    if not embedding_sources:
        return []
        
    embedding_ids = vals.get('embeddings_ids', [])
    
    for i in range(len(embedding_sources)):
        name = None
        src = embedding_sources[i] if i < len(embedding_sources) else None
        id_val = embedding_ids[i] if i < len(embedding_ids) else None

        if src == "File" and id_val:
            name = id_val
        elif src in ["Civitai", "Custom URL"] and id_val:
            path, status_msg = get_embedding_path(src, id_val, CIVITAI_API_KEY)
            if path is None:
                raise gr.Error(f"Embedding '{id_val}' failed to download: {status_msg}")
                name = path
        
        if name:
            embeddings.append(name)
            
    return embeddings

def process_controlnet_inputs(vals):
    controlnets = []
    cn_images = vals.get('controlnet_images', [])
    if not cn_images:
        return []

    cn_strengths = vals.get('controlnet_strengths', [])
    cn_filepaths = vals.get('controlnet_filepaths', [])

    for i in range(len(cn_images)):
        image_pil = cn_images[i]
        strength = cn_strengths[i] if i < len(cn_strengths) else 1.0
        cn_path = cn_filepaths[i] if i < len(cn_filepaths) else "None"

        if image_pil is not None and strength > 0 and cn_path and cn_path != "None":
            image_filename = save_temp_image_from_pil(image_pil, f"cn_{i}")
            controlnets.append({
                "image": image_filename,
                "strength": strength,
                "control_net_name": cn_path,
                "start_percent": 0.0,
                "end_percent": 1.0,
            })
    return controlnets

def process_ipadapter_inputs(vals):
    ipadapters = []
    ipa_images = vals.get('ipadapter_images', [])
    if not ipa_images:
        return []

    ipa_presets = vals.get('ipadapter_presets', [])
    ipa_weights = vals.get('ipadapter_weights', [])
    ipa_lora_strengths = vals.get('ipadapter_lora_strengths', [])
    
    ipadapter_presets_config = load_ipadapter_presets()
    faceid_presets_sd15 = ipadapter_presets_config.get("IPAdapter_FaceID_presets", {}).get("SD1.5", [])
    faceid_presets_sdxl = ipadapter_presets_config.get("IPAdapter_FaceID_presets", {}).get("SDXL", [])
    all_faceid_presets = faceid_presets_sd15 + faceid_presets_sdxl

    for i in range(len(ipa_images)):
        image_pil = ipa_images[i]
        preset = ipa_presets[i] if i < len(ipa_presets) else None
        weight = ipa_weights[i] if i < len(ipa_weights) else 1.0
        lora_strength = ipa_lora_strengths[i] if i < len(ipa_lora_strengths) else 0.6

        if image_pil is not None and weight > 0 and preset:
            image_filename = save_temp_image_from_pil(image_pil, f"ipa_{i}")
            loader_type = 'FaceID' if preset in all_faceid_presets else 'Unified'
            item_data = {
                "image": image_filename,
                "preset": preset,
                "weight": weight,
                "loader_type": loader_type
            }
            if loader_type == 'FaceID':
                item_data['lora_strength'] = lora_strength
            ipadapters.append(item_data)
    
    if ipadapters:
        final_preset = vals.get('ipadapter_final_preset')
        final_weight = vals.get('ipadapter_final_weight')
        final_embeds_scaling = vals.get('ipadapter_embeds_scaling')
        final_combine_method = vals.get('ipadapter_combine_method')
        model_type = vals.get('model_type_state')
        
        if final_preset and final_weight is not None and final_embeds_scaling:
            final_loader_type = 'FaceID' if final_preset in all_faceid_presets else 'Unified'
            
            final_settings = {
                'is_final_settings': True,
                'model_type': model_type,
                'final_preset': final_preset,
                'final_weight': final_weight,
                'final_embeds_scaling': final_embeds_scaling,
                'final_loader_type': final_loader_type,
                'final_combine_method': final_combine_method
            }
            if final_loader_type == 'FaceID':
                final_settings['final_lora_strength'] = vals.get('ipadapter_final_lora_strength', 0.6)
            
            ipadapters.append(final_settings)

    return ipadapters

def process_style_inputs(vals):
    styles = []
    style_images = vals.get('style_images', [])
    if not style_images:
        return []
        
    style_strengths = vals.get('style_strengths', [])
    
    for i in range(len(style_images)):
        image_pil = style_images[i]
        strength = style_strengths[i] if i < len(style_strengths) else 1.0

        if image_pil is not None and strength > 0:
            image_filename = save_temp_image_from_pil(image_pil, f"style_{i}")
            styles.append({
                "image": image_filename,
                "strength": strength,
            })
    return styles

def process_conditioning_inputs(vals):
    conditionings = []
    prompts = vals.get('conditioning_prompts', [])
    if not prompts:
        return []

    widths = vals.get('conditioning_widths', [])
    heights = vals.get('conditioning_heights', [])
    xs = vals.get('conditioning_xs', [])
    ys = vals.get('conditioning_ys', [])
    strengths = vals.get('conditioning_strengths', [])

    for i in range(len(prompts)):
        prompt_text = prompts[i]
        if prompt_text and prompt_text.strip():
            conditionings.append({
                "prompt": prompt_text,
                "width": widths[i] if i < len(widths) else 512,
                "height": heights[i] if i < len(heights) else 512,
                "x": xs[i] if i < len(xs) else 0,
                "y": ys[i] if i < len(ys) else 0,
                "strength": strengths[i] if i < len(strengths) else 1.0,
            })
    return conditionings