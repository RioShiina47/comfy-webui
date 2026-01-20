import gradio as gr
import os
from .config import CIVITAI_API_KEY
from .download_utils import get_lora_path, get_embedding_path
from .utils import save_temp_image
from .yaml_loader import load_and_merge_yaml

def process_lora_inputs(all_ui_values: dict, prefix: str):
    """
    Processes LoRA-related UI values from a given prefix.
    Handles file downloads for Civitai/URL sources and constructs the lora_chain list.
    """
    key = lambda name: f"{prefix}_{name}"
    
    loras = []
    lora_sources = all_ui_values.get(key('loras_sources'), [])
    if not lora_sources:
        return []
        
    lora_ids_txt = all_ui_values.get(key('loras_ids'), [])
    lora_ids_dd = all_ui_values.get(key('loras_file_dropdowns'), [])
    lora_scales = all_ui_values.get(key('loras_scales'), [])
    
    for i in range(len(lora_sources)):
        scale = lora_scales[i] if i < len(lora_scales) else 1.0
        if scale is not None and scale != 0:
            name = None
            src = lora_sources[i] if i < len(lora_sources) else None

            id_val = None
            if src == "File":
                id_val = lora_ids_dd[i] if i < len(lora_ids_dd) else None
            else:
                id_val = lora_ids_txt[i] if i < len(lora_ids_txt) else None

            if src == "Upload File" and id_val:
                name = id_val
            elif src == "File" and id_val:
                os_specific_subpath = id_val.replace("/", os.sep)
                name = os.path.join("file", os_specific_subpath)
            elif src in ["Civitai", "Custom URL"] and id_val:
                path, status_msg = get_lora_path(src, id_val, CIVITAI_API_KEY)
                if path is None:
                    raise gr.Error(f"LoRA '{id_val}' failed to download: {status_msg}")
                name = path
            
            if name:
                loras.append({"lora_name": name, "strength_model": scale, "strength_clip": scale})
    return loras


def process_embedding_inputs(all_ui_values: dict, prefix: str):
    key = lambda name: f"{prefix}_{name}"
    embeddings = []
    embedding_sources = all_ui_values.get(key('embeddings_sources'), [])
    if not embedding_sources:
        return []
        
    embedding_ids = all_ui_values.get(key('embeddings_ids'), [])
    
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


def process_controlnet_inputs(all_ui_values: dict, prefix: str):
    key = lambda name: f"{prefix}_{name}"
    controlnets = []
    cn_images = all_ui_values.get(key('controlnet_images'), [])
    if not cn_images:
        return []

    cn_strengths = all_ui_values.get(key('controlnet_strengths'), [])
    cn_filepaths = all_ui_values.get(key('controlnet_filepaths'), [])

    for i in range(len(cn_images)):
        image_pil = cn_images[i]
        strength = cn_strengths[i] if i < len(cn_strengths) else 1.0
        cn_path = cn_filepaths[i] if i < len(cn_filepaths) else "None"

        if image_pil is not None and strength > 0 and cn_path and cn_path != "None":
            image_filename = save_temp_image(image_pil)
            controlnets.append({
                "image": image_filename,
                "strength": strength,
                "control_net_name": cn_path,
                "start_percent": 0.0,
                "end_percent": 1.0,
            })
    return controlnets


def process_diffsynth_controlnet_inputs(all_ui_values: dict, prefix: str):
    key = lambda name: f"{prefix}_{name}"
    controlnets = []
    cn_images = all_ui_values.get(key('diffsynth_controlnet_images'), [])
    if not cn_images:
        return []

    cn_strengths = all_ui_values.get(key('diffsynth_controlnet_strengths'), [])
    cn_filepaths = all_ui_values.get(key('diffsynth_controlnet_filepaths'), [])

    for i in range(len(cn_images)):
        image_pil = cn_images[i]
        strength = cn_strengths[i] if i < len(cn_strengths) else 1.0
        cn_path = cn_filepaths[i] if i < len(cn_filepaths) else "None"

        if image_pil is not None and strength > 0 and cn_path and cn_path != "None":
            image_filename = save_temp_image(image_pil)
            controlnets.append({
                "image": image_filename,
                "strength": strength,
                "control_net_name": cn_path,
            })
    return controlnets


def process_ipadapter_inputs(all_ui_values: dict, prefix: str, ipadapter_presets_config: dict):
    key = lambda name: f"{prefix}_{name}"
    ipadapters = []
    ipa_images = all_ui_values.get(key('ipadapter_images'), [])
    if not ipadapters:
        return []

    final_preset = all_ui_values.get(key('ipadapter_final_preset'))
    ipa_weights = all_ui_values.get(key('ipadapter_weights'), [])
    ipa_lora_strengths = all_ui_values.get(key('ipadapter_lora_strengths'), [])
    
    faceid_presets_sd15 = ipadapter_presets_config.get("IPAdapter_FaceID_presets", {}).get("SD1.5", [])
    faceid_presets_sdxl = ipadapter_presets_config.get("IPAdapter_FaceID_presets", {}).get("SDXL", [])
    all_faceid_presets = faceid_presets_sd15 + faceid_presets_sdxl

    for i in range(len(ipa_images)):
        image_pil = ipa_images[i]
        preset = final_preset
        weight = ipa_weights[i] if i < len(ipa_weights) else 1.0
        lora_strength = ipa_lora_strengths[i] if i < len(ipa_lora_strengths) else 0.6

        if image_pil is not None and weight > 0 and preset:
            image_filename = save_temp_image(image_pil)
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
        final_weight = all_ui_values.get(key('ipadapter_final_weight'))
        final_embeds_scaling = all_ui_values.get(key('ipadapter_embeds_scaling'))
        final_combine_method = all_ui_values.get(key('ipadapter_combine_method'))
        model_type = all_ui_values.get(key('model_type_state'))
        
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
                final_settings['final_lora_strength'] = all_ui_values.get(key('ipadapter_final_lora_strength'), 0.6)
            
            ipadapters.append(final_settings)

    return ipadapters


def process_flux1_ipadapter_inputs(all_ui_values: dict, prefix: str):
    key = lambda name: f"{prefix}_{name}"
    ipadapters = []
    ipa_images = all_ui_values.get(key('flux1_ipadapter_images'), [])
    if not ipa_images:
        return []

    ipa_weights = all_ui_values.get(key('flux1_ipadapter_weights'), [])
    ipa_start_percents = all_ui_values.get(key('flux1_ipadapter_start_percents'), [])
    ipa_end_percents = all_ui_values.get(key('flux1_ipadapter_end_percents'), [])

    for i in range(len(ipa_images)):
        image_pil = ipa_images[i]
        weight = ipa_weights[i] if i < len(ipa_weights) else 0.6
        start_percent = ipa_start_percents[i] if i < len(ipa_start_percents) else 0.0
        end_percent = ipa_end_percents[i] if i < len(ipa_end_percents) else 0.6

        if image_pil is not None and weight > 0:
            image_filename = save_temp_image(image_pil)
            item_data = {
                "image": image_filename,
                "weight": weight,
                "start_percent": start_percent,
                "end_percent": end_percent,
            }
            ipadapters.append(item_data)
    
    return ipadapters


def process_sd3_ipadapter_inputs(all_ui_values: dict, prefix: str):
    key = lambda name: f"{prefix}_{name}"
    ipadapters = []
    ipa_images = all_ui_values.get(key('sd3_ipadapter_images'), [])
    if not ipa_images:
        return []

    ipa_weights = all_ui_values.get(key('sd3_ipadapter_weights'), [])
    ipa_start_percents = all_ui_values.get(key('sd3_ipadapter_start_percents'), [])
    ipa_end_percents = all_ui_values.get(key('sd3_ipadapter_end_percents'), [])

    for i in range(len(ipa_images)):
        image_pil = ipa_images[i]
        weight = ipa_weights[i] if i < len(ipa_weights) else 0.5
        start_percent = ipa_start_percents[i] if i < len(ipa_start_percents) else 0.0
        end_percent = ipa_end_percents[i] if i < len(ipa_end_percents) else 1.0

        if image_pil is not None and weight > 0:
            image_filename = save_temp_image(image_pil)
            item_data = {
                "image": image_filename,
                "weight": weight,
                "start_percent": start_percent,
                "end_percent": end_percent,
            }
            ipadapters.append(item_data)
    
    return ipadapters


def process_style_inputs(all_ui_values: dict, prefix: str):
    key = lambda name: f"{prefix}_{name}"
    styles = []
    style_images = all_ui_values.get(key('style_images'), [])
    if not style_images:
        return []
        
    style_strengths = all_ui_values.get(key('style_strengths'), [])
    
    for i in range(len(style_images)):
        image_pil = style_images[i]
        strength = style_strengths[i] if i < len(style_strengths) else 1.0

        if image_pil is not None and strength > 0:
            image_filename = save_temp_image(image_pil)
            styles.append({
                "image": image_filename,
                "strength": strength,
            })
    return styles


def process_conditioning_inputs(all_ui_values: dict, prefix: str):
    key = lambda name: f"{prefix}_{name}"
    conditionings = []
    prompts = all_ui_values.get(key('conditioning_prompts'), [])
    if not prompts:
        return []

    widths = all_ui_values.get(key('conditioning_widths'), [])
    heights = all_ui_values.get(key('conditioning_heights'), [])
    xs = all_ui_values.get(key('conditioning_xs'), [])
    ys = all_ui_values.get(key('conditioning_ys'), [])
    strengths = all_ui_values.get(key('conditioning_strengths'), [])

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

def process_reference_latent_inputs(all_ui_values: dict, prefix: str):
    key = lambda name: f"{prefix}_{name}"
    references = []
    ref_images = all_ui_values.get(key('reference_latent_images'), [])
    if not ref_images:
        return []
    
    for image_pil in ref_images:
        if image_pil is not None:
            image_filename = save_temp_image(image_pil)
            references.append(image_filename)
            
    return references