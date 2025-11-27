import gradio as gr
import os
import shutil
from core.config import LORA_DIR, EMBEDDING_DIR
from .utils import get_model_type, get_model_generation_defaults, parse_generation_parameters_for_ui
from .config_loader import load_model_config, load_model_defaults, load_controlnet_models, load_ipadapter_presets, load_constants_config, load_features_config

constants = load_constants_config()

def on_lora_upload(file_obj):
    if file_obj is None: return gr.update(), gr.update(), None
    
    upload_subdir = "file"
    lora_upload_dir = os.path.join(LORA_DIR, upload_subdir)
    os.makedirs(lora_upload_dir, exist_ok=True)
    
    basename = os.path.basename(file_obj.name)
    new_path = os.path.join(lora_upload_dir, basename)
    shutil.copy(file_obj.name, new_path)
    
    relative_path = os.path.join(upload_subdir, basename)
    
    return relative_path, "File", relative_path

def on_embedding_upload(file_obj):
    if file_obj is None: return gr.update(), gr.update(), None
    
    upload_subdir = "file"
    embedding_upload_dir = os.path.join(EMBEDDING_DIR, upload_subdir)
    os.makedirs(embedding_upload_dir, exist_ok=True)
    
    basename = os.path.basename(file_obj.name)
    new_path = os.path.join(embedding_upload_dir, basename)
    shutil.copy(file_obj.name, new_path)
    
    relative_path = os.path.join(upload_subdir, basename)
    
    return relative_path, "File", relative_path

def update_model_list(architecture_filter: str, category_filter: str):
    model_config = load_model_config()
    checkpoints = model_config.get("Checkpoints", {})

    sdxl_models_data = checkpoints.get("SDXL", {}).get("models", [])
    
    if category_filter != "ALL":
        choices = [m['display_name'] for m in sdxl_models_data if m.get("category") == category_filter]
        default_value = choices[0] if choices else None
        return gr.update(choices=choices, value=default_value)

    sdxl_models_data = checkpoints.get("SDXL", {}).get("models", [])
    sd35_models_data = checkpoints.get("SD3.5", {}).get("models", [])
    sd15_models_data = checkpoints.get("SD1.5", {}).get("models", [])
    flux_models_data = checkpoints.get("FLUX", {}).get("models", [])
    qwen_models_data = checkpoints.get("Qwen-Image", {}).get("models", [])
    hidream_models_data = checkpoints.get("HiDream", {}).get("models", [])
    hunyuan_models_data = checkpoints.get("HunyuanImage", {}).get("models", [])
    chroma_radiance_models_data = checkpoints.get("Chroma1-Radiance", {}).get("models", [])
    chroma1_models_data = checkpoints.get("Chroma1", {}).get("models", [])
    omnigen2_models_data = checkpoints.get("OmniGen2", {}).get("models", [])
    neta_lumina_models_data = checkpoints.get("Neta-Lumina", {}).get("models", [])
    z_image_models_data = checkpoints.get("Z-Image", {}).get("models", [])
    
    sdxl_models_display = [m['display_name'] for m in sdxl_models_data]
    sd35_models_display = [m['display_name'] for m in sd35_models_data]
    sd15_models_display = [m['display_name'] for m in sd15_models_data]
    flux_models_display = [m['display_name'] for m in flux_models_data]
    qwen_models_display = [m['display_name'] for m in qwen_models_data]
    hidream_models_display = [m['display_name'] for m in hidream_models_data]
    hunyuan_models_display = [m['display_name'] for m in hunyuan_models_data]
    chroma_radiance_models_display = [m['display_name'] for m in chroma_radiance_models_data]
    chroma1_models_display = [m['display_name'] for m in chroma1_models_data]
    omnigen2_models_display = [m['display_name'] for m in omnigen2_models_data]
    neta_lumina_models_display = [m['display_name'] for m in neta_lumina_models_data]
    z_image_models_display = [m['display_name'] for m in z_image_models_data]

    choices = []
    if architecture_filter == "ALL":
        choices = z_image_models_display + sdxl_models_display + sd35_models_display + flux_models_display + omnigen2_models_display + neta_lumina_models_display + hunyuan_models_display + hidream_models_display + qwen_models_display + chroma1_models_display + chroma_radiance_models_display + sd15_models_display
    elif architecture_filter == "Z-Image":
        choices = z_image_models_display
    elif architecture_filter == "SDXL":
        choices = sdxl_models_display
    elif architecture_filter == "SD3.5":
        choices = sd35_models_display
    elif architecture_filter == "FLUX":
        choices = flux_models_display
    elif architecture_filter == "OmniGen2":
        choices = omnigen2_models_display
    elif architecture_filter == "Neta-Lumina":
        choices = neta_lumina_models_display
    elif architecture_filter == "HunyuanImage":
        choices = hunyuan_models_display
    elif architecture_filter == "HiDream":
        choices = hidream_models_display
    elif architecture_filter == "Qwen-Image":
        choices = qwen_models_display
    elif architecture_filter == "Chroma1":
        choices = chroma1_models_display
    elif architecture_filter == "Chroma1-Radiance":
        choices = chroma_radiance_models_display
    elif architecture_filter == "SD1.5":
        choices = sd15_models_display
    
    default_value = choices[0] if choices else None
    return gr.update(choices=choices, value=default_value)

def register_shared_events(components, prefix, sdxl_gallery_height, demo):
    key = lambda name: f"{prefix}_{name}"
    model_filter = components[key("model_filter")]
    sdxl_category_filter = components.get(key('sdxl_category_filter'))
    model_dropdown = components[key("model_name")]
    model_type_state = components[key('model_type_state')]
    clip_skip_slider = components.get(key('clip_skip'))
    guidance_slider = components.get(key('guidance'))
    aspect_ratio_dropdown = components.get(key('aspect_ratio_dropdown'))
    width_num = components.get(key('width'))
    height_num = components.get(key('height'))
    steps_slider = components.get(key('steps'))
    cfg_slider = components.get(key('cfg'))
    sampler_dropdown = components.get(key('sampler_name'))
    scheduler_dropdown = components.get(key('scheduler'))
    lora_accordion = components.get(key('lora_accordion'))
    gallery_component = components.get(key('output_gallery'))
    controlnet_accordion = components.get(key('controlnet_accordion'))
    controlnet_series_list = components.get(key('controlnet_series'))
    controlnet_types_list = components.get(key('controlnet_types'))
    controlnet_filepaths_list = components.get(key('controlnet_filepaths'))
    controlnet_images_list = components.get(key('controlnet_images'))
    ipadapter_accordion = components.get(key('ipadapter_accordion'))
    ipadapter_presets_list = components.get(key('ipadapter_presets'))
    ipadapter_final_preset = components.get(key('ipadapter_final_preset'))
    ipadapter_lora_strengths_list = components.get(key('ipadapter_lora_strengths'))
    ipadapter_final_lora_strength_slider = components.get(key('ipadapter_final_lora_strength'))
    embedding_accordion = components.get(key('embedding_accordion'))
    style_accordion = components.get(key('style_accordion'))
    conditioning_accordion = components.get(key('conditioning_accordion'))


    def on_architecture_filter_change(arch_filter):
        sdxl_filter_visibility = arch_filter in ["SDXL", "ALL"]
        updated_model_list = update_model_list(arch_filter, "ALL")
        return gr.update(visible=sdxl_filter_visibility), gr.update(value="ALL"), updated_model_list

    if sdxl_category_filter:
        model_filter.change(
            fn=on_architecture_filter_change,
            inputs=[model_filter],
            outputs=[sdxl_category_filter, sdxl_category_filter, model_dropdown],
            show_progress=False,
            show_api=False
        )
        sdxl_category_filter.change(
            fn=update_model_list,
            inputs=[model_filter, sdxl_category_filter],
            outputs=[model_dropdown],
            show_progress=False,
            show_api=False
        )
        demo.load(
            fn=on_architecture_filter_change,
            inputs=[model_filter],
            outputs=[sdxl_category_filter, sdxl_category_filter, model_dropdown],
            show_api=False
        )
    else:
        model_filter.change(
            fn=lambda arch: update_model_list(arch, "ALL"),
            inputs=[model_filter],
            outputs=[model_dropdown],
            show_progress=False,
            show_api=False
        )
        demo.load(
            fn=lambda arch: update_model_list(arch, "ALL"),
            inputs=[model_filter],
            outputs=[model_dropdown],
            show_api=False
        )

    def on_model_change(selected_model_name):
        updates = {}
        model_config = load_model_config()
        model_defaults = load_model_defaults()
        ipadapter_presets_config = load_ipadapter_presets()
        features_config = load_features_config()

        if not selected_model_name:
            model_type = "sdxl"
        else:
            model_type = get_model_type(selected_model_name, model_config)

        updates[model_type_state] = model_type
        
        arch_features = features_config.get(model_type, features_config.get('default', {}))
        enabled_chains = arch_features.get('enabled_chains', [])

        chain_map = {
            'lora': lora_accordion,
            'controlnet': controlnet_accordion,
            'ipadapter': ipadapter_accordion,
            'embedding': embedding_accordion,
            'style': style_accordion,
            'conditioning': conditioning_accordion
        }

        for chain_key, accordion_component in chain_map.items():
            if accordion_component:
                is_enabled = chain_key in enabled_chains
                updates[accordion_component] = gr.update(visible=is_enabled)
        
        is_flux = (model_type == 'flux')
        is_sd15 = (model_type == 'sd15')
        
        if guidance_slider:
            updates[guidance_slider] = gr.update(visible=is_flux)
        
        if clip_skip_slider:
            updates[clip_skip_slider] = gr.update(visible=is_sd15, maximum=2 if is_sd15 else 4)

        defaults = get_model_generation_defaults(selected_model_name, model_type, model_defaults)
        
        cn_config = load_controlnet_models()
        arch_key_map = {"sdxl": "SDXL", "sd35": "SDXL", "sd15": "SD1.5", "flux": "FLUX", "qwen-image": "Qwen-Image", "hunyuanimage": "HunyuanImage", "chroma1": "Chroma1", "chroma1-radiance": "Chroma1-Radiance", "omnigen2": "OmniGen2", "neta-lumina": "Neta-Lumina", "z-image": "Z-Image"}
        arch_key = arch_key_map.get(model_type, "SDXL")
        
        if controlnet_accordion and 'controlnet' in enabled_chains:
            controlnet_visible = arch_key in cn_config
            updates[controlnet_accordion] = gr.update(visible=controlnet_visible)
            
            type_choices = []
            if controlnet_visible:
                all_types = [t for model in cn_config[arch_key] for t in model.get("Type", [])]
                type_choices = sorted(list(set(all_types)))
            
            default_type = type_choices[0] if type_choices else None
            type_update = gr.update(choices=type_choices, value=default_type)

            series_choices = []
            if controlnet_visible and default_type:
                series_choices = sorted(list(set(
                    model.get("Series", "Default")
                    for model in cn_config[arch_key]
                    if default_type in model.get("Type", [])
                )))
            
            default_series = series_choices[0] if series_choices else None
            series_update = gr.update(choices=series_choices, value=default_series)

            filepath = "None"
            if controlnet_visible and default_type and default_series:
                for model in cn_config.get(arch_key, []):
                    if model.get("Series") == default_series and default_type in model.get("Type", []):
                        filepath = model.get("Filepath")
                        break
            
            if controlnet_types_list:
                for type_dd in controlnet_types_list: updates[type_dd] = type_update
            if controlnet_series_list:
                for series_dd in controlnet_series_list: updates[series_dd] = series_update
            if controlnet_filepaths_list:
                for filepath_state in controlnet_filepaths_list: updates[filepath_state] = filepath
        
        if ipadapter_accordion and 'ipadapter' in enabled_chains:
            ipadapter_visible = model_type in ["sdxl", "sd15", "sd35"]
            updates[ipadapter_accordion] = gr.update(visible=ipadapter_visible)
            
            if ipadapter_visible:
                arch_key_ipa = "SDXL" if model_type in ["sdxl", "sd35"] else "SD1.5"
                unified_presets = ipadapter_presets_config.get("IPAdapter_presets", {}).get(arch_key_ipa, [])
                faceid_presets = ipadapter_presets_config.get("IPAdapter_FaceID_presets", {}).get(arch_key_ipa, [])
                all_presets = unified_presets + faceid_presets
                default_preset = unified_presets[0] if unified_presets else (faceid_presets[0] if faceid_presets else None)
                
                if ipadapter_final_preset:
                    updates[ipadapter_final_preset] = gr.update(choices=all_presets, value=default_preset)

                is_faceid = default_preset in faceid_presets
                if ipadapter_final_lora_strength_slider:
                    updates[ipadapter_final_lora_strength_slider] = gr.update(visible=is_faceid)
                
                if ipadapter_lora_strengths_list:
                    for lora_strength_slider in ipadapter_lora_strengths_list:
                        updates[lora_strength_slider] = gr.update(visible=is_faceid)

        resolutions = constants.get('RESOLUTION_MAP', {}).get(model_type, constants.get('RESOLUTION_MAP', {}).get("sdxl", {}))
        
        if all([aspect_ratio_dropdown, width_num, height_num]):
            default_ratio = list(resolutions.keys())[0]
            new_w, new_h = resolutions[default_ratio]
            updates[aspect_ratio_dropdown] = gr.update(choices=list(resolutions.keys()), value=default_ratio)
            updates[width_num] = gr.update(value=new_w)
            updates[height_num] = gr.update(value=new_h)

        if steps_slider: updates[steps_slider] = gr.update(value=defaults.get('steps'))
        if cfg_slider: updates[cfg_slider] = gr.update(value=defaults.get('cfg'))
        if sampler_dropdown: updates[sampler_dropdown] = gr.update(value=defaults.get('sampler_name'))
        if scheduler_dropdown: updates[scheduler_dropdown] = gr.update(value=defaults.get('scheduler'))
        
        positive_prompt_textbox = components.get(key('positive_prompt'))
        negative_prompt_textbox = components.get(key('negative_prompt'))
        if positive_prompt_textbox:
            updates[positive_prompt_textbox] = gr.update(value=defaults.get('positive_prompt'))
        if negative_prompt_textbox:
            updates[negative_prompt_textbox] = gr.update(value=defaults.get('negative_prompt'))

        if gallery_component: updates[gallery_component] = gr.update(height=sdxl_gallery_height)
        
        return updates

    on_model_change_outputs = {
        "state": model_type_state, "clip_skip": clip_skip_slider, "guidance": guidance_slider, "style_accordion": style_accordion, "aspect": aspect_ratio_dropdown,
        "w": width_num, "h": height_num, "steps": steps_slider, "cfg": cfg_slider,
        "sampler": sampler_dropdown, "scheduler": scheduler_dropdown, "lora": lora_accordion,
        "embedding": embedding_accordion, "gallery": gallery_component, "cn_accordion": controlnet_accordion,
        "ipa_accordion": ipadapter_accordion, "ipa_final_preset": ipadapter_final_preset,
        "ipa_final_lora_slider": ipadapter_final_lora_strength_slider,
        "conditioning_accordion": conditioning_accordion
    }
    
    positive_prompt_textbox = components.get(key('positive_prompt'))
    negative_prompt_textbox = components.get(key('negative_prompt'))
    if positive_prompt_textbox: on_model_change_outputs['positive_prompt'] = positive_prompt_textbox
    if negative_prompt_textbox: on_model_change_outputs['negative_prompt'] = negative_prompt_textbox
    
    if controlnet_types_list:
        for i, cn_type in enumerate(controlnet_types_list): on_model_change_outputs[f'cn_type_{i}'] = cn_type
    if controlnet_series_list:
        for i, cn_series in enumerate(controlnet_series_list): on_model_change_outputs[f'cn_series_{i}'] = cn_series
    if controlnet_filepaths_list:
        for i, cn_filepath in enumerate(controlnet_filepaths_list): on_model_change_outputs[f'cn_filepath_{i}'] = cn_filepath
    if ipadapter_presets_list:
        for i, ipa_preset in enumerate(ipadapter_presets_list): on_model_change_outputs[f'ipa_preset_{i}'] = ipa_preset
    if ipadapter_lora_strengths_list:
        for i, lora_strength in enumerate(ipadapter_lora_strengths_list): on_model_change_outputs[f'ipa_lora_strength_{i}'] = lora_strength

    filtered_outputs = [v for k, v in on_model_change_outputs.items() if v is not None]

    if filtered_outputs:
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=filtered_outputs,
            show_api=False
        )

    def on_final_preset_change(preset_value):
        ipadapter_presets_config = load_ipadapter_presets()
        faceid_presets_sd15 = ipadapter_presets_config.get("IPAdapter_FaceID_presets", {}).get("SD1.5", [])
        faceid_presets_sdxl = ipadapter_presets_config.get("IPAdapter_FaceID_presets", {}).get("SDXL", [])
        all_faceid_presets = faceid_presets_sd15 + faceid_presets_sdxl
        is_faceid = preset_value in all_faceid_presets
        
        updates = [gr.update(visible=is_faceid)]
        for _ in ipadapter_lora_strengths_list:
            updates.append(gr.update(visible=is_faceid))
        
        return tuple(updates)

    if ipadapter_final_preset and ipadapter_final_lora_strength_slider and ipadapter_lora_strengths_list:
         outputs_to_update = [ipadapter_final_lora_strength_slider] + ipadapter_lora_strengths_list
         ipadapter_final_preset.change(
            fn=on_final_preset_change,
            inputs=[ipadapter_final_preset],
            outputs=outputs_to_update,
            show_progress=False,
            show_api=False
        )

    def on_cn_type_change(selected_type, model_type):
        cn_config = load_controlnet_models()
        arch_key = {"sdxl": "SDXL", "sd35": "SDXL", "sd15": "SD1.5", "flux": "FLUX", "qwen-image": "Qwen-Image", "hunyuanimage": "HunyuanImage", "chroma1": "Chroma1", "chroma1-radiance": "Chroma1-Radiance", "omnigen2": "OmniGen2", "neta-lumina": "Neta-Lumina", "z-image": "Z-Image"}.get(model_type, "SDXL")
        
        series_choices = []
        if arch_key in cn_config and selected_type:
            series_choices = sorted(list(set(
                model.get("Series", "Default")
                for model in cn_config[arch_key]
                if selected_type in model.get("Type", [])
            )))

        default_series = series_choices[0] if series_choices else None
        
        filepath = "None"
        if default_series:
            for model in cn_config.get(arch_key, []):
                if model.get("Series") == default_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
                    
        return gr.update(choices=series_choices, value=default_series), filepath

    def on_cn_series_change(selected_series, selected_type, model_type):
        cn_config = load_controlnet_models()
        arch_key = {"sdxl": "SDXL", "sd35": "SDXL", "sd15": "SD1.5", "flux": "FLUX", "qwen-image": "Qwen-Image", "hunyuanimage": "HunyuanImage", "chroma1": "Chroma1", "chroma1-radiance": "Chroma1-Radiance", "omnigen2": "OmniGen2", "neta-lumina": "Neta-Lumina", "z-image": "Z-Image"}.get(model_type, "SDXL")
        
        filepath = "None"
        if arch_key in cn_config and selected_series and selected_type:
            for model in cn_config[arch_key]:
                if model.get("Series") == selected_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
        return filepath

    if controlnet_series_list and controlnet_types_list and controlnet_filepaths_list:
        for i in range(constants.get('MAX_CONTROLNETS', 5)):
            controlnet_types_list[i].change(
                fn=on_cn_type_change,
                inputs=[controlnet_types_list[i], model_type_state],
                outputs=[controlnet_series_list[i], controlnet_filepaths_list[i]],
                show_progress=False,
                show_api=False
            )
            controlnet_series_list[i].change(
                fn=on_cn_series_change,
                inputs=[controlnet_series_list[i], controlnet_types_list[i], model_type_state],
                outputs=[controlnet_filepaths_list[i]],
                show_progress=False,
                show_api=False
            )

    if controlnet_accordion and controlnet_images_list:
        def on_accordion_expand(*images):
            return [gr.update() for _ in images]
        
        controlnet_accordion.expand(
            fn=on_accordion_expand,
            inputs=controlnet_images_list,
            outputs=controlnet_images_list,
            show_progress=False,
            show_api=False
        )

    if all([aspect_ratio_dropdown, width_num, height_num]):
        def on_aspect_ratio_change(ratio_key, model_type):
            resolution_map = constants.get('RESOLUTION_MAP', {})
            res_map = resolution_map.get(model_type, resolution_map.get("sdxl", {}))
            w, h = res_map.get(ratio_key, (1024, 1024))
            return w, h
        aspect_ratio_dropdown.change(fn=on_aspect_ratio_change, inputs=[aspect_ratio_dropdown, model_type_state], outputs=[width_num, height_num], show_api=False)

        if f"{prefix}_input_image" in components:
            def on_image_upload(image_pil, model_type):
                if image_pil:
                    resolution_map = constants.get('RESOLUTION_MAP', {})
                    res_map = resolution_map.get(model_type, resolution_map.get("sdxl", {}))
                    w, h = image_pil.size
                    matched_ratio = next((k for k, v in res_map.items() if v == [w, h]), "Custom")
                    return w, h, matched_ratio
                return gr.update(), gr.update(), gr.update()
            image_input = components[f"{prefix}_input_image"]
            image_input.upload(fn=on_image_upload, inputs=[image_input, model_type_state], outputs=[width_num, height_num, aspect_ratio_dropdown], show_api=False)
            image_input.change(fn=on_image_upload, inputs=[image_input, model_type_state], outputs=[width_num, height_num, aspect_ratio_dropdown], show_api=False)

    if f'{prefix}_lora_count_state' in components:
        lora_count = components[f'{prefix}_lora_count_state']
        add_lora_btn = components[f'{prefix}_add_lora_button']
        del_lora_btn = components[f'{prefix}_delete_lora_button']
        lora_rows = components[f'{prefix}_lora_rows']
        lora_ids = components[f'{prefix}_loras_ids']
        lora_scales = components[f'{prefix}_loras_scales']
        def add_lora_row(count):
            count += 1
            return (count, gr.update(visible=count < constants.get('MAX_LORAS', 5)), gr.update(visible=count > 1)) + tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_LORAS', 5)))
        def delete_lora_row(count):
            count -= 1
            id_updates, scale_updates = [gr.update()] * constants.get('MAX_LORAS', 5), [gr.update()] * constants.get('MAX_LORAS', 5)
            id_updates[count], scale_updates[count] = "", 1.0
            row_updates = tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_LORAS', 5)))
            return (count, gr.update(visible=True), gr.update(visible=count > 1)) + row_updates + tuple(id_updates) + tuple(scale_updates)
        add_lora_outputs = [lora_count, add_lora_btn, del_lora_btn] + lora_rows
        del_lora_outputs = [lora_count, add_lora_btn, del_lora_btn] + lora_rows + lora_ids + lora_scales
        add_lora_btn.click(fn=add_lora_row, inputs=[lora_count], outputs=add_lora_outputs, show_progress=False, show_api=False)
        del_lora_btn.click(fn=delete_lora_row, inputs=[lora_count], outputs=del_lora_outputs, show_progress=False, show_api=False)

    if f'{prefix}_embedding_count_state' in components:
        embedding_count = components[f'{prefix}_embedding_count_state']
        add_embedding_btn = components[f'{prefix}_add_embedding_button']
        del_embedding_btn = components[f'{prefix}_delete_embedding_button']
        embedding_rows = components[f'{prefix}_embedding_rows']
        
        def add_embedding_row(count):
            count += 1
            return (count, gr.update(visible=count < constants.get('MAX_EMBEDDINGS', 5)), gr.update(visible=count > 1)) + tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_EMBEDDINGS', 5)))
        
        def delete_embedding_row(count):
            count -= 1
            row_updates = tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_EMBEDDINGS', 5)))
            return (count, gr.update(visible=True), gr.update(visible=count > 1)) + row_updates
        
        add_embedding_outputs = [embedding_count, add_embedding_btn, del_embedding_btn] + embedding_rows
        del_embedding_outputs = [embedding_count, add_embedding_btn, del_embedding_btn] + embedding_rows
        
        add_embedding_btn.click(fn=add_embedding_row, inputs=[embedding_count], outputs=add_embedding_outputs, show_progress=False, show_api=False)
        del_embedding_btn.click(fn=delete_embedding_row, inputs=[embedding_count], outputs=del_embedding_outputs, show_progress=False, show_api=False)
        
    if f'{prefix}_controlnet_count_state' in components:
        cn_count = components[f'{prefix}_controlnet_count_state']
        add_cn_btn = components[f'{prefix}_add_controlnet_button']
        del_cn_btn = components[f'{prefix}_delete_controlnet_button']
        cn_rows = components[f'{prefix}_controlnet_rows']
        cn_images = components[f'{prefix}_controlnet_images']
        cn_strengths = components[f'{prefix}_controlnet_strengths']

        def add_cn_row(count):
            count += 1
            return (count, gr.update(visible=count < constants.get('MAX_CONTROLNETS', 5)), gr.update(visible=count > 1)) + tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_CONTROLNETS', 5)))
        
        def delete_cn_row(count):
            count -= 1
            image_updates = [gr.update()] * constants.get('MAX_CONTROLNETS', 5)
            strength_updates = [gr.update()] * constants.get('MAX_CONTROLNETS', 5)
            image_updates[count] = None 
            strength_updates[count] = 1.0
            row_updates = tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_CONTROLNETS', 5)))
            return (count, gr.update(visible=True), gr.update(visible=count > 1)) + row_updates + tuple(image_updates) + tuple(strength_updates)

        add_cn_outputs = [cn_count, add_cn_btn, del_cn_btn] + cn_rows
        del_cn_outputs = [cn_count, add_cn_btn, del_cn_btn] + cn_rows + cn_images + cn_strengths
        
        add_cn_btn.click(fn=add_cn_row, inputs=[cn_count], outputs=add_cn_outputs, show_progress=False, show_api=False)
        del_cn_btn.click(fn=delete_cn_row, inputs=[cn_count], outputs=del_cn_outputs, show_progress=False, show_api=False)

    if f'{prefix}_ipadapter_count_state' in components:
        ipa_count = components[f'{prefix}_ipadapter_count_state']
        add_ipa_btn = components[f'{prefix}_add_ipadapter_button']
        del_ipa_btn = components[f'{prefix}_delete_ipadapter_button']
        ipa_rows = components[f'{prefix}_ipadapter_rows']
        ipa_images = components[f'{prefix}_ipadapter_images']
        ipa_weights = components[f'{prefix}_ipadapter_weights']
        
        def add_ipa_row(count):
            count += 1
            return (count, gr.update(visible=count < constants.get('MAX_IPADAPTERS', 5)), gr.update(visible=count > 1)) + tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_IPADAPTERS', 5)))

        def delete_ipa_row(count):
            count -= 1
            image_updates = [gr.update()] * constants.get('MAX_IPADAPTERS', 5)
            weight_updates = [gr.update()] * constants.get('MAX_IPADAPTERS', 5)
            image_updates[count] = None 
            weight_updates[count] = 1.0
            row_updates = tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_IPADAPTERS', 5)))
            return (count, gr.update(visible=True), gr.update(visible=count > 1)) + row_updates + tuple(image_updates) + tuple(weight_updates)

        add_ipa_outputs = [ipa_count, add_ipa_btn, del_ipa_btn] + ipa_rows
        del_ipa_outputs = [ipa_count, add_ipa_btn, del_ipa_btn] + ipa_rows + ipa_images + ipa_weights
        
        add_ipa_btn.click(fn=add_ipa_row, inputs=[ipa_count], outputs=add_ipa_outputs, show_progress=False, show_api=False)
        del_ipa_btn.click(fn=delete_ipa_row, inputs=[ipa_count], outputs=del_ipa_outputs, show_progress=False, show_api=False)

    if f'{prefix}_style_count_state' in components:
        style_count = components[f'{prefix}_style_count_state']
        add_style_btn = components[f'{prefix}_add_style_button']
        del_style_btn = components[f'{prefix}_delete_style_button']
        style_rows = components[f'{prefix}_style_rows']
        style_images = components[f'{prefix}_style_images']
        style_strengths = components[f'{prefix}_style_strengths']
        
        def add_style_row(count):
            count += 1
            return (count, gr.update(visible=count < constants.get('MAX_STYLES', 5)), gr.update(visible=count > 1)) + tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_STYLES', 5)))

        def delete_style_row(count):
            count -= 1
            image_updates = [gr.update()] * constants.get('MAX_STYLES', 5)
            strength_updates = [gr.update()] * constants.get('MAX_STYLES', 5)
            image_updates[count] = None
            strength_updates[count] = 1.0
            row_updates = tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_STYLES', 5)))
            return (count, gr.update(visible=True), gr.update(visible=count > 1)) + row_updates + tuple(image_updates) + tuple(strength_updates)

        add_style_outputs = [style_count, add_style_btn, del_style_btn] + style_rows
        del_style_outputs = [style_count, add_style_btn, del_style_btn] + style_rows + style_images + style_strengths
        
        add_style_btn.click(fn=add_style_row, inputs=[style_count], outputs=add_style_outputs, show_progress=False, show_api=False)
        del_style_btn.click(fn=delete_style_row, inputs=[style_count], outputs=del_style_outputs, show_progress=False, show_api=False)

    if f'{prefix}_conditioning_count_state' in components:
        cond_count = components[f'{prefix}_conditioning_count_state']
        add_cond_btn = components[f'{prefix}_add_conditioning_button']
        del_cond_btn = components[f'{prefix}_delete_conditioning_button']
        cond_rows = components[f'{prefix}_conditioning_rows']
        cond_prompts = components[f'{prefix}_conditioning_prompts']
        cond_widths = components[f'{prefix}_conditioning_widths']
        cond_heights = components[f'{prefix}_conditioning_heights']
        
        def add_cond_row(count, current_w, current_h):
            count += 1
            vis_updates = tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_CONDITIONINGS', 10)))
            width_updates = [gr.update()] * constants.get('MAX_CONDITIONINGS', 10)
            height_updates = [gr.update()] * constants.get('MAX_CONDITIONINGS', 10)
            if count > 0:
                width_updates[count-1] = gr.update(value=current_w)
                height_updates[count-1] = gr.update(value=current_h)
            
            return (count, gr.update(visible=count < constants.get('MAX_CONDITIONINGS', 10)), gr.update(visible=count > 1)) + vis_updates + tuple(width_updates) + tuple(height_updates)

        def delete_cond_row(count):
            count -= 1
            row_updates = tuple(gr.update(visible=i < count) for i in range(constants.get('MAX_CONDITIONINGS', 10)))
            prompt_updates = [gr.update()] * constants.get('MAX_CONDITIONINGS', 10)
            if count >= 0:
                prompt_updates[count] = gr.update(value="")
            
            return (count, gr.update(visible=True), gr.update(visible=count > 1)) + row_updates + tuple(prompt_updates)

        add_cond_outputs = [cond_count, add_cond_btn, del_cond_btn] + cond_rows + cond_widths + cond_heights
        del_cond_outputs = [cond_count, add_cond_btn, del_cond_btn] + cond_rows + cond_prompts

        click_inputs = [cond_count]
        if width_num: click_inputs.append(width_num)
        else: click_inputs.append(gr.State(value=512))
        if height_num: click_inputs.append(height_num)
        else: click_inputs.append(gr.State(value=512))

        add_cond_btn.click(
            fn=add_cond_row, 
            inputs=click_inputs, 
            outputs=add_cond_outputs, 
            show_progress=False,
            show_api=False
        )
        del_cond_btn.click(
            fn=delete_cond_row, 
            inputs=[cond_count], 
            outputs=del_cond_outputs, 
            show_progress=False,
            show_api=False
        )


    parse_button = components[f"{prefix}_parse_prompt_button"]
    positive_prompt = components[f"{prefix}_positive_prompt"]

    output_map = {
        'positive_prompt': positive_prompt, 'negative_prompt': components[f"{prefix}_negative_prompt"],
        'model_name': model_dropdown, 'seed': components.get(f"{prefix}_seed"),
        'steps': components.get(f"{prefix}_steps"), 'cfg': components.get(f"{prefix}_cfg"),
        'sampler_name': components.get(f"{prefix}_sampler_name"), 'scheduler': components.get(f"{prefix}_scheduler"),
        'clip_skip': components.get(f"{prefix}_clip_skip"), 'width': components.get(f"{prefix}_width"),
        'height': components.get(f"{prefix}_height")
    }

    output_keys = [
        'positive_prompt', 'negative_prompt', 'model_name', 'seed', 'steps', 
        'cfg', 'sampler_name', 'scheduler', 'clip_skip', 'width', 'height'
    ]

    final_outputs = [output_map[key] for key in output_keys if output_map[key] is not None]

    def on_parse_prompt_wrapper(prompt_text):
        parsed_data = parse_generation_parameters_for_ui(prompt_text)
        return_values = []
        for key in output_keys:
            if output_map[key] is not None:
                return_values.append(parsed_data.get(key, gr.update()))
        return tuple(return_values)

    parse_button.click(
        fn=on_parse_prompt_wrapper,
        inputs=[positive_prompt],
        outputs=final_outputs,
        show_api=False
    )