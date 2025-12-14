import gradio as gr
import os
import shutil
from core.config import LORA_DIR, EMBEDDING_DIR
from core.shared_ui import register_ui_chain_events
from .utils import get_model_type, get_model_generation_defaults, parse_generation_parameters_for_ui
from .config_loader import load_model_config, load_model_defaults, load_controlnet_models, load_diffsynth_controlnet_models, load_ipadapter_presets, load_constants_config, load_features_config, load_architectures_config

constants = load_constants_config()

def update_model_list(architecture_filter: str, category_filter: str):
    model_config = load_model_config()
    checkpoints = model_config.get("Checkpoints", {})
    arch_config = load_architectures_config()
    ordered_architectures = arch_config.get("architecture_order", [])

    if category_filter != "ALL" and architecture_filter == "SDXL":
        sdxl_models_data = checkpoints.get("SDXL", {}).get("models", [])
        choices = [m['display_name'] for m in sdxl_models_data if m.get("category") == category_filter]
        default_value = choices[0] if choices else None
        return gr.update(choices=choices, value=default_value)

    choices = []
    architectures_to_load = ordered_architectures if architecture_filter == "ALL" else [architecture_filter]

    for arch_name in architectures_to_load:
        if arch_name in checkpoints:
            models_data = checkpoints[arch_name].get("models", [])
            choices.extend([m['display_name'] for m in models_data])
    
    default_value = choices[0] if choices else None
    return gr.update(choices=choices, value=default_value)

def get_controlnet_key_for_model_type(model_type):
    """Helper function to find the controlnet_key from model_architectures.yaml based on model_type."""
    arch_config = load_architectures_config().get("architectures", {})
    for arch_name, details in arch_config.items():
        if details.get("model_type") == model_type:
            return details.get("controlnet_key", arch_name)
    return "SDXL"

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
    diffsynth_controlnet_accordion = components.get(key('diffsynth_controlnet_accordion'))
    diffsynth_controlnet_series_list = components.get(key('diffsynth_controlnet_series'))
    diffsynth_controlnet_types_list = components.get(key('diffsynth_controlnet_types'))
    diffsynth_controlnet_filepaths_list = components.get(key('diffsynth_controlnet_filepaths'))
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
            'controlnet_model_patch': diffsynth_controlnet_accordion,
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
        
        arch_key = get_controlnet_key_for_model_type(model_type)
        
        cn_config = load_controlnet_models()
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
        
        diffsynth_cn_config = load_diffsynth_controlnet_models()
        if diffsynth_controlnet_accordion and 'controlnet_model_patch' in enabled_chains:
            diffsynth_cn_visible = arch_key in diffsynth_cn_config
            updates[diffsynth_controlnet_accordion] = gr.update(visible=diffsynth_cn_visible)

            type_choices = []
            if diffsynth_cn_visible:
                all_types = [t for model in diffsynth_cn_config[arch_key] for t in model.get("Type", [])]
                type_choices = sorted(list(set(all_types)))
            
            default_type = type_choices[0] if type_choices else None
            type_update = gr.update(choices=type_choices, value=default_type)

            series_choices = []
            if diffsynth_cn_visible and default_type:
                series_choices = sorted(list(set(
                    model.get("Series", "Default")
                    for model in diffsynth_cn_config[arch_key]
                    if default_type in model.get("Type", [])
                )))
            
            default_series = series_choices[0] if series_choices else None
            series_update = gr.update(choices=series_choices, value=default_series)

            filepath = "None"
            if diffsynth_cn_visible and default_type and default_series:
                for model in diffsynth_cn_config.get(arch_key, []):
                    if model.get("Series") == default_series and default_type in model.get("Type", []):
                        filepath = model.get("Filepath")
                        break
            
            if diffsynth_controlnet_types_list:
                for type_dd in diffsynth_controlnet_types_list: updates[type_dd] = type_update
            if diffsynth_controlnet_series_list:
                for series_dd in diffsynth_controlnet_series_list: updates[series_dd] = series_update
            if diffsynth_controlnet_filepaths_list:
                for filepath_state in diffsynth_controlnet_filepaths_list: updates[filepath_state] = filepath

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
        "diffsynth_cn_accordion": diffsynth_controlnet_accordion,
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

    if diffsynth_controlnet_types_list:
        for i, cn_type in enumerate(diffsynth_controlnet_types_list): on_model_change_outputs[f'diffsynth_cn_type_{i}'] = cn_type
    if diffsynth_controlnet_series_list:
        for i, cn_series in enumerate(diffsynth_controlnet_series_list): on_model_change_outputs[f'diffsynth_cn_series_{i}'] = cn_series
    if diffsynth_controlnet_filepaths_list:
        for i, cn_filepath in enumerate(diffsynth_controlnet_filepaths_list): on_model_change_outputs[f'diffsynth_cn_filepath_{i}'] = cn_filepath

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
        arch_key = get_controlnet_key_for_model_type(model_type)
        
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
        arch_key = get_controlnet_key_for_model_type(model_type)
        
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
    
    def on_diffsynth_cn_type_change(selected_type, model_type):
        cn_config = load_diffsynth_controlnet_models()
        arch_key = get_controlnet_key_for_model_type(model_type)
        
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

    def on_diffsynth_cn_series_change(selected_series, selected_type, model_type):
        cn_config = load_diffsynth_controlnet_models()
        arch_key = get_controlnet_key_for_model_type(model_type)
        
        filepath = "None"
        if arch_key in cn_config and selected_series and selected_type:
            for model in cn_config[arch_key]:
                if model.get("Series") == selected_series and selected_type in model.get("Type", []):
                    filepath = model.get("Filepath")
                    break
        return filepath

    if diffsynth_controlnet_series_list and diffsynth_controlnet_types_list and diffsynth_controlnet_filepaths_list:
        for i in range(constants.get('MAX_CONTROLNETS', 5)):
            diffsynth_controlnet_types_list[i].change(
                fn=on_diffsynth_cn_type_change,
                inputs=[diffsynth_controlnet_types_list[i], model_type_state],
                outputs=[diffsynth_controlnet_series_list[i], diffsynth_controlnet_filepaths_list[i]],
                show_progress=False,
                show_api=False
            )
            diffsynth_controlnet_series_list[i].change(
                fn=on_diffsynth_cn_series_change,
                inputs=[diffsynth_controlnet_series_list[i], diffsynth_controlnet_types_list[i], model_type_state],
                outputs=[diffsynth_controlnet_filepaths_list[i]],
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

    register_ui_chain_events(components, prefix)


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