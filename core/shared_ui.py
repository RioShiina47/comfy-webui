import gradio as gr
import os
import shutil
import yaml
from .config import LORA_DIR
from .yaml_loader import load_and_merge_yaml, ROOT_DIR, deep_merge_dicts

_cached_ui_constants = None

def get_ui_constants():
    """Automatically and recursively discovers and merges ui_constants.yaml across modules and global configs."""
    global _cached_ui_constants
    if _cached_ui_constants is not None:
        return _cached_ui_constants

    constants = {}

    # 1. Global config in yaml/ui_constants.yaml
    global_config = load_and_merge_yaml("ui_constants.yaml")
    if global_config and isinstance(global_config, dict):
        constants = deep_merge_dicts(constants, global_config)

    # 2. Recursively discover all ui_constants.yaml in module/
    modules_dir = os.path.join(ROOT_DIR, "module")
    if os.path.isdir(modules_dir):
        for root, _, files in os.walk(modules_dir):
            if "ui_constants.yaml" in files:
                filepath = os.path.join(root, "ui_constants.yaml")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        mod_config = yaml.safe_load(f) or {}
                        if isinstance(mod_config, dict):
                            constants = deep_merge_dicts(constants, mod_config)
                except Exception as e:
                    print(f"Warning: Error loading ui_constants from {filepath}: {e}")

    _cached_ui_constants = constants
    return constants

def on_lora_upload(file_obj):
    """Event handler for uploading a LoRA file directly."""
    if file_obj is None:
        return gr.update(), gr.update(), None
    
    upload_subdir = "upload_file"
    lora_upload_dir = os.path.join(LORA_DIR, upload_subdir)
    os.makedirs(lora_upload_dir, exist_ok=True)
    
    basename = os.path.basename(file_obj.name)
    new_path = os.path.join(lora_upload_dir, basename)
    shutil.copy(file_obj.name, new_path)
    
    relative_path = os.path.join(upload_subdir, basename)
    
    return relative_path, "Upload File", relative_path

def create_lora_ui(components, prefix, module_lora_dir=None, required_lora_dirs=None, accordion_label="LoRA Settings"):
    """Creates the UI for LoRA settings and adds them to the components dict."""
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_loras = constants.get('MAX_LORAS', 5)

    if required_lora_dirs and isinstance(required_lora_dirs, list):
        for subdir in required_lora_dirs:
            full_path = os.path.join(LORA_DIR, "file", subdir)
            os.makedirs(full_path, exist_ok=True)
            
    all_lora_dirs = []
    if required_lora_dirs:
        all_lora_dirs.extend(required_lora_dirs)
    if module_lora_dir and module_lora_dir not in all_lora_dirs:
        all_lora_dirs.append(module_lora_dir)
    
    base_source_choices = ["Civitai", "Hugging Face", "Custom URL", "Upload File"]
    lora_source_choices = base_source_choices + ["File"] if all_lora_dirs else base_source_choices

    def get_loras_from_dirs(subdirs):
        if not subdirs:
            return []
        
        all_files_with_labels = []
        base_search_path = os.path.join(LORA_DIR, "file")

        for subdir in subdirs:
            lora_root_dir = os.path.join(base_search_path, subdir)
            if not os.path.isdir(lora_root_dir):
                continue

            for root, _, files in os.walk(lora_root_dir):
                for filename in files:
                    if filename.lower().endswith(('.safetensors', '.pt', '.bin', '.ckpt')):
                        full_path = os.path.join(root, filename)
                        
                        value_path = os.path.relpath(full_path, base_search_path).replace("\\", "/")
                        display_path = os.path.relpath(full_path, lora_root_dir).replace("\\", "/")
                        
                        display_name = display_path if display_path != filename else filename
                        
                        all_files_with_labels.append((display_name, value_path))
        
        return sorted(all_files_with_labels, key=lambda x: x[0])

    with gr.Accordion(accordion_label, open=False) as lora_accordion:
        components[key('lora_accordion')] = lora_accordion
        lora_rows, sources, ids_txt, ids_dd, scales, files = [], [], [], [], [], []
        components.update({
            key('lora_rows'): lora_rows, 
            key('loras_sources'): sources, 
            key('loras_ids'): ids_txt, 
            key('loras_file_dropdowns'): ids_dd,
            key('loras_scales'): scales, 
            key('loras_files'): files
        })
        
        for i in range(max_loras):
            with gr.Row(visible=(i < 1)) as row:
                sources.append(gr.Dropdown(label=f"LoRA {i+1}", choices=lora_source_choices, value="Civitai", scale=1, interactive=True))
                with gr.Column(scale=2, min_width=100):
                    ids_txt.append(gr.Textbox(label="ID/Repo/URL", interactive=True, visible=True))
                    ids_dd.append(gr.Dropdown(label="File", choices=get_loras_from_dirs(all_lora_dirs), interactive=True, visible=False))
                scales.append(gr.Slider(label="Weight", minimum=-1.0, maximum=2.0, step=0.05, value=1.0, scale=2, interactive=True))
                upload_btn = gr.UploadButton("Upload", file_types=[".safetensors", ".pt", ".bin", ".ckpt"], scale=1)
                files.append(gr.State(None))
                lora_rows.append(row)
                upload_btn.upload(fn=on_lora_upload, inputs=[upload_btn], outputs=[ids_txt[i], sources[i], files[i]], api_name=False)

        def update_lora_input_visibility(source_choice):
            is_file_dropdown = source_choice == "File"
            return gr.update(visible=not is_file_dropdown), gr.update(visible=is_file_dropdown)

        for i in range(max_loras):
            sources[i].change(
                fn=update_lora_input_visibility,
                inputs=[sources[i]],
                outputs=[ids_txt[i], ids_dd[i]],
                api_name=False
            )

        with gr.Row():
            components[key('add_lora_button')] = gr.Button("✚ Add LoRA")
            components[key('delete_lora_button')] = gr.Button("➖ Delete LoRA", visible=False)
        components[key('lora_count_state')] = gr.State(1)

def register_ui_chain_events(components, prefix):
    """
    Registers event handlers for all dynamic chain UIs (add/delete buttons).
    This function should be called for any module that uses these shared UI components.
    """
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()

    def _add_row_factory(count_state_key, add_btn_key, del_btn_key, rows_key, max_count):
        def _add_row(count):
            count += 1
            return (count, gr.update(visible=count < max_count), gr.update(visible=count > 1)) + tuple(gr.update(visible=i < count) for i in range(max_count))
        
        if all(k in components for k in [count_state_key, add_btn_key, del_btn_key, rows_key]):
            add_btn = components[add_btn_key]
            del_btn = components[del_btn_key]
            count_state = components[count_state_key]
            rows = components[rows_key]
            outputs = [count_state, add_btn, del_btn] + rows
            add_btn.click(fn=_add_row, inputs=[count_state], outputs=outputs, show_progress=False, api_name=False)
    
    def _delete_row_factory(count_state_key, add_btn_key, del_btn_key, rows_key, max_count, reset_keys=[]):
        def _delete_row(count, *args):
            count -= 1
            
            count_update = count
            add_btn_update = gr.update(visible=True)
            del_btn_update = gr.update(visible=count > 1)
            
            row_updates = [gr.update(visible=i < count) for i in range(max_count)]
            
            all_reset_updates_dict = {}
            for i, k in enumerate(reset_keys):
                reset_list = components.get(k, [])
                
                for comp in reset_list:
                    all_reset_updates_dict[comp] = gr.update()
                
                if count < len(reset_list):
                    default_val = args[i]
                    component_to_reset = reset_list[count]
                    all_reset_updates_dict[component_to_reset] = gr.update(value=default_val)
            
            final_reset_updates = []
            for k in reset_keys:
                reset_list = components.get(k, [])
                for comp in reset_list:
                    final_reset_updates.append(all_reset_updates_dict.get(comp, gr.update()))

            return (count_update, add_btn_update, del_btn_update) + tuple(row_updates) + tuple(final_reset_updates)

        if all(k in components for k in [count_state_key, add_btn_key, del_btn_key, rows_key]):
            add_btn = components[add_btn_key]
            del_btn = components[del_btn_key]
            count_state = components[count_state_key]
            rows = components[rows_key]
            
            inputs = [count_state]
            outputs = [count_state, add_btn, del_btn] + rows
            
            for k in reset_keys:
                if k in components:
                    default_val = None
                    if "image" in k: default_val = None
                    elif "weight" in k:
                        if "flux1" in k: default_val = 0.6
                        elif "sd3" in k: default_val = 0.5
                        else: default_val = 1.0
                    elif "scale" in k or "strength" in k: default_val = 1.0
                    elif "start" in k: default_val = 0.0
                    elif "end" in k:
                        if "sd3" in k: default_val = 1.0
                        elif "flux1" in k: default_val = 0.6
                        else: default_val = 1.0
                    else: default_val = ""
                    
                    inputs.append(gr.State(default_val))
                    outputs.extend(components[k])

            del_btn.click(fn=_delete_row, inputs=inputs, outputs=outputs, show_progress=False, api_name=False)

    _add_row_factory(key('lora_count_state'), key('add_lora_button'), key('delete_lora_button'), key('lora_rows'), constants.get('MAX_LORAS', 5))
    _delete_row_factory(key('lora_count_state'), key('add_lora_button'), key('delete_lora_button'), key('lora_rows'), constants.get('MAX_LORAS', 5), reset_keys=[key('loras_ids'), key('loras_scales')])

    _add_row_factory(key('embedding_count_state'), key('add_embedding_button'), key('delete_embedding_button'), key('embedding_rows'), constants.get('MAX_EMBEDDINGS', 5))
    _delete_row_factory(key('embedding_count_state'), key('add_embedding_button'), key('delete_embedding_button'), key('embedding_rows'), constants.get('MAX_EMBEDDINGS', 5), reset_keys=[key('embeddings_ids')])

    _add_row_factory(key('controlnet_count_state'), key('add_controlnet_button'), key('delete_controlnet_button'), key('controlnet_rows'), constants.get('MAX_CONTROLNETS', 5))
    _delete_row_factory(key('controlnet_count_state'), key('add_controlnet_button'), key('delete_controlnet_button'), key('controlnet_rows'), constants.get('MAX_CONTROLNETS', 5), reset_keys=[key('controlnet_images'), key('controlnet_strengths')])

    _add_row_factory(key('anima_controlnet_lllite_count_state'), key('add_anima_controlnet_lllite_button'), key('delete_anima_controlnet_lllite_button'), key('anima_controlnet_lllite_rows'), constants.get('MAX_CONTROLNETS', 5))
    _delete_row_factory(key('anima_controlnet_lllite_count_state'), key('add_anima_controlnet_lllite_button'), key('delete_anima_controlnet_lllite_button'), key('anima_controlnet_lllite_rows'), constants.get('MAX_CONTROLNETS', 5), reset_keys=[key('anima_controlnet_lllite_images'), key('anima_controlnet_lllite_strengths'), key('anima_controlnet_lllite_start_percents'), key('anima_controlnet_lllite_end_percents')])

    _add_row_factory(key('diffsynth_controlnet_count_state'), key('add_diffsynth_controlnet_button'), key('delete_diffsynth_controlnet_button'), key('diffsynth_controlnet_rows'), constants.get('MAX_CONTROLNETS', 5))
    _delete_row_factory(key('diffsynth_controlnet_count_state'), key('add_diffsynth_controlnet_button'), key('delete_diffsynth_controlnet_button'), key('diffsynth_controlnet_rows'), constants.get('MAX_CONTROLNETS', 5), reset_keys=[key('diffsynth_controlnet_images'), key('diffsynth_controlnet_strengths')])

    _add_row_factory(key('krea2_controlnet_count_state'), key('add_krea2_controlnet_button'), key('delete_krea2_controlnet_button'), key('krea2_controlnet_rows'), constants.get('MAX_CONTROLNETS', 5))
    _delete_row_factory(key('krea2_controlnet_count_state'), key('add_krea2_controlnet_button'), key('delete_krea2_controlnet_button'), key('krea2_controlnet_rows'), constants.get('MAX_CONTROLNETS', 5), reset_keys=[key('krea2_controlnet_images'), key('krea2_controlnet_strengths')])
    
    _add_row_factory(key('ipadapter_count_state'), key('add_ipadapter_button'), key('delete_ipadapter_button'), key('ipadapter_rows'), constants.get('MAX_IPADAPTERS', 5))
    _delete_row_factory(key('ipadapter_count_state'), key('add_ipadapter_button'), key('delete_ipadapter_button'), key('ipadapter_rows'), constants.get('MAX_IPADAPTERS', 5), reset_keys=[key('ipadapter_images'), key('ipadapter_weights')])

    _add_row_factory(key('flux1_ipadapter_count_state'), key('add_flux1_ipadapter_button'), key('delete_flux1_ipadapter_button'), key('flux1_ipadapter_rows'), constants.get('MAX_IPADAPTERS', 5))
    _delete_row_factory(key('flux1_ipadapter_count_state'), key('add_flux1_ipadapter_button'), key('delete_flux1_ipadapter_button'), key('flux1_ipadapter_rows'), constants.get('MAX_IPADAPTERS', 5), reset_keys=[key('flux1_ipadapter_images'), key('flux1_ipadapter_weights'), key('flux1_ipadapter_start_percents'), key('flux1_ipadapter_end_percents')])

    _add_row_factory(key('sd3_ipadapter_count_state'), key('add_sd3_ipadapter_button'), key('delete_sd3_ipadapter_button'), key('sd3_ipadapter_rows'), constants.get('MAX_IPADAPTERS', 5))
    _delete_row_factory(key('sd3_ipadapter_count_state'), key('add_sd3_ipadapter_button'), key('delete_sd3_ipadapter_button'), key('sd3_ipadapter_rows'), constants.get('MAX_IPADAPTERS', 5), reset_keys=[key('sd3_ipadapter_images'), key('sd3_ipadapter_weights'), key('sd3_ipadapter_start_percents'), key('sd3_ipadapter_end_percents')])

    _add_row_factory(key('style_count_state'), key('add_style_button'), key('delete_style_button'), key('style_rows'), constants.get('MAX_STYLES', 5))
    _delete_row_factory(key('style_count_state'), key('add_style_button'), key('delete_style_button'), key('style_rows'), constants.get('MAX_STYLES', 5), reset_keys=[key('style_images'), key('style_strengths')])
    
    _add_row_factory(key('reference_latent_count_state'), key('add_reference_latent_button'), key('delete_reference_latent_button'), key('reference_latent_rows'), constants.get('MAX_REFERENCE_LATENTS', 10))
    _delete_row_factory(key('reference_latent_count_state'), key('add_reference_latent_button'), key('delete_reference_latent_button'), key('reference_latent_rows'), constants.get('MAX_REFERENCE_LATENTS', 10), reset_keys=[key('reference_latent_images')])

    _add_row_factory(key('hidream_o1_reference_count_state'), key('add_hidream_o1_reference_button'), key('delete_hidream_o1_reference_button'), key('hidream_o1_reference_rows'), constants.get('MAX_REFERENCE_LATENTS', 10))
    _delete_row_factory(key('hidream_o1_reference_count_state'), key('add_hidream_o1_reference_button'), key('delete_hidream_o1_reference_button'), key('hidream_o1_reference_rows'), constants.get('MAX_REFERENCE_LATENTS', 10), reset_keys=[key('hidream_o1_reference_images')])

    max_joyai_refs = constants.get('MAX_JOYAI_REFERENCES', 6)
    _add_row_factory(key('joyai_reference_count_state'), key('add_joyai_reference_button'), key('delete_joyai_reference_button'), key('joyai_reference_rows'), max_joyai_refs)
    _delete_row_factory(key('joyai_reference_count_state'), key('add_joyai_reference_button'), key('delete_joyai_reference_button'), key('joyai_reference_rows'), max_joyai_refs, reset_keys=[key('joyai_reference_images')])
    _add_row_factory(key('joyai_image_count_state'), key('add_joyai_image_button'), key('delete_joyai_image_button'), key('joyai_image_rows'), max_joyai_refs)
    _delete_row_factory(key('joyai_image_count_state'), key('add_joyai_image_button'), key('delete_joyai_image_button'), key('joyai_image_rows'), max_joyai_refs, reset_keys=[key('joyai_image_images')])

    max_ref_imgs = constants.get('MAX_REFERENCE_IMAGES', 10)
    _add_row_factory(key('reference_image_count_state'), key('add_reference_image_button'), key('delete_reference_image_button'), key('reference_image_rows'), max_ref_imgs)
    _delete_row_factory(key('reference_image_count_state'), key('add_reference_image_button'), key('delete_reference_image_button'), key('reference_image_rows'), max_ref_imgs, reset_keys=[key('reference_image_images')])

    max_boogu_edits = constants.get('MAX_BOOGU_IMAGE_EDITS', 10)
    _add_row_factory(key('boogu_image_edit_count_state'), key('add_boogu_image_edit_button'), key('delete_boogu_image_edit_button'), key('boogu_image_edit_rows'), max_boogu_edits)
    _delete_row_factory(key('boogu_image_edit_count_state'), key('add_boogu_image_edit_button'), key('delete_boogu_image_edit_button'), key('boogu_image_edit_rows'), max_boogu_edits, reset_keys=[key('boogu_image_edit_images')])

    max_qwen_edits = constants.get('MAX_QWEN_IMAGE_EDITS', 3)
    _add_row_factory(key('qwen_image_edit_count_state'), key('add_qwen_image_edit_button'), key('delete_qwen_image_edit_button'), key('qwen_image_edit_rows'), max_qwen_edits)
    _delete_row_factory(key('qwen_image_edit_count_state'), key('add_qwen_image_edit_button'), key('delete_qwen_image_edit_button'), key('qwen_image_edit_rows'), max_qwen_edits, reset_keys=[key('qwen_image_edit_images')])

    max_krea2_identity_edits = constants.get('MAX_KREA2_IDENTITY_EDITS', 2)
    _add_row_factory(key('krea2_identity_edit_count_state'), key('add_krea2_identity_edit_button'), key('delete_krea2_identity_edit_button'), key('krea2_identity_edit_rows'), max_krea2_identity_edits)
    _delete_row_factory(key('krea2_identity_edit_count_state'), key('add_krea2_identity_edit_button'), key('delete_krea2_identity_edit_button'), key('krea2_identity_edit_rows'), max_krea2_identity_edits, reset_keys=[key('krea2_identity_edit_images')])

    max_krea2_style_references = constants.get('MAX_KREA2_STYLE_REFERENCES', 3)
    _add_row_factory(key('krea2_style_reference_count_state'), key('add_krea2_style_reference_button'), key('delete_krea2_style_reference_button'), key('krea2_style_reference_rows'), max_krea2_style_references)
    _delete_row_factory(key('krea2_style_reference_count_state'), key('add_krea2_style_reference_button'), key('delete_krea2_style_reference_button'), key('krea2_style_reference_rows'), max_krea2_style_references, reset_keys=[key('krea2_style_reference_images')])

    if all(k in components for k in [key('conditioning_count_state'), key('add_conditioning_button'), key('delete_conditioning_button'), key('conditioning_rows')]):
        add_cond_btn = components[key('add_conditioning_button')]
        del_cond_btn = components[key('delete_conditioning_button')]
        cond_count = components[key('conditioning_count_state')]
        cond_rows = components[key('conditioning_rows')]
        cond_prompts = components[key('conditioning_prompts')]
        cond_widths = components[key('conditioning_widths')]
        cond_heights = components[key('conditioning_heights')]
        width_num = components.get(f"{prefix}_width", gr.State(512))
        height_num = components.get(f"{prefix}_height", gr.State(512))

        def add_cond_row(count, current_w, current_h):
            count += 1
            max_cond = constants.get('MAX_CONDITIONINGS', 10)
            vis_updates = tuple(gr.update(visible=i < count) for i in range(max_cond))
            width_updates = [gr.update()] * max_cond
            height_updates = [gr.update()] * max_cond
            if count > 0:
                width_updates[count-1] = gr.update(value=current_w)
                height_updates[count-1] = gr.update(value=current_h)
            
            return (count, gr.update(visible=count < max_cond), gr.update(visible=count > 1)) + vis_updates + tuple(width_updates) + tuple(height_updates)

        def delete_cond_row(count):
            count -= 1
            max_cond = constants.get('MAX_CONDITIONINGS', 10)
            row_updates = tuple(gr.update(visible=i < count) for i in range(max_cond))
            prompt_updates = [gr.update()] * max_cond
            if count >= 0:
                prompt_updates[count] = gr.update(value="")
            
            return (count, gr.update(visible=True), gr.update(visible=count > 1)) + row_updates + tuple(prompt_updates)

        add_cond_outputs = [cond_count, add_cond_btn, del_cond_btn] + cond_rows + cond_widths + cond_heights
        del_cond_outputs = [cond_count, add_cond_btn, del_cond_btn] + cond_rows + cond_prompts

        add_cond_btn.click(fn=add_cond_row, inputs=[cond_count, width_num, height_num], outputs=add_cond_outputs, show_progress=False, api_name=False)
        del_cond_btn.click(fn=delete_cond_row, inputs=[cond_count], outputs=del_cond_outputs, show_progress=False, api_name=False)
