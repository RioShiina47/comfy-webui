import gradio as gr
import os
import shutil
from .config import LORA_DIR, EMBEDDING_DIR
from .yaml_loader import load_and_merge_yaml

def get_ui_constants():
    """Loads constants for shared UI components."""
    return load_and_merge_yaml("ui_constants.yaml")

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

def on_embedding_upload(file_obj):
    """Event handler for uploading an embedding file directly."""
    if file_obj is None: return gr.update(), gr.update(), None
    
    upload_subdir = "file"
    embedding_upload_dir = os.path.join(EMBEDDING_DIR, upload_subdir)
    os.makedirs(embedding_upload_dir, exist_ok=True)
    
    basename = os.path.basename(file_obj.name)
    new_path = os.path.join(embedding_upload_dir, basename)
    shutil.copy(file_obj.name, new_path)
    
    relative_path = os.path.join(upload_subdir, basename)
    
    return relative_path, "File", relative_path

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
    
    base_source_choices = ["Civitai", "Custom URL", "Upload File"]
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
                    ids_txt.append(gr.Textbox(label="ID/URL/File", placeholder="e.g., 133755", interactive=True, visible=True))
                    ids_dd.append(gr.Dropdown(label="File", choices=get_loras_from_dirs(all_lora_dirs), interactive=True, visible=False))
                scales.append(gr.Slider(label="Weight", minimum=-1.0, maximum=2.0, step=0.05, value=1.0, scale=2, interactive=True))
                upload_btn = gr.UploadButton("Upload", file_types=[".safetensors", ".pt", ".bin", ".ckpt"], scale=1)
                files.append(gr.State(None))
                lora_rows.append(row)
                upload_btn.upload(fn=on_lora_upload, inputs=[upload_btn], outputs=[ids_txt[i], sources[i], files[i]], show_api=False)

        def update_lora_input_visibility(source_choice):
            is_file_dropdown = source_choice == "File"
            return gr.update(visible=not is_file_dropdown), gr.update(visible=is_file_dropdown)

        for i in range(max_loras):
            sources[i].change(
                fn=update_lora_input_visibility,
                inputs=[sources[i]],
                outputs=[ids_txt[i], ids_dd[i]],
                show_api=False
            )

        with gr.Row():
            components[key('add_lora_button')] = gr.Button("✚ Add LoRA")
            components[key('delete_lora_button')] = gr.Button("➖ Delete LoRA", visible=False)
        components[key('lora_count_state')] = gr.State(1)

def create_embedding_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_embeddings = constants.get('MAX_EMBEDDINGS', 5)
    source_choices = constants.get('LORA_SOURCE_CHOICES', [])
    with gr.Accordion("Embedding Settings", open=False, visible=True) as embedding_accordion:
        components[key('embedding_accordion')] = embedding_accordion
        gr.Markdown("💡 **Tip:** To use a downloaded embedding, type `embedding:folder_name/filename` in your prompt. Example: `embedding:civitai/12345.safetensors`")
        embedding_rows, sources, ids, files = [], [], [], []
        components.update({
            key('embedding_rows'): embedding_rows, 
            key('embeddings_sources'): sources, 
            key('embeddings_ids'): ids, 
            key('embeddings_files'): files
        })
        for i in range(max_embeddings):
            with gr.Row(visible=(i < 1)) as row:
                sources.append(gr.Dropdown(label=f"Embedding {i+1}", choices=source_choices, value="Civitai", scale=1, interactive=True))
                ids.append(gr.Textbox(label="ID/URL/File", placeholder="e.g., 12345", scale=3, interactive=True))
                upload_btn = gr.UploadButton("Upload", file_types=[".safetensors", ".pt"], scale=1)
                files.append(gr.State(None))
                embedding_rows.append(row)
                upload_btn.upload(fn=on_embedding_upload, inputs=[upload_btn], outputs=[ids[i], sources[i], files[i]], show_api=False)
        with gr.Row():
            components[key('add_embedding_button')] = gr.Button("✚ Add Embedding")
            components[key('delete_embedding_button')] = gr.Button("➖ Delete Embedding", visible=False)
        components[key('embedding_count_state')] = gr.State(1)


def create_controlnet_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_controlnets = constants.get('MAX_CONTROLNETS', 5)
    with gr.Accordion("ControlNet Settings", open=False) as controlnet_accordion:
        components[key('controlnet_accordion')] = controlnet_accordion
        
        cn_rows, images, series, types, strengths, filepaths = [], [], [], [], [], []
        components.update({
            key('controlnet_rows'): cn_rows,
            key('controlnet_images'): images,
            key('controlnet_series'): series,
            key('controlnet_types'): types,
            key('controlnet_strengths'): strengths,
            key('controlnet_filepaths'): filepaths
        })
        
        for i in range(max_controlnets):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"Control Image {i+1}", type="pil", sources="upload", height=256))
                with gr.Column(scale=2):
                    types.append(gr.Dropdown(label="Type", choices=[], interactive=True))
                    series.append(gr.Dropdown(label="Series", choices=[], interactive=True))
                    strengths.append(gr.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                    filepaths.append(gr.State(None))
                cn_rows.append(row)

        with gr.Row():
            components[key('add_controlnet_button')] = gr.Button("✚ Add ControlNet")
            components[key('delete_controlnet_button')] = gr.Button("➖ Delete ControlNet", visible=False)
        components[key('controlnet_count_state')] = gr.State(1)

def create_diffsynth_controlnet_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_controlnets = constants.get('MAX_CONTROLNETS', 5)
    with gr.Accordion("DiffSynth ControlNet Settings", open=False) as diffsynth_controlnet_accordion:
        components[key('diffsynth_controlnet_accordion')] = diffsynth_controlnet_accordion
        
        cn_rows, images, series, types, strengths, filepaths = [], [], [], [], [], []
        components.update({
            key('diffsynth_controlnet_rows'): cn_rows,
            key('diffsynth_controlnet_images'): images,
            key('diffsynth_controlnet_series'): series,
            key('diffsynth_controlnet_types'): types,
            key('diffsynth_controlnet_strengths'): strengths,
            key('diffsynth_controlnet_filepaths'): filepaths
        })
        
        for i in range(max_controlnets):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"Control Image {i+1}", type="pil", sources="upload", height=256))
                with gr.Column(scale=2):
                    types.append(gr.Dropdown(label="Type", choices=[], interactive=True))
                    series.append(gr.Dropdown(label="Series", choices=[], interactive=True))
                    strengths.append(gr.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                    filepaths.append(gr.State(None))
                cn_rows.append(row)

        with gr.Row():
            components[key('add_diffsynth_controlnet_button')] = gr.Button("✚ Add DiffSynth ControlNet")
            components[key('delete_diffsynth_controlnet_button')] = gr.Button("➖ Delete DiffSynth ControlNet", visible=False)
        components[key('diffsynth_controlnet_count_state')] = gr.State(1)

def create_ipadapter_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_ipadapters = constants.get('MAX_IPADAPTERS', 5)
    with gr.Accordion("IPAdapter Settings", open=False) as ipadapter_accordion:
        components[key('ipadapter_accordion')] = ipadapter_accordion
        
        with gr.Row():
            components[key('ipadapter_final_preset')] = gr.Dropdown(label="Preset", choices=[], interactive=True)
            components[key('ipadapter_embeds_scaling')] = gr.Dropdown(
                label="Embeds Scaling", 
                choices=['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],
                value='V only',
                interactive=True
            )
        
        with gr.Row():
            components[key('ipadapter_combine_method')] = gr.Dropdown(
                label="Combine Method",
                choices=["concat", "add", "subtract", "average", "norm average", "max", "min"],
                value="concat",
                interactive=True
            )
            components[key('ipadapter_final_weight')] = gr.Slider(label="Final Weight", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True)
            components[key('ipadapter_final_lora_strength')] = gr.Slider(label="Final LoRA Strength", minimum=0.0, maximum=2.0, step=0.05, value=0.6, interactive=True, visible=False)

        gr.Markdown("---")
        
        ipa_rows, images, presets, weights, lora_strengths = [], [], [], [], []
        components.update({
            key('ipadapter_rows'): ipa_rows,
            key('ipadapter_images'): images,
            key('ipadapter_presets'): presets,
            key('ipadapter_weights'): weights,
            key('ipadapter_lora_strengths'): lora_strengths
        })
        
        for i in range(max_ipadapters):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"IPAdapter Image {i+1}", type="pil", sources="upload", height=256))
                with gr.Column(scale=2):
                    weights.append(gr.Slider(label="Weight", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                    lora_strengths.append(gr.Slider(label="LoRA Strength", minimum=0.0, maximum=2.0, step=0.05, value=0.6, interactive=True, visible=False))
                ipa_rows.append(row)

        with gr.Row():
            components[key('add_ipadapter_button')] = gr.Button("✚ Add IPAdapter")
            components[key('delete_ipadapter_button')] = gr.Button("➖ Delete IPAdapter", visible=False)
        components[key('ipadapter_count_state')] = gr.State(1)

def create_style_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_styles = constants.get('MAX_STYLES', 5)
    with gr.Accordion("Style Settings", open=False) as style_accordion:
        components[key('style_accordion')] = style_accordion
        
        style_rows, images, strengths = [], [], []
        components.update({
            key('style_rows'): style_rows,
            key('style_images'): images,
            key('style_strengths'): strengths
        })
        
        for i in range(max_styles):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"Style Image {i+1}", type="pil", sources="upload"))
                with gr.Column(scale=2):
                    strengths.append(gr.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                style_rows.append(row)

        with gr.Row():
            components[key('add_style_button')] = gr.Button("✚ Add Style Image")
            components[key('delete_style_button')] = gr.Button("➖ Delete Style Image", visible=False)
        components[key('style_count_state')] = gr.State(1)

def create_conditioning_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_conditionings = constants.get('MAX_CONDITIONINGS', 10)
    with gr.Accordion("Conditioning Settings", open=False) as conditioning_accordion:
        components[key('conditioning_accordion')] = conditioning_accordion
        gr.Markdown("💡 **Tip:** Define rectangular areas and assign specific prompts to them. This allows for detailed composition control. Coordinates (X, Y) start from the top-left corner.")
        
        cond_rows, prompts, widths, heights, xs, ys, strengths = [], [], [], [], [], [], []
        components.update({
            key('conditioning_rows'): cond_rows,
            key('conditioning_prompts'): prompts,
            key('conditioning_widths'): widths,
            key('conditioning_heights'): heights,
            key('conditioning_xs'): xs,
            key('conditioning_ys'): ys,
            key('conditioning_strengths'): strengths
        })
        
        for i in range(max_conditionings):
            with gr.Column(visible=(i < 1)) as row_wrapper:
                prompts.append(gr.Textbox(label=f"Area Prompt {i+1}", lines=2, interactive=True))
                with gr.Row():
                    xs.append(gr.Number(label="X", value=0, interactive=True, step=8, scale=1))
                    ys.append(gr.Number(label="Y", value=0, interactive=True, step=8, scale=1))
                    widths.append(gr.Number(label="Width", value=512, interactive=True, step=8, scale=1))
                    heights.append(gr.Number(label="Height", value=512, interactive=True, step=8, scale=1))
                    strengths.append(gr.Slider(label="Strength", minimum=0.1, maximum=2.0, step=0.05, value=1.0, interactive=True, scale=2))
                cond_rows.append(row_wrapper)

        with gr.Row():
            components[key('add_conditioning_button')] = gr.Button("✚ Add Conditioning Area")
            components[key('delete_conditioning_button')] = gr.Button("➖ Delete Conditioning Area", visible=False)
        components[key('conditioning_count_state')] = gr.State(1)

def create_reference_latent_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_refs = constants.get('MAX_REFERENCE_LATENTS', 10)
    with gr.Accordion("Reference Edit Settings", open=False) as ref_accordion:
        components[key('reference_latent_accordion')] = ref_accordion
        gr.Markdown("💡 **Tip:** For multimodal models (like FLUX.2), this feature enables powerful editing and combining capabilities. In txt2img mode, adding a single reference image performs an **Image Edit**, while adding multiple images performs an **Image Combine**.")
        
        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_refs):
                with gr.Column(visible=False, min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"Ref. {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)
        
        components[key('reference_latent_rows')] = ref_image_groups
        components[key('reference_latent_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_reference_latent_button')] = gr.Button("✚ Add Reference Image")
            components[key('delete_reference_latent_button')] = gr.Button("➖ Delete Reference Image", visible=False)
        components[key('reference_latent_count_state')] = gr.State(0)

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
            add_btn.click(fn=_add_row, inputs=[count_state], outputs=outputs, show_progress=False, show_api=False)
    
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

            del_btn.click(fn=_delete_row, inputs=inputs, outputs=outputs, show_progress=False, show_api=False)

    _add_row_factory(key('lora_count_state'), key('add_lora_button'), key('delete_lora_button'), key('lora_rows'), constants.get('MAX_LORAS', 5))
    _delete_row_factory(key('lora_count_state'), key('add_lora_button'), key('delete_lora_button'), key('lora_rows'), constants.get('MAX_LORAS', 5), reset_keys=[key('loras_ids'), key('loras_scales')])

    _add_row_factory(key('embedding_count_state'), key('add_embedding_button'), key('delete_embedding_button'), key('embedding_rows'), constants.get('MAX_EMBEDDINGS', 5))
    _delete_row_factory(key('embedding_count_state'), key('add_embedding_button'), key('delete_embedding_button'), key('embedding_rows'), constants.get('MAX_EMBEDDINGS', 5), reset_keys=[key('embeddings_ids')])

    _add_row_factory(key('controlnet_count_state'), key('add_controlnet_button'), key('delete_controlnet_button'), key('controlnet_rows'), constants.get('MAX_CONTROLNETS', 5))
    _delete_row_factory(key('controlnet_count_state'), key('add_controlnet_button'), key('delete_controlnet_button'), key('controlnet_rows'), constants.get('MAX_CONTROLNETS', 5), reset_keys=[key('controlnet_images'), key('controlnet_strengths')])

    _add_row_factory(key('diffsynth_controlnet_count_state'), key('add_diffsynth_controlnet_button'), key('delete_diffsynth_controlnet_button'), key('diffsynth_controlnet_rows'), constants.get('MAX_CONTROLNETS', 5))
    _delete_row_factory(key('diffsynth_controlnet_count_state'), key('add_diffsynth_controlnet_button'), key('delete_diffsynth_controlnet_button'), key('diffsynth_controlnet_rows'), constants.get('MAX_CONTROLNETS', 5), reset_keys=[key('diffsynth_controlnet_images'), key('diffsynth_controlnet_strengths')])
    
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

        add_cond_btn.click(fn=add_cond_row, inputs=[cond_count, width_num, height_num], outputs=add_cond_outputs, show_progress=False, show_api=False)
        del_cond_btn.click(fn=delete_cond_row, inputs=[cond_count], outputs=del_cond_outputs, show_progress=False, show_api=False)