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
    
    upload_subdir = "file"
    lora_upload_dir = os.path.join(LORA_DIR, upload_subdir)
    os.makedirs(lora_upload_dir, exist_ok=True)
    
    basename = os.path.basename(file_obj.name)
    new_path = os.path.join(lora_upload_dir, basename)
    shutil.copy(file_obj.name, new_path)
    
    relative_path = os.path.join(upload_subdir, basename)
    
    return relative_path, "File", relative_path

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

def create_lora_ui(components, prefix, accordion_label="LoRA Settings"):
    """Creates the UI for LoRA settings and adds them to the components dict."""
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_loras = constants.get('MAX_LORAS', 5)
    lora_source_choices = constants.get('LORA_SOURCE_CHOICES', ["Civitai", "Custom URL", "File"])
    
    with gr.Accordion(accordion_label, open=False) as lora_accordion:
        components[key('lora_accordion')] = lora_accordion
        lora_rows, sources, ids, scales, files = [], [], [], [], []
        components.update({key('lora_rows'): lora_rows, key('loras_sources'): sources, key('loras_ids'): ids, key('loras_scales'): scales, key('loras_files'): files})
        for i in range(max_loras):
            with gr.Row(visible=(i < 1)) as row:
                sources.append(gr.Dropdown(label=f"LoRA {i+1}", choices=lora_source_choices, value="Civitai", scale=1, interactive=True))
                ids.append(gr.Textbox(label="ID/URL/File", placeholder="e.g., 133755", scale=2, interactive=True))
                scales.append(gr.Slider(label="Weight", minimum=-1.0, maximum=2.0, step=0.05, value=1.0, scale=2, interactive=True))
                upload_btn = gr.UploadButton("Upload", file_types=[".safetensors"], scale=1)
                files.append(gr.State(None))
                lora_rows.append(row)
                upload_btn.upload(fn=on_lora_upload, inputs=[upload_btn], outputs=[ids[i], sources[i], files[i]], show_api=False)
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
            updates = {
                count_state_key: count,
                add_btn_key: gr.update(visible=True),
                del_btn_key: gr.update(visible=count > 1)
            }
            for i in range(max_count):
                updates[rows_key[i]] = gr.update(visible=i < count)

            for i, k in enumerate(reset_keys):
                reset_list = components.get(k, [])
                if count < len(reset_list):
                    default_val = args[i] if i < len(args) else (1.0 if "scale" in k or "strength" in k else "")
                    updates[reset_list[count]] = default_val

            output_list = [updates[count_state_key], updates[add_btn_key], updates[del_btn_key]]
            output_list.extend([updates[r] for r in rows])
            for k in reset_keys:
                reset_list = components.get(k, [])
                for comp in reset_list:
                    output_list.append(updates.get(comp, gr.update()))

            return tuple(output_list)

        if all(k in components for k in [count_state_key, add_btn_key, del_btn_key, rows_key]):
            add_btn = components[add_btn_key]
            del_btn = components[del_btn_key]
            count_state = components[count_state_key]
            rows = components[rows_key]
            
            inputs = [count_state]
            outputs = [count_state, add_btn, del_btn] + rows
            default_values_for_reset = []
            
            for k in reset_keys:
                if k in components:
                    default_val = 1.0 if "scale" in k or "strength" in k else ""
                    default_values_for_reset.append(default_val)
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

    _add_row_factory(key('style_count_state'), key('add_style_button'), key('delete_style_button'), key('style_rows'), constants.get('MAX_STYLES', 5))
    _delete_row_factory(key('style_count_state'), key('add_style_button'), key('delete_style_button'), key('style_rows'), constants.get('MAX_STYLES', 5), reset_keys=[key('style_images'), key('style_strengths')])
    
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