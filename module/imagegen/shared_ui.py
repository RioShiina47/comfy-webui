import gradio as gr
import os
import shutil
from core.config import EMBEDDING_DIR
from core.shared_ui import get_ui_constants


def on_embedding_upload(file_obj):
    """Event handler for uploading an embedding file directly."""
    if file_obj is None:
        return gr.update(), gr.update(), None
    
    upload_subdir = "file"
    embedding_upload_dir = os.path.join(EMBEDDING_DIR, upload_subdir)
    os.makedirs(embedding_upload_dir, exist_ok=True)
    
    basename = os.path.basename(file_obj.name)
    new_path = os.path.join(embedding_upload_dir, basename)
    shutil.copy(file_obj.name, new_path)
    
    relative_path = os.path.join(upload_subdir, basename)
    
    return relative_path, "File", relative_path


def create_embedding_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_embeddings = constants.get('MAX_EMBEDDINGS', 5)
    source_choices = constants.get('LORA_SOURCE_CHOICES', [])
    with gr.Accordion("Embedding Settings", open=False, visible=True) as embedding_accordion:
        components[key('embedding_accordion')] = embedding_accordion
        gr.Markdown("💡 **Tip:** To use a downloaded embedding, type `embedding:folder_name/filename` in your prompt. Example: `embedding:civitai/12345.safetensors` or `embedding:huggingface/lazypos.safetensors`")
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
                ids.append(gr.Textbox(label="ID/Repo/URL/File", placeholder="e.g., 12345 or repo/name/file.safetensors", scale=3, interactive=True))
                upload_btn = gr.UploadButton("Upload", file_types=[".safetensors", ".pt"], scale=1)
                files.append(gr.State(None))
                embedding_rows.append(row)
                upload_btn.upload(fn=on_embedding_upload, inputs=[upload_btn], outputs=[ids[i], sources[i], files[i]], api_name=False)
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


def create_anima_controlnet_lllite_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_controlnets = constants.get('MAX_CONTROLNETS', 5)
    with gr.Accordion("Anima ControlNet Lllite Settings", open=False) as anima_cn_accordion:
        components[key('anima_controlnet_lllite_accordion')] = anima_cn_accordion
        
        cn_rows, images, series, types, strengths, filepaths, start_percents, end_percents = [], [], [], [], [], [], [], []
        components.update({
            key('anima_controlnet_lllite_rows'): cn_rows,
            key('anima_controlnet_lllite_images'): images,
            key('anima_controlnet_lllite_series'): series,
            key('anima_controlnet_lllite_types'): types,
            key('anima_controlnet_lllite_strengths'): strengths,
            key('anima_controlnet_lllite_filepaths'): filepaths,
            key('anima_controlnet_lllite_start_percents'): start_percents,
            key('anima_controlnet_lllite_end_percents'): end_percents
        })
        
        for i in range(max_controlnets):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"Control Image {i+1}", type="pil", sources="upload", height=256))
                with gr.Column(scale=2):
                    types.append(gr.Dropdown(label="Type", choices=[], interactive=True))
                    series.append(gr.Dropdown(label="Series", choices=[], interactive=True))
                    strengths.append(gr.Slider(label="Strength", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                    with gr.Row():
                        start_percents.append(gr.Slider(label="Start At", minimum=0.0, maximum=1.0, step=0.01, value=0.0, interactive=True))
                        end_percents.append(gr.Slider(label="End At", minimum=0.0, maximum=1.0, step=0.01, value=1.0, interactive=True))
                    filepaths.append(gr.State(None))
                cn_rows.append(row)

        with gr.Row():
            components[key('add_anima_controlnet_lllite_button')] = gr.Button("✚ Add Lllite")
            components[key('delete_anima_controlnet_lllite_button')] = gr.Button("➖ Delete Lllite", visible=False)
        components[key('anima_controlnet_lllite_count_state')] = gr.State(1)


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
        gr.Markdown("💡 **Tip:** For multimodal models, this feature enables powerful editing and combining capabilities. In txt2img mode, adding a single reference image performs an **Image Edit**, while adding multiple images performs an **Image Combine**.")
        
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


def create_hidream_o1_reference_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_refs = constants.get('MAX_REFERENCE_LATENTS', 10)
    with gr.Accordion("HiDream-O1 Reference Edit Settings", open=False) as ref_accordion:
        components[key('hidream_o1_reference_accordion')] = ref_accordion
        gr.Markdown("💡 **Tip:** For multimodal models, this feature enables powerful editing and combining capabilities. In txt2img mode, adding a single reference image performs an **Image Edit**, while adding multiple images performs an **Image Combine**.")
        
        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_refs):
                with gr.Column(visible=False, min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"Ref. {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)
        
        components[key('hidream_o1_reference_rows')] = ref_image_groups
        components[key('hidream_o1_reference_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_hidream_o1_reference_button')] = gr.Button("✚ Add Reference Image")
            components[key('delete_hidream_o1_reference_button')] = gr.Button("➖ Delete Reference Image", visible=False)
        components[key('hidream_o1_reference_count_state')] = gr.State(0)


def create_sensenova_reference_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = get_ui_constants()
    max_refs = constants.get('MAX_REFERENCE_LATENTS', 10)
    with gr.Accordion("SenseNova Reference Edit Settings", open=False) as ref_accordion:
        components[key('sensenova_reference_accordion')] = ref_accordion
        gr.Markdown("💡 **Tip:** For SenseNova models, this feature enables reference image editing and combining capabilities. In txt2img mode, adding a single reference image performs an **Image Edit**, while adding multiple images performs an **Image Combine**.")
        
        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_refs):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"Ref. {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)
        
        components[key('sensenova_reference_rows')] = ref_image_groups
        components[key('sensenova_reference_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_sensenova_reference_button')] = gr.Button("✚ Add Reference Image")
            components[key('delete_sensenova_reference_button')] = gr.Button("➖ Delete Reference Image", visible=False)
        components[key('sensenova_reference_count_state')] = gr.State(1)
        components[key('all_sensenova_reference_components_flat')] = ref_image_inputs


def create_joyai_image_ui(components, prefix, max_units=None):
    if max_units is None:
        constants = get_ui_constants()
        max_units = constants.get('MAX_JOYAI_REFERENCES', 6)
    key = lambda name: f"{prefix}_{name}"
    with gr.Accordion("JoyAI Reference Edit Settings", open=False) as ref_accordion:
        components[key('joyai_image_accordion')] = ref_accordion
        components[key('joyai_reference_accordion')] = ref_accordion
        gr.Markdown("💡 **Tip:** For multimodal models, this feature enables powerful editing and combining capabilities. In txt2img mode, adding a single reference image performs an **Image Edit** (JoyAI-Image-Edit recommended), while adding multiple images performs an **Image Combine** (JoyAI-Image-Edit-Plus recommended).")
        
        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"Ref. {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)
        
        components[key('joyai_image_rows')] = ref_image_groups
        components[key('joyai_image_images')] = ref_image_inputs
        components[key('joyai_reference_rows')] = ref_image_groups
        components[key('joyai_reference_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_joyai_image_button')] = gr.Button("✚ Add Reference Image")
            components[key('delete_joyai_image_button')] = gr.Button("➖ Delete Reference Image", visible=False)
            components[key('add_joyai_reference_button')] = components[key('add_joyai_image_button')]
            components[key('delete_joyai_reference_button')] = components[key('delete_joyai_image_button')]
        components[key('joyai_image_count_state')] = gr.State(1)
        components[key('joyai_reference_count_state')] = components[key('joyai_image_count_state')]
        
        components[key('all_joyai_image_components_flat')] = ref_image_inputs
        components[key('all_joyai_reference_components_flat')] = ref_image_inputs

create_joyai_reference_ui = create_joyai_image_ui


def create_reference_image_ui(components, prefix, max_units=None):
    if max_units is None:
        constants = get_ui_constants()
        max_units = constants.get('MAX_REFERENCE_IMAGES', 10)
    key = lambda name: f"{prefix}_{name}"
    with gr.Accordion("Mage-Flow Reference Edit Settings", open=False) as ref_accordion:
        components[key('reference_image_accordion')] = ref_accordion
        gr.Markdown("💡 **Tip:** (Mage-Flow-Edit-Turbo/Mage-Flow-Edit recommended) For multimodal models, this feature enables powerful editing and combining capabilities. In txt2img mode, adding a single reference image performs an **Image Edit**, while adding multiple images performs an **Image Combine**.")
        
        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"Ref. {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)
        
        components[key('reference_image_rows')] = ref_image_groups
        components[key('reference_image_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_reference_image_button')] = gr.Button("✚ Add Reference Image")
            components[key('delete_reference_image_button')] = gr.Button("➖ Delete Reference Image", visible=False)
        components[key('reference_image_count_state')] = gr.State(1)
        
        components[key('all_reference_image_components_flat')] = ref_image_inputs


def create_boogu_image_edit_ui(components, prefix, max_units=None):
    if max_units is None:
        constants = get_ui_constants()
        max_units = constants.get('MAX_BOOGU_IMAGE_EDITS', 10)
    key = lambda name: f"{prefix}_{name}"
    with gr.Accordion("Boogu-Image Edit Settings", open=False) as ref_accordion:
        components[key('boogu_image_edit_accordion')] = ref_accordion
        gr.Markdown("💡 **Tip:** (Boogu-Image-Edit-Turbo/Boogu-Image-Edit recommended) In txt2img mode, adding a single reference image performs an **Image Edit**, while adding multiple images performs an **Image Combine**.")
        
        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"Ref. {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)
        
        components[key('boogu_image_edit_rows')] = ref_image_groups
        components[key('boogu_image_edit_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_boogu_image_edit_button')] = gr.Button("✚ Add Reference Image")
            components[key('delete_boogu_image_edit_button')] = gr.Button("➖ Delete Reference Image", visible=False)
        components[key('boogu_image_edit_count_state')] = gr.State(1)
        
        components[key('all_boogu_image_edit_components_flat')] = ref_image_inputs


def create_qwen_image_edit_ui(components, prefix, max_units=None):
    if max_units is None:
        constants = get_ui_constants()
        max_units = constants.get('MAX_QWEN_IMAGE_EDITS', 3)
    key = lambda name: f"{prefix}_{name}"
    with gr.Accordion("Qwen-Image Edit Settings", open=False) as ref_accordion:
        components[key('qwen_image_edit_accordion')] = ref_accordion
        gr.Markdown("💡 **Tip:** (lightx2v/Qwen-Image-Edit-2511-Lightning recommended) In txt2img mode, adding a single reference image performs an **Image Edit**, while adding multiple images performs an **Image Combine**.")
        
        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"Ref. {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)
        
        components[key('qwen_image_edit_rows')] = ref_image_groups
        components[key('qwen_image_edit_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_qwen_image_edit_button')] = gr.Button("✚ Add Reference Image")
            components[key('delete_qwen_image_edit_button')] = gr.Button("➖ Delete Reference Image", visible=False)
        components[key('qwen_image_edit_count_state')] = gr.State(1)
        
        components[key('all_qwen_image_edit_components_flat')] = ref_image_inputs


def create_krea2_identity_edit_ui(components, prefix, max_units=None):
    if max_units is None:
        constants = get_ui_constants()
        max_units = constants.get('MAX_KREA2_IDENTITY_EDITS', 2)
    key = lambda name: f"{prefix}_{name}"
    with gr.Accordion("Krea2 Identity Edit Settings", open=False) as ref_accordion:
        components[key('krea2_identity_edit_accordion')] = ref_accordion
        gr.Markdown("💡 **Tip:** Processed using the [lbouaraba/comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) node. (Krea-2-Turbo recommended, Krea-2-Raw need set ZeroGPU Duration (s) to 120 ) In txt2img mode, adding a single reference image performs an **Image Edit**, while adding multiple images performs an **Image Combine**.")
        
        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"Ref. {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)
        
        components[key('krea2_identity_edit_rows')] = ref_image_groups
        components[key('krea2_identity_edit_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_krea2_identity_edit_button')] = gr.Button("✚ Add Reference Image")
            components[key('delete_krea2_identity_edit_button')] = gr.Button("➖ Delete Reference Image", visible=False)
        components[key('krea2_identity_edit_count_state')] = gr.State(1)
        
        components[key('all_krea2_identity_edit_components_flat')] = ref_image_inputs


def create_krea2_style_reference_ui(components, prefix, max_units=None):
    if max_units is None:
        constants = get_ui_constants()
        max_units = constants.get('MAX_KREA2_STYLE_REFERENCES', 3)
    key = lambda name: f"{prefix}_{name}"
    with gr.Accordion("Krea2 Style Reference Edit Settings", open=False) as ref_accordion:
        components[key('krea2_style_reference_accordion')] = ref_accordion
        gr.Markdown("💡 **Tip:** (Krea-2-Turbo recommended) Add style reference images to perform style reference editing.")
        
        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"Ref. {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)
        
        components[key('krea2_style_reference_rows')] = ref_image_groups
        components[key('krea2_style_reference_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_krea2_style_reference_button')] = gr.Button("✚ Add Reference Image")
            components[key('delete_krea2_style_reference_button')] = gr.Button("➖ Delete Reference Image", visible=False)
        components[key('krea2_style_reference_count_state')] = gr.State(1)
        
        components[key('all_krea2_style_reference_components_flat')] = ref_image_inputs

