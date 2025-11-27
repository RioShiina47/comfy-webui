import gradio as gr
from core import node_info_manager
from .config_loader import load_constants_config, load_model_config
from .event_handlers import on_lora_upload, on_embedding_upload
from .vae_utils import on_vae_upload

constants = load_constants_config()

def create_model_architecture_filter_ui(prefix: str):
    key = lambda name: f"{prefix}_{name}"
    components = {}
    with gr.Row():
        components[key('model_filter')] = gr.Radio(
            ["ALL", "Z-Image", "FLUX", "OmniGen2", "Neta-Lumina", "SDXL", "SD3.5", "HunyuanImage", "HiDream", "Qwen-Image", "Chroma1", "Chroma1-Radiance", "SD1.5"], 
            label="Model Architecture", 
            value="ALL",
            interactive=True
        )
    return components

def create_sdxl_category_filter_ui(prefix: str, **kwargs):
    key = lambda name: f"{prefix}_{name}"
    
    model_config = load_model_config()
    sdxl_models = model_config.get("Checkpoints", {}).get("SDXL", {}).get("models", [])
    
    categories = set()
    for model in sdxl_models:
        if "category" in model and model["category"]:
            categories.add(model["category"])
            
    choices = ["ALL"] + sorted(list(categories))

    return gr.Dropdown(
        label="Filter Models",
        choices=choices,
        value="ALL",
        interactive=True,
        visible=True,
        **kwargs
    )
    
def create_base_ui_components(prefix: str):
    key = lambda name: f"{prefix}_{name}"
    components = {}
    
    components.update(create_model_architecture_filter_ui(prefix))
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            with gr.Row():
                components[key('sdxl_category_filter')] = create_sdxl_category_filter_ui(prefix, scale=1)
                components[key('model_name')] = gr.Dropdown(
                    label="Base Model", 
                    choices=[], 
                    value=None, 
                    interactive=True,
                    scale=3
                )
        with gr.Column(scale=1, min_width=120):
            with gr.Column():
                components[key('parse_prompt_button')] = gr.Button("↙️ Parse")
                components[key('run_button')] = gr.Button("🚀 Generate", variant="primary", elem_classes=["run-shortcut"])
    
    components[key('positive_prompt')] = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt or paste generation info here...", interactive=True)
    components[key('negative_prompt')] = gr.Textbox(label="Negative Prompt", lines=3, interactive=True)
    return components

def create_generation_parameters_ui(prefix: str):
    key = lambda name: f"{prefix}_{name}"
    components = {}

    sampler_choices = node_info_manager.get_node_input_options("KSampler", "sampler_name")
    scheduler_choices = node_info_manager.get_node_input_options("KSampler", "scheduler")
    
    resolution_presets = constants.get('RESOLUTION_MAP', {}).get("sdxl", {})
    
    with gr.Row():
        default_ratio = list(resolution_presets.keys())[0]
        components[key('aspect_ratio_dropdown')] = gr.Dropdown(label="Aspect Ratio", choices=list(resolution_presets.keys()), value=default_ratio, interactive=True)
    with gr.Row():
        w, h = resolution_presets[default_ratio]
        components[key('width')] = gr.Number(value=w, label="Width", interactive=True)
        components[key('height')] = gr.Number(value=h, label="Height", interactive=True)
    with gr.Row():
        components[key('sampler_name')] = gr.Dropdown(label="Sampler", choices=sampler_choices, value="euler", interactive=True)
        components[key('scheduler')] = gr.Dropdown(label="Scheduler", choices=scheduler_choices, value="simple", interactive=True)
    with gr.Row():
        components[key('steps')] = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=25, interactive=True)
        components[key('cfg')] = gr.Slider(label="CFG Scale", minimum=1.0, maximum=15.0, step=0.5, value=7.0, interactive=True)
    with gr.Row():
        components[key('seed')] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0, interactive=True, scale=1)
        components[key('guidance')] = gr.Slider(
            label="Guidance", minimum=1.0, maximum=10.0, step=0.1, value=3.5, visible=False, interactive=True, scale=1
        )
        components[key('clip_skip')] = gr.Slider(
            label="Clip Skip", minimum=1, maximum=4, step=1, value=1, visible=False, interactive=True, scale=1
        )
    with gr.Row():
        components[key('batch_count')] = gr.Slider(label="Batch Count", minimum=1, maximum=50, step=1, value=1, interactive=True)
        components[key('batch_size')] = gr.Slider(label="Batch Size", minimum=1, maximum=8, step=1, value=1, interactive=True)
    return components

def create_vae_override_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = load_constants_config()
    source_choices = ["None"] + constants.get('LORA_SOURCE_CHOICES', [])

    with gr.Accordion("VAE Settings (Override)", open=False) as vae_accordion:
        components[key('vae_accordion')] = vae_accordion
        with gr.Row():
            components[key('vae_source')] = gr.Dropdown(
                label="VAE Source", 
                choices=source_choices, 
                value="None", 
                scale=1, 
                interactive=True
            )
            components[key('vae_id')] = gr.Textbox(
                label="ID/URL/File", 
                placeholder="e.g., 293549", 
                scale=3, 
                interactive=True
            )
            upload_btn = gr.UploadButton(
                "Upload", 
                file_types=[".safetensors", ".pt", ".bin"], 
                scale=1
            )
            components[key('vae_file')] = gr.State(None)
            upload_btn.upload(
                fn=on_vae_upload, 
                inputs=[upload_btn], 
                outputs=[
                    components[key('vae_id')], 
                    components[key('vae_source')], 
                    components[key('vae_file')]
                ],
                show_api=False
            )

def create_lora_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    with gr.Accordion("LoRA Settings", open=False) as lora_accordion:
        components[key('lora_accordion')] = lora_accordion
        lora_rows, sources, ids, scales, files = [], [], [], [], []
        components.update({key('lora_rows'): lora_rows, key('loras_sources'): sources, key('loras_ids'): ids, key('loras_scales'): scales, key('loras_files'): files})
        for i in range(constants.get('MAX_LORAS', 5)):
            with gr.Row(visible=(i < 1)) as row:
                sources.append(gr.Dropdown(label=f"LoRA {i+1}", choices=constants.get('LORA_SOURCE_CHOICES', []), value="Civitai", scale=1, interactive=True))
                ids.append(gr.Textbox(label="ID/URL/File", placeholder="e.g., 133755", scale=2, interactive=True))
                scales.append(gr.Slider(label="Weight", minimum=-1.0, maximum=2.0, step=0.05, value=1.0, scale=2, interactive=True))
                upload_btn = gr.UploadButton("Upload", file_types=[".safetensors"], scale=1)
                files.append(gr.State(None)); lora_rows.append(row)
                upload_btn.upload(fn=on_lora_upload, inputs=[upload_btn], outputs=[ids[i], sources[i], files[i]], show_api=False)
        with gr.Row():
            components[key('add_lora_button')] = gr.Button("✚ Add LoRA")
            components[key('delete_lora_button')] = gr.Button("➖ Delete LoRA", visible=False)
        components[key('lora_count_state')] = gr.State(1)

def create_embedding_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
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
        for i in range(constants.get('MAX_EMBEDDINGS', 5)):
            with gr.Row(visible=(i < 1)) as row:
                sources.append(gr.Dropdown(label=f"Embedding {i+1}", choices=constants.get('LORA_SOURCE_CHOICES', []), value="Civitai", scale=1, interactive=True))
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
        
        for i in range(constants.get('MAX_CONTROLNETS', 5)):
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

def create_ipadapter_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
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
        
        for i in range(constants.get('MAX_IPADAPTERS', 5)):
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

def create_style_ui(components, prefix, max_styles=None):
    if max_styles is None:
        max_styles = constants.get('MAX_STYLES', 5)
    key = lambda name: f"{prefix}_{name}"
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
        
        for i in range(constants.get('MAX_CONDITIONINGS', 10)):
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