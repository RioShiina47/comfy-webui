import gradio as gr
from core import node_info_manager
from core.shared_ui import (
    create_lora_ui, create_embedding_ui, create_controlnet_ui, 
    create_diffsynth_controlnet_ui, create_ipadapter_ui, 
    create_style_ui, create_conditioning_ui
)
from .config_loader import load_constants_config, load_model_config, load_architectures_config
from .vae_utils import on_vae_upload

constants = load_constants_config()

def create_model_architecture_filter_ui(prefix: str):
    key = lambda name: f"{prefix}_{name}"
    components = {}
    
    config = load_architectures_config()
    ordered_architectures = config.get("architecture_order", [])
    choices = ["ALL"] + ordered_architectures

    with gr.Row():
        components[key('model_filter')] = gr.Radio(
            choices, 
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

def create_flux1_ipadapter_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = load_constants_config()
    max_ipadapters = constants.get('MAX_IPADAPTERS', 5)
    with gr.Accordion("IPAdapter Settings (FLUX)", open=False) as flux1_ipadapter_accordion:
        components[key('flux1_ipadapter_accordion')] = flux1_ipadapter_accordion
        
        ipa_rows, images, weights, start_percents, end_percents = [], [], [], [], []
        components.update({
            key('flux1_ipadapter_rows'): ipa_rows,
            key('flux1_ipadapter_images'): images,
            key('flux1_ipadapter_weights'): weights,
            key('flux1_ipadapter_start_percents'): start_percents,
            key('flux1_ipadapter_end_percents'): end_percents,
        })
        
        for i in range(max_ipadapters):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"IPAdapter Image {i+1}", type="pil", sources="upload", height=256))
                with gr.Column(scale=2):
                    weights.append(gr.Slider(label="Weight", minimum=0.0, maximum=2.0, step=0.05, value=0.6, interactive=True))
                    with gr.Row():
                        start_percents.append(gr.Slider(label="Start At", minimum=0.0, maximum=1.0, step=0.01, value=0.0, interactive=True))
                        end_percents.append(gr.Slider(label="End At", minimum=0.0, maximum=1.0, step=0.01, value=0.6, interactive=True))
                ipa_rows.append(row)

        with gr.Row():
            components[key('add_flux1_ipadapter_button')] = gr.Button("✚ Add IPAdapter (FLUX)")
            components[key('delete_flux1_ipadapter_button')] = gr.Button("➖ Delete IPAdapter (FLUX)", visible=False)
        components[key('flux1_ipadapter_count_state')] = gr.State(1)

def create_sd3_ipadapter_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = load_constants_config()
    max_ipadapters = constants.get('MAX_IPADAPTERS', 5)
    with gr.Accordion("IPAdapter Settings (SD3)", open=False) as sd3_ipadapter_accordion:
        components[key('sd3_ipadapter_accordion')] = sd3_ipadapter_accordion
        
        ipa_rows, images, weights, start_percents, end_percents = [], [], [], [], []
        components.update({
            key('sd3_ipadapter_rows'): ipa_rows,
            key('sd3_ipadapter_images'): images,
            key('sd3_ipadapter_weights'): weights,
            key('sd3_ipadapter_start_percents'): start_percents,
            key('sd3_ipadapter_end_percents'): end_percents,
        })
        
        for i in range(max_ipadapters):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"IPAdapter Image {i+1}", type="pil", sources="upload", height=256))
                with gr.Column(scale=2):
                    weights.append(gr.Slider(label="Weight", minimum=0.0, maximum=2.0, step=0.05, value=0.5, interactive=True))
                    with gr.Row():
                        start_percents.append(gr.Slider(label="Start At", minimum=0.0, maximum=1.0, step=0.01, value=0.0, interactive=True))
                        end_percents.append(gr.Slider(label="End At", minimum=0.0, maximum=1.0, step=0.01, value=1.0, interactive=True))
                ipa_rows.append(row)

        with gr.Row():
            components[key('add_sd3_ipadapter_button')] = gr.Button("✚ Add IPAdapter (SD3)")
            components[key('delete_sd3_ipadapter_button')] = gr.Button("➖ Delete IPAdapter (SD3)", visible=False)
        components[key('sd3_ipadapter_count_state')] = gr.State(1)

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