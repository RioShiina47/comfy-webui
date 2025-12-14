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