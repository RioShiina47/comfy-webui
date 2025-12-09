import gradio as gr
from PIL import Image

from .sd_shared import (
    create_lora_ui, create_controlnet_ui, create_ipadapter_ui, create_embedding_ui,
    register_shared_events,
    create_model_architecture_filter_ui, create_sdxl_category_filter_ui,
    create_run_generation_logic, create_style_ui,
    create_conditioning_ui, create_vae_override_ui,
    create_diffsynth_controlnet_ui
)
from .image_gen_logic import process_inputs as process_inputs_logic

UI_INFO = { 
    "main_tab": "ImageGen", 
    "sub_tab": "Outpaint",
    "run_button_text": "🎨 Outpaint"
}
PREFIX = "outpaint"

def create_ui():
    components = {}
    key = lambda name: f"{PREFIX}_{name}"
    
    from core import node_info_manager
    sampler_choices = node_info_manager.get_node_input_options("KSampler", "sampler_name")
    scheduler_choices = node_info_manager.get_node_input_options("KSampler", "scheduler")
    
    with gr.Column():
        components[key('model_type_state')] = gr.State("sdxl")
        
        components.update(create_model_architecture_filter_ui(PREFIX))
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Row():
                    components[key('sdxl_category_filter')] = create_sdxl_category_filter_ui(prefix=PREFIX, scale=1)
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
                    components[key('run_button')] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
        
        with gr.Row():
            with gr.Column(scale=1):
                components[key('input_image')] = gr.Image(type="pil", label="Input Image", height=255)
            with gr.Column(scale=2):
                components[key('positive_prompt')] = gr.Textbox(label="Prompt", lines=3, placeholder="Enter your prompt or paste generation info here...", interactive=True)
                components[key('negative_prompt')] = gr.Textbox(label="Negative Prompt", lines=3, interactive=True)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components[key('left')] = gr.Slider(label="Pad Left", minimum=0, maximum=512, step=8, value=64)
                    components[key('top')] = gr.Slider(label="Pad Top", minimum=0, maximum=512, step=8, value=64)
                    components[key('right')] = gr.Slider(label="Pad Right", minimum=0, maximum=512, step=8, value=64)
                    components[key('bottom')] = gr.Slider(label="Pad Bottom", minimum=0, maximum=512, step=8, value=64)
                
                components[key('feathering')] = gr.Slider(label="Feathering / Grow Mask", minimum=0, maximum=100, step=1, value=10)
                
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

            with gr.Column(scale=1):
                components[key('output_gallery')] = gr.Gallery(
                    label="Result", show_label=False, object_fit="contain", height=631
                )
        
        create_lora_ui(components, PREFIX)
        create_controlnet_ui(components, PREFIX)
        create_diffsynth_controlnet_ui(components, PREFIX)
        create_ipadapter_ui(components, PREFIX)
        create_embedding_ui(components, PREFIX)
        create_conditioning_ui(components, PREFIX)
        create_vae_override_ui(components, PREFIX)
        create_style_ui(components, PREFIX)
                
    components['run_button'] = components[f'{PREFIX}_run_button']
    return components

def get_main_output_components(components: dict):
    return [
        components[f'{PREFIX}_output_gallery'],
        components[f'{PREFIX}_run_button']
    ]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    register_shared_events(components, PREFIX, sdxl_gallery_height=631, demo=demo)

def process_inputs(ui_values, seed_override=None):
    return process_inputs_logic('outpaint', ui_values, seed_override)

run_generation = create_run_generation_logic(process_inputs, UI_INFO, PREFIX)