import gradio as gr

from .sd_shared import ( 
    create_base_ui_components, create_generation_parameters_ui, 
    create_lora_ui, create_controlnet_ui, create_ipadapter_ui, create_embedding_ui, 
    create_style_ui, create_conditioning_ui, create_vae_override_ui,
    register_shared_events,
    create_run_generation_logic
)
from .image_gen_logic import process_inputs as process_inputs_logic

UI_INFO = { 
    "main_tab": "ImageGen", 
    "sub_tab": "txt2img", 
    "run_button_text": "🚀 Generate"
}
PREFIX = "txt2img" 

def create_ui():
    components = {}
    
    with gr.Column():
        components[f'{PREFIX}_model_type_state'] = gr.State("sdxl")
        
        components.update(create_base_ui_components(PREFIX))
        
        with gr.Row():
            with gr.Column(scale=1):
                components.update(create_generation_parameters_ui(PREFIX))

            with gr.Column(scale=1):
                components[f'{PREFIX}_output_gallery'] = gr.Gallery(
                    label="Result", show_label=False, object_fit="contain", height=590
                )
        
        create_lora_ui(components, PREFIX)
        create_controlnet_ui(components, PREFIX)
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
    register_shared_events(components, PREFIX, sdxl_gallery_height=590, demo=demo)

def process_inputs(ui_values, seed_override=None):
    return process_inputs_logic('txt2img', ui_values, seed_override)

run_generation = create_run_generation_logic(process_inputs, UI_INFO, PREFIX)