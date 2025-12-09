import gradio as gr
from PIL import Image
import numpy as np

from .sd_shared import (
    create_lora_ui, create_controlnet_ui, create_ipadapter_ui, create_embedding_ui,
    parse_generation_parameters_for_ui, register_shared_events,
    create_model_architecture_filter_ui, create_sdxl_category_filter_ui,
    create_run_generation_logic, create_style_ui,
    create_conditioning_ui, create_vae_override_ui,
    create_diffsynth_controlnet_ui
)
from .image_gen_logic import process_inputs as process_inputs_logic

UI_INFO = { 
    "main_tab": "ImageGen", 
    "sub_tab": "Inpaint",
    "run_button_text": "🎨 Inpaint" 
}
PREFIX = "inpaint"

def create_ui():
    components = {}
    key = lambda name: f"{PREFIX}_{name}"

    from core import node_info_manager
    sampler_choices = node_info_manager.get_node_input_options("KSampler", "sampler_name")
    scheduler_choices = node_info_manager.get_node_input_options("KSampler", "scheduler")
    
    with gr.Column():
        components[key('model_type_state')] = gr.State("sdxl")
        
        with gr.Row() as model_selection_row:
            components.update(create_model_architecture_filter_ui(PREFIX))
        with gr.Row(equal_height=True) as run_buttons_row:
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
        
        components[key('model_and_run_rows')] = [model_selection_row, run_buttons_row]

        with gr.Row() as main_content_row:
            with gr.Column(scale=1) as editor_column:
                components[key('view_mode')] = gr.Radio(
                    ["Normal View", "Fullscreen View"], 
                    label="Editor View", 
                    value="Normal View", 
                    interactive=True
                )
                components[key('input_image_dict')] = gr.ImageEditor(
                    type="pil", 
                    label="Input Image & Mask", 
                    height=272
                )
            components[key('editor_column')] = editor_column

            with gr.Column(scale=2) as prompts_column:
                components[key('positive_prompt')] = gr.Textbox(label="Prompt", lines=6, placeholder="Enter your prompt or paste generation info here...", interactive=True)
                components[key('negative_prompt')] = gr.Textbox(label="Negative Prompt", lines=6, interactive=True)
            components[key('prompts_column')] = prompts_column

        with gr.Row() as params_and_gallery_row:
            with gr.Column(scale=1):
                with gr.Row():
                    components[key('denoise')] = gr.Slider(
                        label="Denoise", minimum=0.0, maximum=1.0, step=0.05, value=1.0
                    )
                    components[key('grow_mask_by')] = gr.Slider(
                        label="Grow Mask By", minimum=0, maximum=64, step=1, value=6
                    )
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
                    label="Result", show_label=False, object_fit="contain", height=468
                )
        components[key('params_and_gallery_row')] = params_and_gallery_row

        with gr.Column() as accordion_wrapper:
             create_lora_ui(components, PREFIX)
             create_controlnet_ui(components, PREFIX)
             create_diffsynth_controlnet_ui(components, PREFIX)
             create_ipadapter_ui(components, PREFIX)
             create_embedding_ui(components, PREFIX)
             create_conditioning_ui(components, PREFIX)
             create_vae_override_ui(components, PREFIX)
             create_style_ui(components, PREFIX)
        components[key('accordion_wrapper')] = accordion_wrapper
                
    components['run_button'] = components[f'{PREFIX}_run_button']
    return components

def get_main_output_components(components: dict):
    return [
        components[f'{PREFIX}_output_gallery'],
        components[f'{PREFIX}_run_button']
    ]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    prefix = PREFIX
    key = lambda name: f"{PREFIX}_{name}"
    
    def toggle_fullscreen_view(view_mode):
        is_fullscreen = (view_mode == "Fullscreen View")
        
        other_elements_visible = not is_fullscreen
        editor_height = 1024 if is_fullscreen else 272
        
        updates = {}
        for row in components[key('model_and_run_rows')]:
            updates[row] = gr.update(visible=other_elements_visible)
            
        updates[components[key('prompts_column')]] = gr.update(visible=other_elements_visible)
        updates[components[key('params_and_gallery_row')]] = gr.update(visible=other_elements_visible)
        updates[components[key('accordion_wrapper')]] = gr.update(visible=other_elements_visible)
        updates[components[key('input_image_dict')]] = gr.update(height=editor_height)
        
        return updates

    output_components_for_toggle = []
    for row in components[key('model_and_run_rows')]:
        output_components_for_toggle.append(row)
    output_components_for_toggle.extend([
        components[key('prompts_column')],
        components[key('params_and_gallery_row')],
        components[key('accordion_wrapper')],
        components[key('input_image_dict')]
    ])

    components[key('view_mode')].change(
        fn=toggle_fullscreen_view,
        inputs=[components[key('view_mode')]],
        outputs=output_components_for_toggle,
        show_progress=False,
        show_api=False
    )
    
    register_shared_events(components, PREFIX, sdxl_gallery_height=468, demo=demo)

    model_dropdown = components[key("model_name")]
    
    parse_button = components[key("parse_prompt_button")]
    positive_prompt = components[key("positive_prompt")]
    output_map = {
        'positive_prompt': positive_prompt,
        'negative_prompt': components[key("negative_prompt")],
        'model_name': model_dropdown, 'seed': components.get(key("seed")),
        'steps': components.get(key("steps")), 'cfg': components.get(key("cfg")),
        'sampler_name': components.get(key("sampler_name")), 'scheduler': components.get(key("scheduler")),
        'clip_skip': components.get(key("clip_skip"))
    }
    output_keys = ['positive_prompt', 'negative_prompt', 'model_name', 'seed', 'steps', 'cfg', 'sampler_name', 'scheduler', 'clip_skip']
    final_outputs = [output_map[key] for key in output_keys if output_map[key] is not None]

    def on_parse_prompt_wrapper(prompt_text):
        parsed_data = parse_generation_parameters_for_ui(prompt_text)
        return_values = [parsed_data.get(key, gr.update()) for key in output_keys if output_map[key] is not None]
        return tuple(return_values)

    parse_button.click(
        fn=on_parse_prompt_wrapper,
        inputs=[positive_prompt],
        outputs=final_outputs,
        show_api=False
    )

def process_inputs(ui_values, seed_override=None):
    return process_inputs_logic('inpaint', ui_values, seed_override)

run_generation = create_run_generation_logic(process_inputs, UI_INFO, PREFIX)