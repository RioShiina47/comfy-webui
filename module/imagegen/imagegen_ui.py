import gradio as gr
from PIL import Image

from .shared.ui_components import (
    create_model_architecture_filter_ui, create_sdxl_category_filter_ui,
    create_lora_ui, create_controlnet_ui, create_ipadapter_ui, create_embedding_ui,
    create_anima_controlnet_lllite_ui, create_krea2_controlnet_ui,
    create_style_ui, create_conditioning_ui,
    create_diffsynth_controlnet_ui, create_flux1_ipadapter_ui, create_sd3_ipadapter_ui,
    create_reference_latent_ui, create_hidream_o1_reference_ui, create_joyai_image_ui, create_joyai_reference_ui,
    create_reference_image_ui, create_pid_ui,
    create_boogu_image_edit_ui, create_qwen_image_edit_ui,
    create_krea2_identity_edit_ui, create_krea2_style_reference_ui,
)
from .shared.event_handlers import register_shared_events
from .shared.generation import create_run_generation_logic
from .shared.vae_utils import create_vae_override_ui
from .shared.config_loader import load_constants_config
from .imagegen_logic import process_inputs as process_inputs_logic

UI_INFO = {
    "main_tab": "ImageGen",
    "sub_tab": "ImageGen",
    "run_button_text": "Run"
}
PREFIX = "imagegen"
TYPE_CHOICES = ["Txt2Img", "Img2Img", "Inpaint", "Outpaint", "Hires. Fix"]
TYPE_MAP = {
    'Txt2Img': 'txt2img',
    'Img2Img': 'img2img',
    'Inpaint': 'inpaint',
    'Outpaint': 'outpaint',
    'Hires. Fix': 'hires_fix',
    'txt2img': 'txt2img',
    'img2img': 'img2img',
    'inpaint': 'inpaint',
    'outpaint': 'outpaint',
    'hires_fix': 'hires_fix'
}

def create_ui():
    components = {}
    key = lambda name: f"{PREFIX}_{name}"
    
    from core import node_info_manager
    sampler_choices = node_info_manager.get_node_input_options("KSampler", "sampler_name")
    if not sampler_choices:
        sampler_choices = [
            "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
            "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
            "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2", "res_multistep", "er_sde"
        ]
    scheduler_choices = node_info_manager.get_node_input_options("KSampler", "scheduler")
    if not scheduler_choices:
        scheduler_choices = [
            "normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"
        ]
    constants = load_constants_config()
    resolution_presets = constants.get('RESOLUTION_MAP', {}).get("sdxl", {})
    default_ratio = list(resolution_presets.keys())[0] if resolution_presets else "1:1 (Square)"
    default_w, default_h = resolution_presets.get(default_ratio, (1024, 1024))
    
    with gr.Column():
        components[key('type')] = gr.Radio(
            choices=TYPE_CHOICES,
            value="Txt2Img",
            label="Task",
            interactive=True
        )
        components[key('model_type_state')] = gr.State("sdxl")
        
        with gr.Row() as arch_row:
            components.update(create_model_architecture_filter_ui(PREFIX))
        components[key('arch_row')] = arch_row
        
        with gr.Row() as model_and_run_row:
            components[key('sdxl_category_filter')] = create_sdxl_category_filter_ui(prefix=PREFIX, scale=1)
            components[key('model_name')] = gr.Dropdown(
                label="Base Model",
                choices=[],
                value=None,
                interactive=True,
                scale=3
            )
            with gr.Column(scale=1):
                components[key('run_button')] = gr.Button("Run", variant="primary", elem_classes=["run-shortcut"])
        components[key('model_and_run_rows')] = [arch_row, model_and_run_row]

        with gr.Row() as inputs_prompts_row:
            with gr.Column(scale=1, visible=False) as image_input_col:
                components[key('input_image')] = gr.Image(
                    type="pil",
                    label="Input Image",
                    height=255,
                    visible=False
                )
                with gr.Column(visible=False) as inpaint_box:
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
                components[key('inpaint_box')] = inpaint_box
            components[key('image_input_col')] = image_input_col

            with gr.Column(scale=2) as prompts_col:
                components[key('positive_prompt')] = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Enter your prompt or paste generation info here...",
                    interactive=True
                )
                components[key('negative_prompt')] = gr.Textbox(
                    label="Negative Prompt",
                    lines=3,
                    interactive=True
                )
            components[key('prompts_col')] = prompts_col
        components[key('inputs_prompts_row')] = inputs_prompts_row

        with gr.Row() as params_and_gallery_row:
            with gr.Column(scale=1) as params_col:
                # Txt2Img aspect ratio & width/height
                with gr.Row(visible=True) as aspect_ratio_row:
                    components[key('aspect_ratio_dropdown')] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(resolution_presets.keys()),
                        value=default_ratio,
                        interactive=True,
                        allow_custom_value=True
                    )
                components[key('aspect_ratio_row')] = aspect_ratio_row

                with gr.Row(visible=True) as width_height_row:
                    components[key('width')] = gr.Number(value=default_w, label="Width", interactive=True)
                    components[key('height')] = gr.Number(value=default_h, label="Height", interactive=True)
                components[key('width_height_row')] = width_height_row

                # Img2Img / Inpaint / Hires. Fix denoise & grow mask by
                with gr.Row(visible=False) as denoise_row:
                    components[key('denoise')] = gr.Slider(
                        label="Denoise",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0
                    )
                    components[key('grow_mask_by')] = gr.Slider(
                        label="Grow Mask By",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=6,
                        visible=False
                    )
                components[key('denoise_row')] = denoise_row

                # Outpaint pad & feathering parameters
                with gr.Column(visible=False) as outpaint_pads_col:
                    with gr.Row():
                        components[key('left')] = gr.Slider(label="Pad Left", minimum=0, maximum=512, step=8, value=64)
                        components[key('right')] = gr.Slider(label="Pad Right", minimum=0, maximum=512, step=8, value=64)
                    with gr.Row():
                        components[key('top')] = gr.Slider(label="Pad Top", minimum=0, maximum=512, step=8, value=64)
                        components[key('bottom')] = gr.Slider(label="Pad Bottom", minimum=0, maximum=512, step=8, value=64)
                    components[key('feathering')] = gr.Slider(label="Feathering / Grow Mask", minimum=0, maximum=100, step=1, value=10)
                components[key('outpaint_pads_col')] = outpaint_pads_col

                # Hires. Fix upscaler parameters
                with gr.Row(visible=False) as hires_upscaler_row:
                    components[key('hires_upscaler')] = gr.Dropdown(
                        label="Upscaler",
                        choices=["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],
                        value="nearest-exact"
                    )
                    components[key('hires_scale_by')] = gr.Slider(
                        label="Upscale by",
                        minimum=1.0,
                        maximum=4.0,
                        step=0.1,
                        value=1.5
                    )
                components[key('hires_upscaler_row')] = hires_upscaler_row

                # Common parameters
                with gr.Row():
                    components[key('sampler_name')] = gr.Dropdown(label="Sampler", choices=sampler_choices, value="euler", interactive=True, allow_custom_value=True)
                    components[key('scheduler')] = gr.Dropdown(label="Scheduler", choices=scheduler_choices, value="simple", interactive=True, allow_custom_value=True)
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
            components[key('params_col')] = params_col

            with gr.Column(scale=1):
                components[key('output_gallery')] = gr.Gallery(
                    label="Result", show_label=False, object_fit="contain", height=590, columns=2
                )
        components[key('params_and_gallery_row')] = params_and_gallery_row

        with gr.Column() as accordion_wrapper:
            create_lora_ui(components, PREFIX)
            create_controlnet_ui(components, PREFIX)
            create_anima_controlnet_lllite_ui(components, PREFIX)
            create_krea2_controlnet_ui(components, PREFIX)
            create_diffsynth_controlnet_ui(components, PREFIX)
            create_ipadapter_ui(components, PREFIX)
            create_flux1_ipadapter_ui(components, PREFIX)
            create_sd3_ipadapter_ui(components, PREFIX)
            create_embedding_ui(components, PREFIX)
            create_conditioning_ui(components, PREFIX)
            create_reference_latent_ui(components, PREFIX)
            create_hidream_o1_reference_ui(components, PREFIX)
            create_joyai_image_ui(components, PREFIX)
            create_reference_image_ui(components, PREFIX)
            create_boogu_image_edit_ui(components, PREFIX)
            create_qwen_image_edit_ui(components, PREFIX)
            create_krea2_identity_edit_ui(components, PREFIX)
            create_krea2_style_reference_ui(components, PREFIX)
            create_vae_override_ui(components, PREFIX)
            create_style_ui(components, PREFIX)
            create_pid_ui(components, PREFIX)
        components[key('accordion_wrapper')] = accordion_wrapper

    components['run_button'] = components[key('run_button')]
    return components

def get_main_output_components(components: dict):
    return [
        components[f'{PREFIX}_output_gallery'],
        components[f'{PREFIX}_run_button']
    ]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    key = lambda name: f"{PREFIX}_{name}"
    
    # 1. Register base events for architecture/model selection and chain accordions
    register_shared_events(components, PREFIX, sdxl_gallery_height=590, demo=demo)

    # 2. Dynamic Task type change event
    def on_type_change(type_val, model_name):
        from .shared.config_loader import load_model_config, load_features_config
        from .shared.utils import get_model_type

        is_txt2img = (type_val == "Txt2Img")
        is_img2img = (type_val == "Img2Img")
        is_inpaint = (type_val == "Inpaint")
        is_outpaint = (type_val == "Outpaint")
        is_hires_fix = (type_val == "Hires. Fix")
        
        denoise_val = 1.0 if is_txt2img else (0.75 if is_img2img else (1.0 if is_inpaint else (0.55 if is_hires_fix else 1.0)))
        run_text = "Run Inpaint" if is_inpaint else ("Run Outpaint" if is_outpaint else ("Run Hires. Fix" if is_hires_fix else "Run"))
        
        prompt_lines = 6 if is_inpaint else 3
        gallery_cols = 2 if is_txt2img else 1
        gallery_height = 590 if is_txt2img else 468
        
        # Check if model supports PID (pid only supports txt2img)
        model_config = load_model_config()
        features_config = load_features_config()
        model_type = get_model_type(model_name, model_config) if model_name else "sdxl"
        arch_features = features_config.get(model_type, features_config.get('default', {}))
        pid_supported = 'pid' in arch_features.get('enabled_chains', [])
        pid_visible = is_txt2img and pid_supported

        updates = {
            components[key('image_input_col')]: gr.update(visible=not is_txt2img),
            components[key('input_image')]: gr.update(visible=(is_img2img or is_outpaint or is_hires_fix)),
            components[key('inpaint_box')]: gr.update(visible=is_inpaint),
            components[key('positive_prompt')]: gr.update(lines=prompt_lines),
            components[key('negative_prompt')]: gr.update(lines=prompt_lines),
            components[key('aspect_ratio_row')]: gr.update(visible=is_txt2img),
            components[key('width_height_row')]: gr.update(visible=is_txt2img),
            components[key('denoise_row')]: gr.update(visible=(is_img2img or is_inpaint or is_hires_fix)),
            components[key('denoise')]: gr.update(value=denoise_val),
            components[key('grow_mask_by')]: gr.update(visible=is_inpaint),
            components[key('outpaint_pads_col')]: gr.update(visible=is_outpaint),
            components[key('hires_upscaler_row')]: gr.update(visible=is_hires_fix),
            components[key('run_button')]: gr.update(value=run_text),
            components[key('output_gallery')]: gr.update(columns=gallery_cols, height=gallery_height),
        }
        if key('pid_accordion') in components:
            updates[components[key('pid_accordion')]] = gr.update(visible=pid_visible)

        return updates

    type_change_outputs = [
        components[key('image_input_col')],
        components[key('input_image')],
        components[key('inpaint_box')],
        components[key('positive_prompt')],
        components[key('negative_prompt')],
        components[key('aspect_ratio_row')],
        components[key('width_height_row')],
        components[key('denoise_row')],
        components[key('denoise')],
        components[key('grow_mask_by')],
        components[key('outpaint_pads_col')],
        components[key('hires_upscaler_row')],
        components[key('run_button')],
        components[key('output_gallery')],
    ]
    if key('pid_accordion') in components:
        type_change_outputs.append(components[key('pid_accordion')])

    components[key('type')].change(
        fn=on_type_change,
        inputs=[components[key('type')], components[key('model_name')]],
        outputs=type_change_outputs,
        show_progress=False,
        api_name=False
    )

    # 3. Fullscreen view toggle for inpaint editor
    def toggle_fullscreen_view(view_mode):
        is_fullscreen = (view_mode == "Fullscreen View")
        other_elements_visible = not is_fullscreen
        editor_height = 800 if is_fullscreen else 272
        
        updates = {
            components[key('type')]: gr.update(visible=other_elements_visible),
            components[key('prompts_col')]: gr.update(visible=other_elements_visible),
            components[key('params_and_gallery_row')]: gr.update(visible=other_elements_visible),
            components[key('accordion_wrapper')]: gr.update(visible=other_elements_visible),
            components[key('input_image_dict')]: gr.update(height=editor_height),
        }
        
        for row in components.get(key('model_and_run_rows'), []):
            updates[row] = gr.update(visible=other_elements_visible)
            
        return updates

    fs_outputs = [
        components[key('type')],
        components[key('prompts_col')],
        components[key('params_and_gallery_row')],
        components[key('accordion_wrapper')],
        components[key('input_image_dict')]
    ]
    for row in components.get(key('model_and_run_rows'), []):
        fs_outputs.append(row)

    components[key('view_mode')].change(
        fn=toggle_fullscreen_view,
        inputs=[components[key('view_mode')]],
        outputs=fs_outputs,
        show_progress=False,
        api_name=False
    )

def process_inputs(ui_values, seed_override=None):
    raw_task = ui_values.get(f'{PREFIX}_type', 'Txt2Img')
    task_type = TYPE_MAP.get(raw_task, 'txt2img')
    return process_inputs_logic(task_type, ui_values, seed_override, prefix=PREFIX)

run_generation = create_run_generation_logic(process_inputs, UI_INFO, PREFIX)
