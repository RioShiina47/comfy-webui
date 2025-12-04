import gradio as gr
from .controlnet_auxiliary_preprocessor_logic import process_inputs, MAX_DYNAMIC_CONTROLS
from core.utils import create_simple_run_generation
from core import node_info_manager

UI_INFO = {
    "workflow_recipe": None,
    "main_tab": "Tools",
    "sub_tab": "ControlNet Auxiliary Preprocessors",
    "run_button_text": "🕹️ Run Preprocessor"
}

def get_preprocessor_choices():
    all_node_info = node_info_manager.get_all_node_info()
    if not all_node_info:
        return [("Error: Node info not loaded", "")]

    choices = []
    for class_type, info in all_node_info.items():
        if "ControlNet Preprocessors" in info.get("category", ""):
            display_name = info.get("display_name", class_type)
            choices.append((display_name, class_type))
    
    choices.sort(key=lambda x: x[0])
    return choices

def update_param_visibility(preprocessor_name):
    all_updates = []
    
    if not preprocessor_name:
        for _ in range(MAX_DYNAMIC_CONTROLS * 3):
            all_updates.append(gr.update(visible=False))
        return tuple(all_updates)

    node_info = node_info_manager.get_node_info(preprocessor_name)
    if not node_info:
        for _ in range(MAX_DYNAMIC_CONTROLS * 3):
            all_updates.append(gr.update(visible=False))
        return tuple(all_updates)

    params = node_info.get("input", {}).get("optional", {})
    
    sliders_params = []
    combos_params = []
    checkboxes_params = []

    for name, details in params.items():
        if name in ["resolution", "image"]: continue
        
        param_type = details[0]
        is_bool_combo = isinstance(param_type, list) and set(s.lower() for s in param_type) == {'enable', 'disable'}

        if isinstance(param_type, str) and param_type.upper() in ["INT", "FLOAT"]:
            sliders_params.append({"name": name, "details": details})
        elif isinstance(param_type, list) and not is_bool_combo:
            combos_params.append({"name": name, "details": details})
        elif is_bool_combo:
            checkboxes_params.append({"name": name, "details": details})

    for i in range(MAX_DYNAMIC_CONTROLS):
        if i < len(sliders_params):
            param = sliders_params[i]
            config = param["details"][1] if len(param["details"]) > 1 else {}
            all_updates.append(gr.update(
                label=param["name"],
                minimum=config.get("min"),
                maximum=config.get("max"),
                step=config.get("step"),
                value=config.get("default"),
                visible=True
            ))
        else:
            all_updates.append(gr.update(visible=False))

    for i in range(MAX_DYNAMIC_CONTROLS):
        if i < len(combos_params):
            param = combos_params[i]
            config = param["details"][1] if len(param["details"]) > 1 else {}
            all_updates.append(gr.update(
                label=param["name"],
                choices=param["details"][0],
                value=config.get("default"),
                visible=True
            ))
        else:
            all_updates.append(gr.update(visible=False))

    for i in range(MAX_DYNAMIC_CONTROLS):
        if i < len(checkboxes_params):
            param = checkboxes_params[i]
            config = param["details"][1] if len(param["details"]) > 1 else {}
            default_val = config.get("default", "disable").lower() == "enable"
            all_updates.append(gr.update(
                label=param["name"],
                value=default_val,
                visible=True
            ))
        else:
            all_updates.append(gr.update(visible=False))
            
    return tuple(all_updates)

def create_ui():
    components = {}
    all_preprocessors = get_preprocessor_choices()
    with gr.Column():
        gr.Markdown("## ControlNet Auxiliary Preprocessors")
        gr.Markdown("💡 **Tip:** Upload an image or video and select a preprocessor to generate a control map. The resolution will be automatically determined based on the input.")
        with gr.Row():
            with gr.Column(scale=1):
                components['input_type'] = gr.Radio(["Image", "Video"], label="Input Type", value="Image")
                components['input_image'] = gr.Image(type="pil", label="Input Image", visible=True)
                components['input_video'] = gr.Video(label="Input Video", visible=False)
                default_choice = "CannyEdgePreprocessor" if any(p[1] == "CannyEdgePreprocessor" for p in all_preprocessors) else (all_preprocessors[0][1] if all_preprocessors else None)
                components['preprocessor_name'] = gr.Dropdown(label="Preprocessor", choices=all_preprocessors, value=default_choice)
                
                param_sliders, param_combos, param_checkboxes = [], [], []
                with gr.Accordion("Preprocessor Parameters", open=True):
                    for i in range(MAX_DYNAMIC_CONTROLS):
                        slider = gr.Slider(visible=False, label=f"dyn_slider_{i}", interactive=True)
                        param_sliders.append(slider)
                    for i in range(MAX_DYNAMIC_CONTROLS):
                        combo = gr.Dropdown(visible=False, label=f"dyn_combo_{i}", interactive=True)
                        param_combos.append(combo)
                    for i in range(MAX_DYNAMIC_CONTROLS):
                        checkbox = gr.Checkbox(visible=False, label=f"dyn_checkbox_{i}", interactive=True)
                        param_checkboxes.append(checkbox)

                components['param_sliders_list'] = param_sliders
                components['param_combos_list'] = param_combos
                components['param_checkboxes_list'] = param_checkboxes

            with gr.Column(scale=1):
                components['output_gallery'] = gr.Gallery(label="Image Output", show_label=False, object_fit="contain", height=488, visible=True, interactive=False)
                components['output_video'] = gr.Video(label="Video Output", show_label=False, height=488, visible=False, interactive=False)
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])

    return components

def get_main_output_components(components: dict):
    return [
        components['output_gallery'], 
        components['output_video'],
        components['run_button']
    ]
    
def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    def update_input_visibility(choice):
        is_image = choice == "Image"
        return {
            components['input_image']: gr.update(visible=is_image),
            components['input_video']: gr.update(visible=not is_image),
            components['output_gallery']: gr.update(visible=is_image),
            components['output_video']: gr.update(visible=not is_image),
        }
    components['input_type'].change(fn=update_input_visibility, inputs=[components['input_type']], outputs=list(update_input_visibility("Image").keys()), show_api=False)
    
    all_dynamic_outputs = components['param_sliders_list'] + components['param_combos_list'] + components['param_checkboxes_list']
    
    components['preprocessor_name'].change(
        fn=update_param_visibility, 
        inputs=[components['preprocessor_name']], 
        outputs=all_dynamic_outputs,
        show_progress=False,
        show_api=False
    )
    demo.load(
        fn=update_param_visibility,
        inputs=[components['preprocessor_name']],
        outputs=all_dynamic_outputs,
        show_api=False
    )

run_generation = create_simple_run_generation(
    process_inputs,
    lambda status, files: (status, files, files)
)