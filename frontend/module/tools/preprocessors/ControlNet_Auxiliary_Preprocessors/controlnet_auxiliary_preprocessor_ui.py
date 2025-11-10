import gradio as gr
import os
import random
import shutil
from PIL import Image
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.comfy_api import run_workflow_and_get_output
from core import node_info_manager
from core.workflow_assembler import WorkflowAssembler
from core.workflow_utils import get_filename_prefix

UI_INFO = {
    "workflow_recipe": None,
    "main_tab": "Tools",
    "sub_tab": "ControlNet Auxiliary Preprocessors",
    "run_button_text": "🕹️ Run Preprocessor"
}

MAX_DYNAMIC_CONTROLS = 8

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

def save_temp_file(file_obj, name_prefix: str, is_video=False) -> str:
    if file_obj is None: return None
    if is_video:
        ext = os.path.splitext(file_obj)[1] or ".mp4"
        temp_filename = f"temp_{name_prefix}_{random.randint(1000, 9999)}{ext}"
        save_path = os.path.join(COMFYUI_INPUT_PATH, temp_filename)
        shutil.copy(file_obj, save_path)
    else:
        temp_filename = f"temp_{name_prefix}_{random.randint(1000, 9999)}.png"
        save_path = os.path.join(COMFYUI_INPUT_PATH, temp_filename)
        file_obj.save(save_path, "PNG")
    print(f"Saved temporary input file to: {save_path}")
    return temp_filename

def make_even(n):
    return n if n % 2 == 0 else n + 1

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

def process_inputs(ui_values):
    is_video = ui_values.get('input_type') == "Video"
    preprocessor_name = ui_values.get('preprocessor_name')
    if not preprocessor_name: raise gr.Error("Please select a preprocessor.")

    input_file_obj = ui_values.get('input_video') if is_video else ui_values.get('input_image')
    if input_file_obj is None: raise gr.Error("Please provide an input image or video.")

    metadata = get_media_metadata(input_file_obj, is_video=is_video)
    w, h, fps = metadata['width'], metadata['height'], metadata['fps']
    if w == 0 or h == 0: raise gr.Error("Could not get the dimensions of the input file.")

    node_info = node_info_manager.get_node_info(preprocessor_name)
    resolution_config = node_info.get("input", {}).get("optional", {}).get("resolution", [None, {}])[1]
    final_resolution = resolution_config.get("default", max(w,h))

    recipe_path = "controlnet_base_recipe.yaml"
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)

    local_ui_values = {}
    if is_video:
        local_ui_values['input_video_filename'] = save_temp_file(input_file_obj, "cn_input", is_video=True)
    else:
        local_ui_values['input_image_filename'] = save_temp_file(input_file_obj, "cn_input", is_video=False)
        
    local_ui_values['filename_prefix'] = get_filename_prefix()
    workflow = assembler.assemble(local_ui_values)
    
    preprocessor_id = assembler._get_unique_id()
    preprocessor_node = assembler._get_node_template_from_api(preprocessor_name)
    
    preprocessor_node['_meta']['title'] = node_info.get("display_name", preprocessor_name)
    preprocessor_node['inputs']['resolution'] = final_resolution
    
    params_in_node = node_info.get("input", {}).get("optional", {})
    
    sliders_params, combos_params, checkboxes_params = [], [], []
    for name, details in params_in_node.items():
        if name in ["resolution", "image"]: continue
        param_type = details[0]
        is_bool_combo = isinstance(param_type, list) and set(s.lower() for s in param_type) == {'enable', 'disable'}
        if isinstance(param_type, str) and param_type.upper() in ["INT", "FLOAT"]: sliders_params.append(name)
        elif isinstance(param_type, list) and not is_bool_combo: combos_params.append(name)
        elif is_bool_combo: checkboxes_params.append(name)
        
    for i, name in enumerate(sliders_params):
        if i < MAX_DYNAMIC_CONTROLS:
            preprocessor_node['inputs'][name] = ui_values.get(f'param_slider_{i}')
    for i, name in enumerate(combos_params):
        if i < MAX_DYNAMIC_CONTROLS:
            preprocessor_node['inputs'][name] = ui_values.get(f'param_combo_{i}')
    for i, name in enumerate(checkboxes_params):
        if i < MAX_DYNAMIC_CONTROLS:
            is_enabled = ui_values.get(f'param_checkbox_{i}')
            preprocessor_node['inputs'][name] = "enable" if is_enabled else "disable"

    workflow[preprocessor_id] = preprocessor_node

    input_node_id = assembler.node_map['get_frames'] if is_video else assembler.node_map['load_image']
    workflow[preprocessor_id]['inputs']['image'] = [input_node_id, 0]

    output_types = node_info.get("output", ["IMAGE"])
    for i, out_type in enumerate(output_types):
        if i >= 2: break 

        current_image_source = [preprocessor_id, i]
        
        if out_type == "MASK":
            mask_converter_id = assembler.node_map['mask_to_image_converter']
            workflow[mask_converter_id]['inputs']['mask'] = current_image_source
            current_image_source = [mask_converter_id, 0]
        
        save_node_params = { 'filename_prefix': f"{local_ui_values['filename_prefix']}_out{i+1}" }

        if is_video:
            resize_id, create_id, save_id = [assembler.node_map[f'{name}_{i+1}'] for name in ['resize_frames', 'create_video', 'save_video']]
            workflow[resize_id]['inputs'].update({'image': current_image_source, 'target_width': make_even(w), 'target_height': make_even(h)})
            workflow[create_id]['inputs'].update({'images': [resize_id, 0], 'fps': fps})
            workflow[save_id]['inputs'].update({'video': [create_id, 0], **save_node_params})
        else:
            save_id = assembler.node_map[f'save_image_{i+1}']
            workflow[save_id]['inputs'].update({'images': current_image_source, **save_node_params})
            
    return workflow, None

def run_generation(ui_values):
    original_run_button_text = UI_INFO["run_button_text"]
    
    yield (None, None, None, gr.update(value="Stop", variant="stop"))

    all_output_files = []
    is_video = ui_values.get('input_type') == "Video"

    try:
        workflow, extra_data = process_inputs(ui_values)
        workflow_package = (workflow, extra_data)

        for status, output_path in run_workflow_and_get_output(workflow_package):
            if output_path and isinstance(output_path, list):
                new_files = [f for f in output_path if f not in all_output_files]
                if new_files:
                    all_output_files.extend(new_files)
            
            gallery_update = all_output_files if not is_video and all_output_files else gr.update()
            video_update = all_output_files[-1] if is_video and all_output_files else gr.update()

            yield (status, gallery_update, video_update, gr.update())

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (f"Error: {e}", gr.update(), gr.update(), gr.update(value=original_run_button_text, variant="primary"))

    finally:
        print("ControlNet preprocessing finished.")
        yield ("Status: Ready", gr.update(), gr.update(), gr.update(value=original_run_button_text, variant="primary"))