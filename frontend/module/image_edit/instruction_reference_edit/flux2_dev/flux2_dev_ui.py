import gradio as gr
import traceback
import math

from .flux2_dev_logic import process_inputs, MAX_REF_IMAGES
from core.comfy_api import run_workflow_and_get_output

UI_INFO = {
    "workflow_recipe": "flux2_dev_recipe.yaml",
    "main_tab": "ImageEdit",
    "sub_tab": "FLUX.2-dev",
    "run_button_text": "🎨 Generate with FLUX.2"
}

ASPECT_RATIO_PRESETS = {
    "1:1 (Square)": (1024, 1024), 
    "3:2 (Photography)": (1248, 832),
    "2:3 (Photography Portrait)": (832, 1248),
    "16:9 (Landscape)": (1344, 768), 
    "9:16 (Portrait)": (768, 1344),
    "4:3 (Classic)": (1152, 896), 
    "3:4 (Classic Portrait)": (896, 1152),
}

def update_resolution(ratio_key, megapixels_str):
    base_w, base_h = ASPECT_RATIO_PRESETS.get(ratio_key, (1024, 1024))
    
    mp_map = {"1MP": 1.0, "2MP": 2.0, "4MP": 4.0}
    mp_multiplier = mp_map.get(megapixels_str, 1.0)
    
    target_pixels = mp_multiplier * 1024 * 1024
    
    current_pixels = base_w * base_h
    scale_factor = math.sqrt(target_pixels / current_pixels)
    
    new_width = int(round((base_w * scale_factor) / 8) * 8)
    new_height = int(round((base_h * scale_factor) / 8) * 8)
    
    return new_width, new_height

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## FLUX.2-dev Text-to-Image Generation")
        gr.Markdown("💡 **Tip:** Provide a prompt to generate an image. Optionally, add reference images to guide the generation.")
        
        with gr.Row():
            components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="e.g., A photo of a cat wearing a wizard hat.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['aspect_ratio'] = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=list(ASPECT_RATIO_PRESETS.keys()),
                    value="1:1 (Square)",
                    interactive=True
                )
                components['megapixels'] = gr.Radio(
                    label="Megapixels",
                    choices=["1MP", "2MP", "4MP"],
                    value="1MP",
                    interactive=True
                )
                components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=20, step=1, value=1)
            
            with gr.Column(scale=1):
                components['output_gallery'] = gr.Gallery(label="Result", show_label=False, object_fit="contain", height=400)

        components['steps'] = gr.State(value=20)
        components['guidance'] = gr.State(value=4.0)
        components['batch_size'] = gr.State(value=1)
        components['sampler_name'] = gr.State(value="euler")

        with gr.Accordion("Reference Images (Optional)", open=False):
            ref_image_groups = []
            ref_image_inputs = []
            with gr.Row():
                for i in range(MAX_REF_IMAGES):
                    with gr.Column(visible=False, min_width=160) as img_col:
                        img_comp = gr.Image(type="pil", label=f"Reference Image {i+1}", sources=["upload"], height=150)
                        ref_image_groups.append(img_col)
                        ref_image_inputs.append(img_comp)
            components['ref_image_groups'] = ref_image_groups
            components['ref_image_inputs'] = ref_image_inputs
            
            with gr.Row():
                components['add_ref_button'] = gr.Button("✚ Add Reference")
                components['delete_ref_button'] = gr.Button("➖ Delete Reference", visible=False)
            components['ref_count_state'] = gr.State(0)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_gallery'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    ref_count_state = components['ref_count_state']
    add_ref_btn = components['add_ref_button']
    del_ref_btn = components['delete_ref_button']
    ref_image_groups = components['ref_image_groups']
    ref_image_inputs = components['ref_image_inputs']

    def add_ref_row(count):
        count += 1
        visibility_updates = tuple(gr.update(visible=i < count) for i in range(MAX_REF_IMAGES))
        return (count, gr.update(visible=count < MAX_REF_IMAGES), gr.update(visible=count > 0)) + visibility_updates
    
    def delete_ref_row(count):
        count -= 1
        image_clear_updates = [gr.update()] * MAX_REF_IMAGES
        image_clear_updates[count] = None
        visibility_updates = tuple(gr.update(visible=i < count) for i in range(MAX_REF_IMAGES))
        return (count, gr.update(visible=True), gr.update(visible=count > 0)) + visibility_updates + tuple(image_clear_updates)

    add_ref_outputs = [ref_count_state, add_ref_btn, del_ref_btn] + ref_image_groups
    del_ref_outputs = [ref_count_state, add_ref_btn, del_ref_btn] + ref_image_groups + ref_image_inputs

    add_ref_btn.click(fn=add_ref_row, inputs=[ref_count_state], outputs=add_ref_outputs, show_progress=False, show_api=False)
    del_ref_btn.click(fn=delete_ref_row, inputs=[ref_count_state], outputs=del_ref_outputs, show_progress=False, show_api=False)
    
    aspect_ratio_dropdown = components['aspect_ratio']
    megapixels_radio = components['megapixels']
    
    width_state = gr.State()
    height_state = gr.State()
    components['width'] = width_state
    components['height'] = height_state
    
    resolution_inputs = [aspect_ratio_dropdown, megapixels_radio]
    resolution_outputs = [width_state, height_state]

    aspect_ratio_dropdown.change(
        fn=update_resolution,
        inputs=resolution_inputs,
        outputs=resolution_outputs,
        show_progress=False,
        show_api=False
    )
    megapixels_radio.change(
        fn=update_resolution,
        inputs=resolution_inputs,
        outputs=resolution_outputs,
        show_progress=False,
        show_api=False
    )

def run_generation(ui_values):
    all_output_files = []
    
    try:
        batch_count = int(ui_values.get('batch_count', 1))
        original_seed = int(ui_values.get('seed', -1))
        
        selected_ratio = ui_values.get('aspect_ratio', "1:1 (Square)")
        megapixels_str = ui_values.get('megapixels', '1MP')
        width, height = update_resolution(selected_ratio, megapixels_str)
        ui_values['width'] = width
        ui_values['height'] = height

        for i in range(batch_count):
            current_seed = original_seed + i if original_seed != -1 else None
            batch_msg = f" (Batch {i + 1}/{batch_count})" if batch_count > 1 else ""
            
            yield (f"Status: Preparing{batch_msg}...", None)
            
            workflow, extra_data = process_inputs(ui_values, seed_override=current_seed)
            workflow_package = (workflow, extra_data)
            
            for status, output_path in run_workflow_and_get_output(workflow_package):
                status_msg = f"Status: {status.replace('Status: ', '')}{batch_msg}"
                
                if output_path and isinstance(output_path, list):
                    all_output_files.extend(p for p in output_path if p not in all_output_files)

                yield (status_msg, all_output_files)

    except Exception as e:
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return

    yield ("Status: Loaded successfully!", all_output_files)