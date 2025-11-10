import gradio as gr
import random
import os
import yaml
from PIL import Image
from core.workflow_assembler import WorkflowAssembler
from core.comfy_api import run_workflow_and_get_output
from core.config import COMFYUI_INPUT_PATH
from core.workflow_utils import get_filename_prefix

UI_INFO = {
    "workflow_recipe": "flux-kontext-dev_recipe.yaml",
    "main_tab": "ImageEdit",
    "sub_tab": "Flux-Kontext-Dev",
    "run_button_text": "🎨 Edit with Kontext"
}

ASPECT_RATIO_PRESETS = {
    "1:1 (Square)": (1024, 1024), 
    "16:9 (Landscape)": (1344, 768), 
    "9:16 (Portrait)": (768, 1344),
    "4:3 (Classic)": (1152, 896), 
    "3:4 (Classic Portrait)": (896, 1152),
    "3:2 (Photography)": (1216, 832),
    "2:3 (Photography Portrait)": (832, 1216)
}

MAX_REF_IMAGES = 8

def save_temp_image(img: Image.Image, name: str) -> str:
    if not isinstance(img, Image.Image):
        return None
    filename = f"temp_flux_kontext_{name}_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    print(f"Saved temporary image to {filepath}")
    return filename

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Flux-Kontext-Dev Image Editing")
        gr.Markdown("💡 **Tip:** Upload an image and provide a text instruction. Add more images for richer context.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_image'] = gr.Image(type="pil", label="Input Image", sources=["upload"], height=255)
            
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Edit Instruction", lines=3, placeholder="e.g., Make it a rainy day.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)

        with gr.Row():
            with gr.Column(scale=1):
                components['aspect_ratio'] = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=list(ASPECT_RATIO_PRESETS.keys()),
                    value="1:1 (Square)",
                    interactive=True
                )
                components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                components['batch_size'] = gr.Slider(label="Batch Size", minimum=1, maximum=4, step=1, value=1)
                components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=20, step=1, value=1)
            
            with gr.Column(scale=1):
                components['output_gallery'] = gr.Gallery(label="Result", show_label=False, object_fit="contain", height=400)

        components['guidance'] = gr.State(value=2.5)
        components['cfg'] = gr.State(value=1.0)
        components['steps'] = gr.State(value=20)
        components['sampler_name'] = gr.State(value="euler")
        components['scheduler'] = gr.State(value="simple")

        with gr.Accordion("Add More Images", open=False):
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

    add_ref_btn.click(
        fn=add_ref_row,
        inputs=[ref_count_state],
        outputs=add_ref_outputs,
        show_progress=False,
        show_api=False
    )
    
    del_ref_btn.click(
        fn=delete_ref_row,
        inputs=[ref_count_state],
        outputs=del_ref_outputs,
        show_progress=False,
        show_api=False
    )

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    
    main_img = local_ui_values.get('input_image')
    if main_img is None:
        raise gr.Error("Please upload an image to edit.")

    all_images = [main_img]
    num_refs = int(local_ui_values.get('ref_count_state', 0))
    ref_images_list = local_ui_values.get('ref_image_inputs', [])
    
    for i in range(num_refs):
        if i < len(ref_images_list) and ref_images_list[i] is not None:
            all_images.append(ref_images_list[i])
    
    image_filenames = [save_temp_image(img, f"ref_{i}") for i, img in enumerate(all_images)]
    local_ui_values['image_stitch_chain'] = image_filenames
    local_ui_values['input_image'] = None

    selected_ratio = local_ui_values.get('aspect_ratio', "1:1 (Square)")
    width, height = ASPECT_RATIO_PRESETS[selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height

    seed = int(local_ui_values.get('seed', -1))
    if seed_override is not None:
        local_ui_values['seed'] = seed_override
    elif seed == -1:
        local_ui_values['seed'] = random.randint(0, 2**32 - 1)
        
    local_ui_values['filename_prefix'] = get_filename_prefix()

    recipe_path = UI_INFO["workflow_recipe"]
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None

def run_generation(ui_values):
    all_output_files = []
    
    try:
        batch_count = int(ui_values.get('batch_count', 1))
        original_seed = int(ui_values.get('seed', -1))

        for i in range(batch_count):
            current_seed = original_seed + i if original_seed != -1 else None
            batch_msg = f" (Batch {i + 1}/{batch_count})" if batch_count > 1 else ""
            
            yield (f"Status: Preparing{batch_msg}...", None)
            
            workflow, extra_data = process_inputs(ui_values, seed_override=current_seed)
            workflow_package = (workflow, extra_data)
            
            for status, output_path in run_workflow_and_get_output(workflow_package):
                status_msg = f"Status: {status.replace('Status: ', '')}{batch_msg}"
                
                if output_path and isinstance(output_path, list):
                    all_output_files.extend(output_path)

                yield (status_msg, all_output_files)

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return

    yield ("Status: Loaded successfully!", all_output_files)