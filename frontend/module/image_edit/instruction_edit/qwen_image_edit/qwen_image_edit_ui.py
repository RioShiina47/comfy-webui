import gradio as gr
import random
import os
import traceback
from PIL import Image

from .qwen_image_edit_logic import process_inputs_logic
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "main_tab": "ImageEdit",
    "sub_tab": "Qwen-Image-Edit",
    "run_button_text": "🎨 Edit Qwen Image"
}

ASPECT_RATIO_PRESETS = {
    "1:1 (Square)": (1328, 1328), 
    "16:9 (Landscape)": (1664, 928), 
    "9:16 (Portrait)": (928, 1664),
    "4:3 (Classic)": (1472, 1104), 
    "3:4 (Classic Portrait)": (1104, 1472),
    "3:2 (Photography)": (1536, 1024),
    "2:3 (Photography Portrait)": (1024, 1536)
}

MAX_REF_IMAGES = 8
REQUIRED_LORA_DIRS = ["qwen-image-edit"]

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Qwen-Image-Edit Lightning")
        gr.Markdown("💡 **Tip:** Upload a main image and provide an instruction. You can add up to 8 reference images, which will be stitched together as context for the edit.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_image'] = gr.Image(type="pil", label="Input Image", sources=["upload"], height=255)
            
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Edit Instruction", lines=3, placeholder="e.g., Make the cat wear a wizard hat.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['model_version'] = gr.Dropdown(
                        label="Model",
                        choices=["Qwen-Image-Edit-2511", "Qwen-Image-Edit-2509", "Qwen-Image-Edit"],
                        value="Qwen-Image-Edit-2511",
                        interactive=True
                )
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="1:1 (Square)",
                        interactive=True
                )
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                with gr.Row():
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=20, step=1, value=1)
            
            with gr.Column(scale=1):
                components['output_gallery'] = gr.Gallery(label="Result", show_label=False, object_fit="contain", height=393)

        create_lora_ui(components, "qwen_edit", required_lora_dirs=REQUIRED_LORA_DIRS)
        
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
    register_ui_chain_events(components, "qwen_edit")
    
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
    
    selected_ratio = local_ui_values.get('aspect_ratio', "1:1 (Square)")
    width, height = ASPECT_RATIO_PRESETS[selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    return process_inputs_logic(local_ui_values, seed_override)

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)