import gradio as gr
import traceback
import math

from .flux2_logic import process_inputs, MAX_REF_IMAGES, ASPECT_RATIO_PRESETS, PREFIX, MODEL_CONFIG
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "flux2_recipe.yaml",
    "main_tab": "ImageEdit",
    "sub_tab": "FLUX.2",
    "run_button_text": "🎨 Generate with FLUX.2"
}

MAX_REF_IMAGES = 9

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## FLUX.2 Image Editing")
        gr.Markdown("💡 **Tip:** Upload an image and provide a text instruction to edit it. You can add more reference images to provide context for the edit.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_image'] = gr.Image(type="pil", label="Input Image", height=255)
            with gr.Column(scale=2):
                components['positive_prompt'] = gr.Textbox(label="Edit Instruction", lines=3, placeholder="e.g., Make it a rainy day.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components['model'] = gr.Dropdown(
                        label="Model",
                        choices=list(MODEL_CONFIG.keys()),
                        value="FLUX.2-klein-4B",
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
                    components['megapixels'] = gr.Radio(
                        label="Megapixels",
                        choices=["1MP", "2MP", "4MP"],
                        value="1MP",
                        interactive=True
                    )
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                with gr.Row():
                    components['batch_size'] = gr.Slider(label="Batch Size", minimum=1, maximum=8, step=1, value=1)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=20, step=1, value=1)
            
            with gr.Column(scale=1):
                components['output_gallery'] = gr.Gallery(label="Result", show_label=False, object_fit="contain", height=493)

        create_lora_ui(components, PREFIX)

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
    register_ui_chain_events(components, PREFIX)

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

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)