import gradio as gr
import traceback
import math

from .flux2_dev_logic import process_inputs, MAX_REF_IMAGES, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation

UI_INFO = {
    "workflow_recipe": "flux2_dev_recipe.yaml",
    "main_tab": "ImageEdit",
    "sub_tab": "FLUX.2-dev",
    "run_button_text": "🎨 Generate with FLUX.2"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## FLUX.2-dev Text-to-Image Generation")
        gr.Markdown("💡 **Tip:** Provide a prompt to generate an image. Optionally, add reference images to guide the generation.")
        
        with gr.Row():
            components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="e.g., A photo of a cat wearing a wizard hat.")
        
        with gr.Row():
            with gr.Column(scale=1):
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
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=20, step=1, value=1)
            
            with gr.Column(scale=1):
                components['output_gallery'] = gr.Gallery(label="Result", show_label=False, object_fit="contain", height=388)

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

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)