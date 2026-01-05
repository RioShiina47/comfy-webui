import gradio as gr
import traceback

from .bytedance_uso_logic import process_inputs, ASPECT_RATIO_PRESETS
from core.utils import create_batched_run_generation
from core.shared_ui import create_lora_ui, register_ui_chain_events

UI_INFO = {
    "workflow_recipe": "bytedance_uso_recipe.yaml",
    "main_tab": "ImageEdit",
    "sub_tab": "ByteDance USO",
    "run_button_text": "🎨 Generate with USO"
}
PREFIX = "uso"
MAX_CONTENT_REFERENCES = 5
MAX_STYLE_REFERENCES = 5

def create_ui():
    components = {}
    key = lambda name: f"{PREFIX}_{name}"
    
    with gr.Column():
        gr.Markdown("## ByteDance USO")
        gr.Markdown("💡 **Tip:** This model is for universal subject-object generation. Add content/style reference images and describe the desired output.")
        
        with gr.Column(scale=1):
            components[key('positive_prompt')] = gr.Textbox(label="Prompt", lines=3, placeholder="e.g., a corgi programming on a macbook")
            components[key('negative_prompt')] = gr.Textbox(label="Negative Prompt", lines=3)
        
        with gr.Row():
            with gr.Column(scale=1):
                default_ratio = list(ASPECT_RATIO_PRESETS.keys())[0]
                with gr.Row():
                    components[key('aspect_ratio')] = gr.Dropdown(
                        label="Aspect Ratio", 
                        choices=list(ASPECT_RATIO_PRESETS.keys()), 
                        value=default_ratio,
                        interactive=True
                    )
                with gr.Row():
                    components[key('seed')] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                with gr.Row():
                    components[key('batch_size')] = gr.Slider(label="Batch Size", minimum=1, maximum=4, step=1, value=1)
                with gr.Row():
                    components[key('batch_count')] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)
            
            with gr.Column(scale=1):
                components[key('output_gallery')] = gr.Gallery(label="Result", show_label=False, object_fit="contain", height=377)

        create_lora_ui(components, PREFIX)

        with gr.Accordion("Content Reference Settings", open=False) as ref_accordion:
            components[key('reference_accordion')] = ref_accordion
            ref_rows, ref_images, ref_guidances = [], [], []
            components.update({
                key('reference_rows'): ref_rows,
                key('reference_images'): ref_images,
                key('reference_guidances'): ref_guidances
            })
            with gr.Row():
                for i in range(MAX_CONTENT_REFERENCES):
                    with gr.Column(visible=(i < 1), scale=1, min_width=160) as col:
                        ref_images.append(gr.Image(label=f"Content Image {i+1}", type="pil", sources="upload", height=150))
                        ref_guidances.append(gr.Slider(label="Guidance Strength", minimum=0.0, maximum=10.0, step=0.1, value=3.5, interactive=True, visible=False))
                        ref_rows.append(col)
            with gr.Row():
                components[key('add_reference_button')] = gr.Button("✚ Add Content Image")
                components[key('delete_reference_button')] = gr.Button("➖ Delete Content Image", visible=False)
            components[key('reference_count_state')] = gr.State(1)

        with gr.Accordion("Style Reference Settings", open=False) as style_ref_accordion:
            components[key('style_reference_accordion')] = style_ref_accordion
            style_rows, style_images = [], []
            components.update({
                key('style_reference_rows'): style_rows,
                key('style_reference_images'): style_images,
            })
            with gr.Row():
                for i in range(MAX_STYLE_REFERENCES):
                    with gr.Column(visible=(i < 1), scale=1, min_width=160) as col:
                        style_images.append(gr.Image(label=f"Style Image {i+1}", type="pil", sources="upload", height=150))
                        style_rows.append(col)
            with gr.Row():
                components[key('add_style_reference_button')] = gr.Button("✚ Add Style Image")
                components[key('delete_style_reference_button')] = gr.Button("➖ Delete Style Image", visible=False)
            components[key('style_reference_count_state')] = gr.State(1)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
        components[key('run_button')] = components['run_button']

    return components

def get_main_output_components(components: dict):
    return [components[f'{PREFIX}_output_gallery'], components[f'{PREFIX}_run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    key = lambda name: f"{PREFIX}_{name}"
    register_ui_chain_events(components, PREFIX)

    ref_count = components[key('reference_count_state')]
    add_ref_btn = components[key('add_reference_button')]
    del_ref_btn = components[key('delete_reference_button')]
    ref_rows = components[key('reference_rows')]
    ref_images = components[key('reference_images')]
    ref_guidances = components[key('reference_guidances')]
    
    def add_ref_row(count):
        count += 1
        return (count, gr.update(visible=count < MAX_CONTENT_REFERENCES), gr.update(visible=count > 1)) + tuple(gr.update(visible=i < count) for i in range(MAX_CONTENT_REFERENCES))
    
    def delete_ref_row(count):
        count -= 1
        image_updates = [gr.update()] * MAX_CONTENT_REFERENCES
        guidance_updates = [gr.update()] * MAX_CONTENT_REFERENCES
        image_updates[count] = None
        guidance_updates[count] = 3.5
        row_updates = tuple(gr.update(visible=i < count) for i in range(MAX_CONTENT_REFERENCES))
        return (count, gr.update(visible=True), gr.update(visible=count > 1)) + row_updates + tuple(image_updates) + tuple(guidance_updates)

    add_ref_outputs = [ref_count, add_ref_btn, del_ref_btn] + ref_rows
    del_ref_outputs = [ref_count, add_ref_btn, del_ref_btn] + ref_rows + ref_images + ref_guidances
    
    add_ref_btn.click(fn=add_ref_row, inputs=[ref_count], outputs=add_ref_outputs, show_progress=False, show_api=False)
    del_ref_btn.click(fn=delete_ref_row, inputs=[ref_count], outputs=del_ref_outputs, show_progress=False, show_api=False)
    
    style_ref_count = components[key('style_reference_count_state')]
    add_style_ref_btn = components[key('add_style_reference_button')]
    del_style_ref_btn = components[key('delete_style_reference_button')]
    style_ref_rows = components[key('style_reference_rows')]
    style_ref_images = components[key('style_reference_images')]
    
    def add_style_ref_row(count):
        count += 1
        return (count, gr.update(visible=count < MAX_STYLE_REFERENCES), gr.update(visible=count > 1)) + tuple(gr.update(visible=i < count) for i in range(MAX_STYLE_REFERENCES))
    
    def delete_style_ref_row(count):
        count -= 1
        image_updates = [gr.update()] * MAX_STYLE_REFERENCES
        image_updates[count] = None
        row_updates = tuple(gr.update(visible=i < count) for i in range(MAX_STYLE_REFERENCES))
        return (count, gr.update(visible=True), gr.update(visible=count > 1)) + row_updates + tuple(image_updates)

    add_style_ref_outputs = [style_ref_count, add_style_ref_btn, del_style_ref_btn] + style_ref_rows
    del_style_ref_outputs = [style_ref_count, add_style_ref_btn, del_style_ref_btn] + style_ref_rows + style_ref_images
    
    add_style_ref_btn.click(fn=add_style_ref_row, inputs=[style_ref_count], outputs=add_style_ref_outputs, show_progress=False, show_api=False)
    del_style_ref_btn.click(fn=delete_style_ref_row, inputs=[style_ref_count], outputs=del_style_ref_outputs, show_progress=False, show_api=False)

    ref_accordion = components[key('reference_accordion')]
    style_ref_accordion = components[key('style_reference_accordion')]

    def on_accordion_expand(*images):
        return [gr.update() for _ in images]

    ref_accordion.expand(
        fn=on_accordion_expand,
        inputs=ref_images,
        outputs=ref_images,
        show_progress=False,
        show_api=False
    )
    
    style_ref_accordion.expand(
        fn=on_accordion_expand,
        inputs=style_ref_images,
        outputs=style_ref_images,
        show_progress=False,
        show_api=False
    )

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)