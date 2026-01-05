import gradio as gr
import traceback

from .flux_fill_inpaint_logic import process_inputs
from core.utils import create_simple_run_generation

UI_INFO = { 
    "main_tab": "ImageEdit", 
    "sub_tab": "FLUX-Fill-Dev Inpaint",
    "run_button_text": "🎨 Fill Image" 
}
PREFIX = "flux_fill"

def create_ui():
    components = {}
    key = lambda name: f"{PREFIX}_{name}"
    
    with gr.Column():
        gr.Markdown("## FLUX Fill (Inpainting)")
        gr.Markdown("💡 **Tip:** Upload an image, draw a mask over the area you want to replace, and describe what you want to fill it with.")
        
        with gr.Row():
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
            
            with gr.Column(scale=2) as params_column:
                components[key('positive_prompt')] = gr.Textbox(label="Prompt", lines=6, placeholder="Describe what to fill the masked area with...", interactive=True)
                components[key('negative_prompt')] = gr.Textbox(label="Negative Prompt", lines=6, interactive=True)
            components[key('params_column')] = params_column
        
        with gr.Row() as seed_gallery_row:
            with gr.Column(scale=1):
                components[key('seed')] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0, interactive=True)
            with gr.Column(scale=1):
                components[key('output_gallery')] = gr.Gallery(
                    label="Result", show_label=False, object_fit="contain", height=392
                )
        components[key('seed_gallery_row')] = seed_gallery_row
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
        components[key('run_button')] = components['run_button']
    
    return components

def get_main_output_components(components: dict):
    return [components[f'{PREFIX}_output_gallery'], components[f'{PREFIX}_run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    key = lambda name: f"{PREFIX}_{name}"

    view_mode_radio = components[key('view_mode')]
    editor_column = components[key('editor_column')]
    params_column = components[key('params_column')]
    run_button = components[key('run_button')]
    image_editor = components[key('input_image_dict')]
    seed_gallery_row = components[key('seed_gallery_row')]

    def toggle_fullscreen_view(view_mode):
        is_fullscreen = (view_mode == "Fullscreen View")
        other_elements_visible = not is_fullscreen
        editor_height = 800 if is_fullscreen else 272
        
        updates = {
            params_column: gr.update(visible=other_elements_visible),
            run_button: gr.update(visible=other_elements_visible),
            image_editor: gr.update(height=editor_height),
            seed_gallery_row: gr.update(visible=other_elements_visible)
        }
        return updates

    output_components_for_toggle = [params_column, run_button, image_editor, seed_gallery_row]
    
    view_mode_radio.change(
        fn=toggle_fullscreen_view,
        inputs=[view_mode_radio],
        outputs=output_components_for_toggle,
        show_progress=False,
        show_api=False
    )

run_generation = create_simple_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)