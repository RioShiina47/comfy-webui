import gradio as gr
import os
import random
from PIL import Image
import numpy as np

from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from module.image_gen.sd_shared import create_mask_from_layer, save_temp_image_from_pil
from core.workflow_utils import get_filename_prefix

UI_INFO = { 
    "workflow_recipe": "flux_fill_inpaint_recipe.yaml",
    "main_tab": "ImageEdit", 
    "sub_tab": "FLUX-Fill-Dev Inpaint",
    "run_button_text": "🎨 Fill Image" 
}
PREFIX = "flux_fill"

def save_temp_image_from_pil_local(img, suffix):
    if not isinstance(img, Image.Image): return None
    filename = f"temp_{PREFIX}_{suffix}_{random.randint(1000, 9999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    return os.path.basename(filepath)

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
                    height=460
                )
            components[key('editor_column')] = editor_column
            
            with gr.Column(scale=1) as params_column:
                components[key('positive_prompt')] = gr.Textbox(label="Prompt", lines=4, placeholder="Describe what to fill the masked area with...", interactive=True)
                components[key('negative_prompt')] = gr.Textbox(label="Negative Prompt", lines=4, interactive=True)
                
                with gr.Row():
                    components[key('seed')] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0, interactive=True, scale=1)
                    components[key('guidance')] = gr.Slider(label="Guidance", minimum=1, maximum=50, step=1, value=30, interactive=True)
                
                components[key('output_gallery')] = gr.Gallery(
                    label="Result", show_label=False, object_fit="contain", height=280
                )
            components[key('params_column')] = params_column
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
        components[key('run_button')] = components['run_button']

    components[key('steps')] = gr.State(25)
    components[key('cfg')] = gr.State(1.0)
    components[key('sampler_name')] = gr.State("euler")
    components[key('scheduler')] = gr.State("simple")
    
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

    def toggle_fullscreen_view(view_mode):
        is_fullscreen = (view_mode == "Fullscreen View")
        other_elements_visible = not is_fullscreen
        editor_height = 800 if is_fullscreen else 460
        
        updates = {
            params_column: gr.update(visible=other_elements_visible),
            run_button: gr.update(visible=other_elements_visible),
            image_editor: gr.update(height=editor_height)
        }
        return updates

    output_components_for_toggle = [params_column, run_button, image_editor]
    
    view_mode_radio.change(
        fn=toggle_fullscreen_view,
        inputs=[view_mode_radio],
        outputs=output_components_for_toggle,
        show_progress=False,
        show_api=False
    )


def process_inputs(ui_values, seed_override=None):
    vals = {k.replace(f'{PREFIX}_', ''): v for k, v in ui_values.items() if isinstance(k, str) and k.startswith(PREFIX)}
    
    img_dict = vals.get('input_image_dict')
    mask_img = create_mask_from_layer(img_dict)
    if not img_dict or img_dict.get('background') is None or mask_img is None:
        raise gr.Error("Input image and a drawn mask are required.")
    
    background_img = img_dict['background'].convert("RGBA")
    _, _, _, mask_alpha = mask_img.split()
    background_alpha_np = np.array(background_img.split()[-1])
    mask_alpha_np = np.array(mask_alpha)
    new_alpha_np = np.minimum(background_alpha_np, 255 - mask_alpha_np)
    new_alpha_pil = Image.fromarray(new_alpha_np, mode='L')
    
    r, g, b, _ = background_img.split()
    composite_image = Image.merge('RGBA', [r, g, b, new_alpha_pil])

    vals['input_image'] = save_temp_image_from_pil_local(composite_image, "fill_composite")

    if not vals.get('positive_prompt') or not vals.get('positive_prompt').strip():
        vals['positive_prompt'] = ' '

    recipe_path = UI_INFO["workflow_recipe"]
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    
    seed = int(vals.get('seed', -1))
    vals['seed'] = seed_override if seed_override is not None else (random.randint(0, 2**32 - 1) if seed == -1 else seed)
    vals['filename_prefix'] = get_filename_prefix()

    workflow = assembler.assemble(vals)
    return workflow, {"extra_pnginfo": {"workflow": ""}}

def run_generation(ui_values):
    from core.comfy_api import run_workflow_and_get_output
    import traceback

    final_files = []
    try:
        yield ("Status: Preparing...", None)
        
        workflow, extra_data = process_inputs(ui_values)
        workflow_package = (workflow, extra_data)
        
        for status, output_files in run_workflow_and_get_output(workflow_package):
            if output_files and isinstance(output_files, list):
                final_files = output_files
            
            yield (status, final_files)

    except Exception as e:
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return

    yield ("Status: Loaded successfully!", final_files)