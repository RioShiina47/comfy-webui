import gradio as gr
import os
import random
from PIL import Image

from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from module.image_gen.sd_shared import save_temp_image_from_pil
from core.workflow_utils import get_filename_prefix

UI_INFO = { 
    "workflow_recipe": "flux_dev_onereward_outpaint_recipe.yaml",
    "main_tab": "ImageEdit", 
    "sub_tab": "Flux.1-Dev-OneReward Outpaint",
    "run_button_text": "🎨 Outpaint with OneReward" 
}
PREFIX = "flux_dev_onereward_outpaint"

def create_ui():
    components = {}
    key = lambda name: f"{PREFIX}_{name}"
    
    with gr.Column():
        gr.Markdown("## FLUX.1-Dev-OneReward (Outpainting)")
        gr.Markdown("💡 **Tip:** Upload an image, use the sliders to define the area to expand, and describe the content for the new area.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components[key('input_image')] = gr.Image(type="pil", label="Input Image", height=255)
            with gr.Column(scale=2):
                components[key('positive_prompt')] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the content of the expanded area...", interactive=True)
                components[key('negative_prompt')] = gr.Textbox(label="Negative Prompt", lines=3, value="", interactive=True)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components[key('left')] = gr.Slider(label="Expand Left", minimum=0, maximum=512, step=64, value=0)
                    components[key('top')] = gr.Slider(label="Expand Up", minimum=0, maximum=512, step=64, value=0)
                with gr.Row():
                    components[key('right')] = gr.Slider(label="Expand Right", minimum=0, maximum=512, step=64, value=256)
                    components[key('bottom')] = gr.Slider(label="Expand Down", minimum=0, maximum=512, step=64, value=0)
                
                components[key('feathering')] = gr.Slider(label="Feathering", minimum=0, maximum=100, step=1, value=0, interactive=True)
                
                with gr.Row():
                    components[key('seed')] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0, interactive=True)
                    components[key('guidance')] = gr.Slider(label="Guidance", minimum=1, maximum=50, step=1, value=30, interactive=True)

            with gr.Column(scale=1):
                components[key('output_gallery')] = gr.Gallery(
                    label="Result", show_label=False, object_fit="contain", height=468
                )
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
        components[key('run_button')] = components['run_button']

    components[key('steps')] = gr.State(20)
    components[key('cfg')] = gr.State(1.0)
    components[key('sampler_name')] = gr.State("euler")
    components[key('scheduler')] = gr.State("normal")
    
    return components

def get_main_output_components(components: dict):
    return [components[f'{PREFIX}_output_gallery'], components[f'{PREFIX}_run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def process_inputs(ui_values, seed_override=None):
    vals = {k.replace(f'{PREFIX}_', ''): v for k, v in ui_values.items() if isinstance(k, str) and k.startswith(PREFIX)}
    
    input_img = vals.get('input_image')
    if input_img is None:
        raise gr.Error("Input image is required for Outpainting.")
    
    vals['input_image'] = save_temp_image_from_pil(input_img, f"{PREFIX}_input")
    
    if not vals.get('positive_prompt') or not vals.get('positive_prompt').strip():
        vals['positive_prompt'] = ' '
        
    if not vals.get('negative_prompt') or not vals.get('negative_prompt').strip():
        vals['negative_prompt'] = ' '

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