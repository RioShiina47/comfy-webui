import gradio as gr
from .qwen_image_layered_logic import process_inputs, PREFIX
from core.utils import create_batched_run_generation

UI_INFO = {
    "main_tab": "ImageEdit",
    "sub_tab": "Qwen-Image Layered",
    "run_button_text": "Generate Layers"
}

def create_ui():
    components = {}
    key = lambda name: f"{PREFIX}_{name}"
    
    with gr.Column():
        gr.Markdown("## Qwen-Image Layered Generation")
        gr.Markdown("💡 **Tip:** Upload an image to generate multiple related layers or variations based on it.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components[key('input_image')] = gr.Image(type="pil", label="Input Image", height=255)
            
            with gr.Column(scale=2):
                components[key('positive_prompt')] = gr.Textbox(
                    label="Positive Prompt (Optional)", 
                    lines=3
                )
                components[key('negative_prompt')] = gr.Textbox(
                    label="Negative Prompt (Optional)", 
                    lines=3
                )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    components[key('layers')] = gr.Slider(label="Number of Layers", minimum=2, maximum=10, step=1, value=2)
                with gr.Row():
                    components[key('mode')] = gr.Radio(
                        ["Fast", "Normal"], 
                        label="Mode", 
                        value="Fast", 
                    info="Fast: 20 steps, CFG 2.5. Normal: 50 steps, CFG 4.0"
                )
                with gr.Row():
                    components[key('seed')] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                with gr.Row():
                    components[key('batch_count')] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)
            
            with gr.Column(scale=1):
                components[key('output_gallery')] = gr.Gallery(label="Result", show_label=False, object_fit="contain", height=392)
        
        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
        components[key('run_button')] = components['run_button']
                
    return components

def get_main_output_components(components: dict):
    key = lambda name: f"{PREFIX}_{name}"
    return [components[key('output_gallery')], components[key('run_button')]]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

run_generation = create_batched_run_generation(
    process_inputs,
    lambda status, files: (status, files)
)