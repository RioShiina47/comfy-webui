import gradio as gr
import traceback
from .depth_anything_tensorrt_logic import process_inputs, ENGINE_CHOICES
from core.utils import create_simple_run_generation

UI_INFO = {
    "main_tab": "Tools",
    "sub_tab": "Depth-Anything-Tensorrt",
    "run_button_text": "🕹️ Run Depth Preprocessor"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Depth Anything (TensorRT)")
        gr.Markdown("💡 **Tip:** Upload an image or video to generate a depth map using a TensorRT-accelerated model.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_type'] = gr.Radio(["Image", "Video"], label="Input Type", value="Image")
                components['input_image'] = gr.Image(type="pil", label="Input Image", visible=True)
                components['input_video'] = gr.Video(label="Input Video", visible=False)
                
                components['engine'] = gr.Dropdown(
                    label="Engine File", 
                    choices=ENGINE_CHOICES, 
                    value=ENGINE_CHOICES[0]
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### Result")
                components['output_gallery'] = gr.Gallery(label="Image Output", show_label=False, object_fit="contain", height=488, visible=True, interactive=False)
                components['output_video'] = gr.Video(label="Result Video", show_label=False, visible=False, interactive=False)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_gallery'], components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    def update_input_visibility(choice):
        is_image = choice == "Image"
        return {
            components['input_image']: gr.update(visible=is_image),
            components['input_video']: gr.update(visible=not is_image),
            components['output_gallery']: gr.update(visible=is_image),
            components['output_video']: gr.update(visible=not is_image),
        }
    components['input_type'].change(fn=update_input_visibility, inputs=[components['input_type']], outputs=list(update_input_visibility("Image").keys()), show_api=False)

run_generation = create_simple_run_generation(
    process_inputs,
    lambda status, files: (status, files, files)
)