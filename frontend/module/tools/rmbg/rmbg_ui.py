import gradio as gr
import traceback
from .rmbg_logic import process_inputs
from core.comfy_api import run_workflow_and_get_output

UI_INFO = {
    "workflow_recipe": "rmbg_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "RMBG",
    "run_button_text": "🎨 Remove Background"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Remove Background (RMBG)")
        gr.Markdown("💡 **Tip:** Upload an image or video to remove its background. You can choose different models and fine-tune the mask.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_type'] = gr.Radio(["Image", "Video"], label="Input Type", value="Image")
                components['input_image'] = gr.Image(type="pil", label="Input Image", visible=True)
                components['input_video'] = gr.Video(label="Input Video", visible=False)
                
                with gr.Accordion("Advanced Settings", open=True):
                    components['model'] = gr.Dropdown(
                        label="Model", 
                        choices=["RMBG-2.0", "INSPYRENET", "BEN", "BEN2"], 
                        value="RMBG-2.0"
                    )
                    components['process_res'] = gr.Slider(label="Processing Resolution", minimum=256, maximum=2048, step=64, value=1024)
                    components['sensitivity'] = gr.Slider(label="Sensitivity", minimum=0, maximum=10, step=1, value=1)
                    components['mask_blur'] = gr.Slider(label="Mask Blur", minimum=0, maximum=50, step=1, value=0)
                    components['mask_offset'] = gr.Slider(label="Mask Offset", minimum=-50, maximum=50, step=1, value=0)
                    components['refine_foreground'] = gr.Checkbox(label="Refine Foreground", value=False)
                    components['invert_output'] = gr.Checkbox(label="Invert Output", value=False)
                    components['background'] = gr.Radio(
                        label="Background Type",
                        choices=["Alpha", "Solid Color"],
                        value="Alpha"
                    )
                    components['background_color'] = gr.ColorPicker(label="Background Color", value="#222222")

            with gr.Column(scale=1):
                gr.Markdown("### Result")
                components['output_gallery'] = gr.Gallery(label="Image Output", show_label=False, object_fit="contain", height=488, visible=True, interactive=False, columns=2, preview=True)
                components['output_video_result'] = gr.Video(label="Result Video", show_label=True, visible=False, interactive=False)
                components['output_video_mask'] = gr.Video(label="Mask Video", show_label=True, visible=False, interactive=False)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_gallery'], components['output_video_result'], components['output_video_mask'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    def update_input_visibility(choice):
        is_image = choice == "Image"
        return {
            components['input_image']: gr.update(visible=is_image),
            components['input_video']: gr.update(visible=not is_image),
            components['output_gallery']: gr.update(visible=is_image),
            components['output_video_result']: gr.update(visible=not is_image),
            components['output_video_mask']: gr.update(visible=not is_image),
        }
    components['input_type'].change(fn=update_input_visibility, inputs=[components['input_type']], outputs=list(update_input_visibility("Image").keys()), show_api=False)

def run_generation(ui_values):
    final_files = []
    try:
        yield ("Status: Preparing...", gr.update(), gr.update(), gr.update())
        
        workflow, extra_data = process_inputs(ui_values)
        workflow_package = (workflow, extra_data)
        
        for status, output_files in run_workflow_and_get_output(workflow_package):
            if output_files and isinstance(output_files, list):
                new_files = [f for f in output_files if f not in final_files]
                if new_files:
                    final_files.extend(new_files)
            
            result_files = sorted([f for f in final_files if "result" in os.path.basename(f)])
            mask_files = sorted([f for f in final_files if "mask" in os.path.basename(f)])
            
            is_video = ui_values.get('input_type') == "Video"
            
            gallery_update = result_files + mask_files if not is_video else gr.update()
            video_result_update = result_files[0] if is_video and result_files else gr.update()
            video_mask_update = mask_files[0] if is_video and mask_files else gr.update()

            yield (status, gallery_update, video_result_update, video_mask_update)

    except Exception as e:
        traceback.print_exc()
        yield (f"Error: {e}", gr.update(), gr.update(), gr.update())
        return

    result_files = sorted([f for f in final_files if "result" in os.path.basename(f)])
    mask_files = sorted([f for f in final_files if "mask" in os.path.basename(f)])
    is_video = ui_values.get('input_type') == "Video"
    gallery_update = result_files + mask_files if not is_video else gr.update()
    video_result_update = result_files[0] if is_video and result_files else gr.update()
    video_mask_update = mask_files[0] if is_video and mask_files else gr.update()
    
    yield ("Status: Loaded successfully!", gallery_update, video_result_update, video_mask_update)