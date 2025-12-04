import gradio as gr
import os
import tempfile
from .ComfyUI_Upscaler_Tensorrt_logic import process_inputs, extract_audio_ffmpeg, merge_video_audio_ffmpeg
from core.comfy_api import run_workflow_and_get_output

UI_INFO = {
    "main_tab": "Tools",
    "sub_tab": "Upscaler-Tensorrt",
    "run_button_text": "🚀 Upscale"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Upscaler (TensorRT)")
        gr.Markdown("💡 **Tip:** Upload an image or video to upscale. Inputs larger than 1920x1920 will be downscaled first to maintain compatibility.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_type'] = gr.Radio(["Image", "Video"], label="Input Type", value="Image")
                components['input_image'] = gr.Image(type="pil", label="Input Image", visible=True)
                components['input_video'] = gr.Video(label="Input Video", visible=False)
                components['upscaler_model'] = gr.State(value="4x-UltraSharpV2_Lite")
                components['resize_to'] = gr.Dropdown(
                    label="Resize Output To", 
                    choices=["none", "HD", "FHD", "2k", "4k", "2x", "3x"], 
                    value="none",
                    info="Resize the final upscaled output to a specific size or multiplier."
                )
                components['precision'] = gr.State(value="fp16")
                
            with gr.Column(scale=1):
                components['output_gallery'] = gr.Gallery(label="Image Output", show_label=False, object_fit="contain", height=508, visible=True, interactive=False)
                components['output_video'] = gr.Video(label="Video Output", show_label=False, height=488, visible=False, interactive=False)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_gallery'], components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    def update_input_visibility(choice):
        is_image = choice == "Image"
        resize_to_default = "none" if is_image else "4k"
        return {
            components['input_image']: gr.update(visible=is_image),
            components['input_video']: gr.update(visible=not is_image),
            components['output_gallery']: gr.update(visible=is_image),
            components['output_video']: gr.update(visible=not is_image),
            components['resize_to']: gr.update(value=resize_to_default)
        }
    components['input_type'].change(fn=update_input_visibility, inputs=[components['input_type']], outputs=list(update_input_visibility("Image").keys()), show_api=False)

def run_generation(ui_values):
    all_output_files = []
    is_video = ui_values.get('input_type') == "Video"
    input_file_path = ui_values.get('input_video') if is_video else ui_values.get('input_image')
    if not input_file_path:
        raise gr.Error("Input file is missing.")

    temp_audio_path = None
    silent_output_path = None
    final_output_path = None
    
    try:
        yield ("Status: Preparing...", gr.update(), gr.update())
        
        if is_video:
            yield ("Status: Extracting audio...", gr.update(), gr.update())
            temp_audio_path = extract_audio_ffmpeg(input_file_path)

        workflow, extra_data = process_inputs(ui_values)
        workflow_package = (workflow, extra_data)
        
        for status, output_files in run_workflow_and_get_output(workflow_package):
            if output_files and isinstance(output_files, list):
                all_output_files = output_files
            
            gallery_update = all_output_files if not is_video and all_output_files else gr.update()
            video_update = all_output_files[0] if is_video and all_output_files else gr.update()
            yield (status, gallery_update, video_update)

        if is_video and all_output_files:
            silent_output_path = all_output_files[0]
            yield ("Status: Merging audio...", gr.update(), silent_output_path)
            final_output_path = merge_video_audio_ffmpeg(silent_output_path, temp_audio_path)
            yield ("Status: Merging complete!", gr.update(), final_output_path)
        else:
            final_output_path = all_output_files[0] if all_output_files else None

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (f"Error: {e}", gr.update(), gr.update())
        return

    finally:
        print("Upscale task finished. Cleaning up temporary files...")
        cleanup_paths = [temp_audio_path]
        if is_video and temp_audio_path and silent_output_path != final_output_path:
            cleanup_paths.append(silent_output_path)
            
        for path in cleanup_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Removed temp file: {path}")
                except Exception as e:
                    print(f"Error removing temp file {path}: {e}")

        gallery_update = all_output_files if not is_video and all_output_files else gr.update()
        video_update = final_output_path if is_video else gr.update()
        yield ("Status: Loaded successfully!", gallery_update, video_update)