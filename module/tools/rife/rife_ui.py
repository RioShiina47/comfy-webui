import gradio as gr
import os
import tempfile
from .rife_logic import process_inputs, extract_audio_ffmpeg, merge_video_audio_ffmpeg
from core.comfy_api import run_workflow_and_get_output

UI_INFO = {
    "workflow_recipe": "rife_recipe.yaml",
    "main_tab": "Tools",
    "sub_tab": "RIFE",
    "run_button_text": "🚀 Interpolate Video"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## RIFE Video Frame Interpolation (TensorRT)")
        gr.Markdown("💡 **Tip:** Upload a video and select the interpolation settings. This uses a pre-built RIFE TensorRT engine (`rife49_ensemble_True_scale_1_sim.engine`) for acceleration.")
        
        components['use_cuda_graph'] = gr.State(True)
        components['keep_model_loaded'] = gr.State(False)
        components['clear_cache_after_n_frames'] = gr.State(100)
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_video'] = gr.Video(label="Input Video")

                components['multiplier'] = gr.Slider(label="Multiplier", minimum=2, maximum=8, step=1, value=2)
                components['fps_mode'] = gr.Radio(
                    label="Framerate Mode",
                    choices=["Keep Original Duration (Interpolate)", "Keep Original Framerate (Slow Motion)"],
                    value="Keep Original Duration (Interpolate)"
                )
                
            with gr.Column(scale=1):
                components['output_video'] = gr.Video(label="Result", show_label=False, interactive=False, height=488)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def run_generation(ui_values):
    original_run_button_text = UI_INFO["run_button_text"]
    
    yield (
        "Status: Preparing...",
        None,
        gr.update(value="Stop", variant="stop")
    )
    
    input_video_path = ui_values.get('input_video')
    if not input_video_path:
        raise gr.Error("Input video is missing.")

    temp_audio_path = None
    silent_video_path = None
    final_video_path = None
    
    try:
        yield ("Status: Extracting audio...", None, gr.update())
        temp_audio_path = extract_audio_ffmpeg(input_video_path)

        workflow, extra_data = process_inputs(ui_values)
        workflow_package = (workflow, extra_data)
        
        for status, output_files in run_workflow_and_get_output(workflow_package):
            if output_files and isinstance(output_files, list):
                silent_video_path = output_files[0]
            
            yield (status, silent_video_path, gr.update())

        if silent_video_path:
            yield ("Status: Merging audio...", silent_video_path, gr.update())
            is_slow_motion = "Slow Motion" in ui_values.get('fps_mode', "")
            multiplier = int(ui_values.get('multiplier', 2))
            final_video_path = merge_video_audio_ffmpeg(silent_video_path, temp_audio_path, multiplier, is_slow_motion)
            yield ("Status: Merging complete!", final_video_path, gr.update())
        else:
             raise RuntimeError("RIFE workflow did not produce a video file.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (
            f"Error: {e}",
            None,
            gr.update(value=original_run_button_text, variant="primary")
        )

    finally:
        print("RIFE generation task finished. Cleaning up temporary files...")
        cleanup_paths = [temp_audio_path, silent_video_path]
        for path in cleanup_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Removed temp file: {path}")
                except Exception as e:
                    print(f"Error removing temp file {path}: {e}")

        yield (
            "Status: Ready",
            final_video_path or gr.update(),
            gr.update(value=original_run_button_text, variant="primary")
        )