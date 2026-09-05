import gradio as gr
import random
import os
import shutil
import time
import tempfile
from PIL import Image
import traceback

from .hunyuan3d2_mv23d_logic import process_inputs
from core.comfy_api import run_workflow_and_get_output

UI_INFO = {
    "main_tab": "3DGen",
    "sub_tab": "Hunyuan3D-2mv",
    "run_button_text": "🧊 Generate 3D Model (Multi-View)",
    "target_backend": "3d_backend"
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Hunyuan3D-2mv: Three-View to 3D")
        gr.Markdown("💡 **Tip:** Upload three views of an object (front, back, left). The background will be removed automatically.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_front_image'] = gr.Image(type="pil", label="Front View Image", sources=["upload"])
                components['input_back_image'] = gr.Image(type="pil", label="Back View Image", sources=["upload"])
                components['input_left_image'] = gr.Image(type="pil", label="Left View Image", sources=["upload"])
                
                components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
            
            with gr.Column(scale=2):
                with gr.Row():
                    components['output_shape_model'] = gr.Model3D(label="Shape Output (.glb)", interactive=False)
                    components['output_textured_model'] = gr.Model3D(label="Textured Output (.glb)", interactive=False)

    return components

def get_main_output_components(components: dict):
    return [
        components['output_shape_model'], 
        components['output_textured_model'], 
        components['run_button']
    ]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def run_generation(ui_values):
    original_run_button_text = UI_INFO["run_button_text"]
    
    yield (
        "Status: Preparing...",
        None,
        None,
        gr.update()
    )
    
    shape_model_path, textured_model_path = None, None
    expected_files = {}
    
    try:
        workflow, extra_data = process_inputs(ui_values)
        expected_files = extra_data.get("expected_files", {})
        workflow_package = (workflow, extra_data)
        
        for status, files in run_workflow_and_get_output(workflow_package):
            yield (status, gr.update(), gr.update(), gr.update())

    except Exception as e:
        traceback.print_exc()
        yield (
            f"Error: {e}",
            None,
            None,
            gr.update()
        )
        return

    finally:
        print("Workflow finished. Manually checking for output files...")
        
        found_all = False
        for _ in range(5):
            if os.path.exists(expected_files.get("shape")) and os.path.exists(expected_files.get("textured")):
                found_all = True
                break
            time.sleep(1)

        if not found_all:
            print("Error: Could not find generated files after workflow completion.")
            yield (
                "Error: Output files not found after execution.",
                None,
                None,
                gr.update()
            )
            return

        print("Output files found. Copying to temporary location for Gradio.")
        try:
            shape_src_path = expected_files.get("shape")
            if shape_src_path and os.path.exists(shape_src_path):
                temp_shape_path = tempfile.NamedTemporaryFile(delete=False, suffix="_shape.glb").name
                shutil.copy(shape_src_path, temp_shape_path)
                shape_model_path = temp_shape_path

            textured_src_path = expected_files.get("textured")
            if textured_src_path and os.path.exists(textured_src_path):
                temp_textured_path = tempfile.NamedTemporaryFile(delete=False, suffix="_textured.glb").name
                shutil.copy(textured_src_path, temp_textured_path)
                textured_model_path = temp_textured_path

            yield (
                "Status: Loaded successfully!",
                shape_model_path,
                textured_model_path,
                gr.update()
            )
        except Exception as e:
            print(f"Error copying files for Gradio: {e}")
            yield (
                f"Error: Could not prepare files for display: {e}",
                None, None,
                gr.update()
            )