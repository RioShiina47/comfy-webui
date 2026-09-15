import gradio as gr
import os
import shutil
import time
import tempfile
import glob
import traceback

from .pixal3d_trellis2_img23d_logic import process_inputs
from core.comfy_api import run_workflow_and_get_output
from core.config import COMFYUI_OUTPUT_PATH

UI_INFO = {
    "main_tab": "3DGen",
    "sub_tab": "Pixal3D & TRELLIS.2",
    "run_button_text": "🧊 Generate 3D Model",
    "target_backend": "default"
}


def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Pixal3D & TRELLIS.2: Image-to-3D")
        gr.Markdown("💡 **Tip:** Upload a single image (PNG/JPG). Background removal, MoGe camera FoV estimation, high-resolution geometry upsampling, and PBR texture baking will be executed automatically.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['input_image'] = gr.Image(type="pil", label="Input Image", sources=["upload"])
                components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    components['target_face_count'] = gr.Slider(
                        label="Decimate Face Count", 
                        minimum=100000, 
                        maximum=1500000, 
                        value=700000, 
                        step=50000
                    )
                    components['texture_resolution'] = gr.Dropdown(
                        label="Texture Resolution", 
                        choices=[1024, 2048, 4096], 
                        value=4096
                    )
                
                components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
            
            with gr.Column(scale=2):
                with gr.Row():
                    components['output_textured_model'] = gr.Model3D(label="Textured PBR Output (.glb)", interactive=False)
                    components['output_shape_model'] = gr.Model3D(label="Painted Mesh Output (.glb)", interactive=False)

    return components


def get_main_output_components(components: dict):
    return [
        components['output_textured_model'],
        components['output_shape_model'],
        components['run_button']
    ]


def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass


def run_generation(ui_values):
    yield (
        "Status: Preparing...",
        None,
        None,
        gr.update()
    )
    
    shape_model_path = None
    textured_model_path = None
    expected_files = {}
    downloaded_files = []
    
    try:
        workflow, extra_data = process_inputs(ui_values)
        expected_files = extra_data.get("expected_files", {})
        workflow_package = (workflow, extra_data)
        
        for status, files in run_workflow_and_get_output(workflow_package):
            if files:
                downloaded_files = files
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
        print("Workflow finished. Processing output 3D files for Gradio...")
        
        # 1. Check if files were downloaded via comfy_api
        for fpath in downloaded_files:
            if isinstance(fpath, str) and fpath.endswith(".glb"):
                fname_lower = os.path.basename(fpath).lower()
                if "shape" in fname_lower and not shape_model_path:
                    shape_model_path = fpath
                elif "textured" in fname_lower and not textured_model_path:
                    textured_model_path = fpath
                elif not textured_model_path:
                    textured_model_path = fpath
                elif not shape_model_path:
                    shape_model_path = fpath

        # 2. Fallback: check expected_files or glob on local filesystem
        if not (shape_model_path and textured_model_path):
            shape_src = expected_files.get("shape")
            textured_src = expected_files.get("textured")
            prefix_shape = expected_files.get("prefix_shape")
            prefix_textured = expected_files.get("prefix_textured")
            
            for _ in range(5):
                if shape_src and os.path.exists(shape_src) and not shape_model_path:
                    shape_model_path = shape_src
                if textured_src and os.path.exists(textured_src) and not textured_model_path:
                    textured_model_path = textured_src
                
                if (not shape_model_path or not textured_model_path) and COMFYUI_OUTPUT_PATH and os.path.exists(COMFYUI_OUTPUT_PATH):
                    if not shape_model_path and prefix_shape:
                        matches = glob.glob(os.path.join(COMFYUI_OUTPUT_PATH, f"{prefix_shape}*.glb".replace('/', os.sep)))
                        if matches:
                            shape_model_path = matches[-1]
                    if not textured_model_path and prefix_textured:
                        matches = glob.glob(os.path.join(COMFYUI_OUTPUT_PATH, f"{prefix_textured}*.glb".replace('/', os.sep)))
                        if matches:
                            textured_model_path = matches[-1]

                if shape_model_path and textured_model_path:
                    break
                time.sleep(1)

        if not textured_model_path and not shape_model_path:
            print("Error: Could not find generated 3D files after workflow completion.")
            yield (
                "Error: Output 3D files not found after execution.",
                None,
                None,
                gr.update()
            )
            return

        print("Output files found. Copying to temporary location for Gradio...")
        try:
            temp_textured = None
            temp_shape = None
            
            if textured_model_path and os.path.exists(textured_model_path):
                temp_textured = tempfile.NamedTemporaryFile(delete=False, suffix="_textured.glb").name
                shutil.copy(textured_model_path, temp_textured)
                
            if shape_model_path and os.path.exists(shape_model_path):
                temp_shape = tempfile.NamedTemporaryFile(delete=False, suffix="_shape.glb").name
                shutil.copy(shape_model_path, temp_shape)

            yield (
                "Status: Loaded successfully!",
                temp_textured,
                temp_shape,
                gr.update()
            )
        except Exception as e:
            print(f"Error preparing 3D files for Gradio: {e}")
            yield (
                f"Error: Could not prepare 3D files for display: {e}",
                None,
                None,
                gr.update()
            )
