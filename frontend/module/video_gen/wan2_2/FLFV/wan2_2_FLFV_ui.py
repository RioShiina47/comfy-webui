import gradio as gr
import random
import os
import yaml
from PIL import Image
from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix

UI_INFO = {
    "workflow_recipe": "wan2_2_FLFV_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "FLFV",
    "run_button_text": "🚀 Generate FLFV Video"
}

ASPECT_RATIO_PRESETS = {
    "16:9 (Landscape)": (1280, 720),
    "9:16 (Portrait)": (720, 1280),
    "1:1 (Square)": (960, 960),
    "4:3 (Classic TV)": (1088, 816),
    "3:4 (Classic Portrait)": (816, 1088),
    "3:2 (Photography)": (1152, 768),
    "2:3 (Photography Portrait)": (768, 1152),
}

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.2 FLFV (First-Last-Frame-Video) (14B) - Lightning")
        gr.Markdown("💡 **Tip:** Please select an aspect ratio for the output video.")
        
        with gr.Row():
            with gr.Column(scale=1):
                components['start_image'] = gr.Image(type="pil", label="Start Image")
            with gr.Column(scale=1):
                components['end_image'] = gr.Image(type="pil", label="End Image")

        with gr.Row():
            with gr.Column(scale=1):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the scene and the desired transformation.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=list(ASPECT_RATIO_PRESETS.keys()),
                        value="16:9 (Landscape)",
                        interactive=True
                    )
                    components['video_length'] = gr.Slider(label="Video Length (frames)", minimum=8, maximum=81, step=1, value=81)
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Video(label="Result", show_label=False, interactive=False, height=472)

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def save_temp_image(image_pil: Image.Image, name_prefix: str) -> str:
    if image_pil is None: return None
    comfyui_input_dir = COMFYUI_INPUT_PATH
    temp_filename = f"temp_{name_prefix}_{random.randint(1000, 9999)}.png"
    save_path = os.path.join(comfyui_input_dir, temp_filename)
    image_pil.save(save_path, "PNG")
    print(f"Saved temporary input image to: {save_path}")
    return temp_filename

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    start_image_pil = local_ui_values.get('start_image')
    end_image_pil = local_ui_values.get('end_image')
    if start_image_pil is None: raise gr.Error("Start image is required.")
    if end_image_pil is None: raise gr.Error("End image is required.")
    
    selected_ratio = local_ui_values.get('aspect_ratio')
    width, height = ASPECT_RATIO_PRESETS[selected_ratio]

    local_ui_values['width'] = width
    local_ui_values['height'] = height
    local_ui_values['start_image'] = save_temp_image(start_image_pil, "flfv_start")
    local_ui_values['end_image'] = save_temp_image(end_image_pil, "flfv_end")
    
    if seed_override is not None:
        local_ui_values['seed'] = seed_override
    else:
        seed = int(local_ui_values.get('seed', -1))
        if seed == -1:
            seed = random.randint(0, 999999999999999)
        local_ui_values['seed'] = seed
    
    local_ui_values['video_length'] = int(local_ui_values.get('video_length', 81))
    local_ui_values['filename_prefix'] = get_filename_prefix()
    
    recipe_path = UI_INFO["workflow_recipe"]
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    final_workflow = assembler.assemble(local_ui_values)
    
    return final_workflow, None

def run_generation(ui_values):
    all_output_files = []
    
    try:
        batch_count = int(ui_values.get('batch_count', 1))
        original_seed = int(ui_values.get('seed', -1))

        for i in range(batch_count):
            current_seed = original_seed + i if original_seed != -1 else None
            batch_msg = f" (Batch {i + 1}/{batch_count})" if batch_count > 1 else ""
            
            yield (f"Status: Preparing{batch_msg}...", None)
            
            workflow, extra_data = process_inputs(ui_values, seed_override=current_seed)
            workflow_package = (workflow, extra_data)
            
            for status, output_path in run_workflow_and_get_output(workflow_package):
                status_msg = f"Status: {status.replace('Status: ', '')}{batch_msg}"
                
                final_video = None
                if output_path and isinstance(output_path, list):
                    all_output_files.extend(output_path)
                    final_video = all_output_files[-1]

                yield (status_msg, final_video)

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return
        
    yield ("Status: Loaded successfully!", all_output_files[-1] if all_output_files else None)