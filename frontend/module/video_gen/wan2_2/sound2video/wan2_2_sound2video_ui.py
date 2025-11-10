import gradio as gr
import random
import os
import shutil
import yaml
import math
import tempfile
import subprocess
from PIL import Image

from core.workflow_assembler import WorkflowAssembler
from core.config import COMFYUI_INPUT_PATH
from core.media_utils import get_media_metadata
from core.comfy_api import run_workflow_and_get_output
from core.workflow_utils import get_filename_prefix

UI_INFO = {
    "workflow_recipe": "wan2_2_sound2video_recipe.yaml",
    "main_tab": "VideoGen",
    "sub_tab": "sound2video",
    "run_button_text": "🔊 Generate from Sound"
}

ASPECT_RATIO_PRESETS = {
    "16:9 (Landscape)": (1280, 720), "9:16 (Portrait)": (720, 1280),
    "1:1 (Square)": (960, 960), "4:3 (Classic TV)": (1088, 816),
    "3:4 (Classic Portrait)": (816, 1088),
    "3:2 (Photography)": (1152, 768),
    "2:3 (Photography Portrait)": (768, 1152),
}
CHUNK_LENGTH = 77
FPS = 16

def create_ui():
    components = {}
    with gr.Column():
        gr.Markdown("## Wan 2.2 S2V (Sound-to-Video) (14B) - Lightning")
        gr.Markdown("💡 **Tip:** Upload a reference image and an audio file. Use 'Preview' for a quick test, or 'Generate Full Video' for the final output.")
        
        components['mode'] = gr.Radio(
            choices=["Preview (77 frames)", "Generate Full Video"],
            value="Preview (77 frames)",
            label="Generation Mode"
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                components['ref_image'] = gr.Image(type="pil", label="Reference Image", height=282)
            with gr.Column(scale=1):
                components['audio_file'] = gr.Audio(type="filepath", label="Audio File")

        with gr.Row():
            with gr.Column(scale=1):
                components['positive_prompt'] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the character and scene.")
                components['negative_prompt'] = gr.Textbox(label="Negative Prompt", lines=3)
                with gr.Row():
                    components['aspect_ratio'] = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=list(ASPECT_RATIO_PRESETS.keys()),
                    value="16:9 (Landscape)",
                    interactive=True
                )
                with gr.Row():
                    components['seed'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    components['batch_count'] = gr.Slider(label="Batch Count", minimum=1, maximum=10, step=1, value=1)

            with gr.Column(scale=1):
                components['output_video'] = gr.Video(
                    label="Result", 
                    show_label=False, 
                    interactive=False, 
                    height=468
                )

        components['run_button'] = gr.Button(UI_INFO["run_button_text"], variant="primary", elem_classes=["run-shortcut"])
                
    return components

def get_main_output_components(components: dict):
    return [components['output_video'], components['run_button']]

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    pass

def save_temp_file(file_obj, name_prefix: str, is_audio=False) -> str:
    if file_obj is None:
        return None

    if is_audio:
        ext = os.path.splitext(file_obj)[1] or ".wav"
        temp_filename = f"temp_{name_prefix}_{random.randint(1000, 9999)}{ext}"
        save_path = os.path.join(COMFYUI_INPUT_PATH, temp_filename)
        shutil.copy(file_obj, save_path)
    else:
        temp_filename = f"temp_{name_prefix}_{random.randint(1000, 9999)}.png"
        save_path = os.path.join(COMFYUI_INPUT_PATH, temp_filename)
        file_obj.save(save_path, "PNG")

    return temp_filename

def process_inputs(ui_values, seed_override=None):
    local_ui_values = ui_values.copy()
    ref_image_pil = local_ui_values.get('ref_image')
    original_audio_path = local_ui_values.get('audio_file')
    if ref_image_pil is None: raise gr.Error("Reference image is required.")
    if original_audio_path is None: raise gr.Error("Audio file is required.")

    local_ui_values['ref_image'] = save_temp_file(ref_image_pil, "s2v_ref")
    local_ui_values['audio_file'] = save_temp_file(original_audio_path, "s2v_audio", is_audio=True)
    
    selected_ratio = local_ui_values.get('aspect_ratio')
    width, height = ASPECT_RATIO_PRESETS[selected_ratio]
    local_ui_values['width'] = width
    local_ui_values['height'] = height
    
    current_seed = 0
    if seed_override is not None:
        current_seed = seed_override
    else:
        seed = int(local_ui_values.get('seed', -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        current_seed = seed
    local_ui_values['seed'] = current_seed
    
    filename_prefix = get_filename_prefix()
    local_ui_values['filename_prefix'] = filename_prefix

    mode = local_ui_values.get('mode', 'Preview (77 frames)')
    if mode == 'Generate Full Video':
        audio_meta = get_media_metadata(original_audio_path, is_video=True)
        duration = audio_meta.get('duration', 0)
        if duration <= 0:
            raise gr.Error("Could not determine audio duration or audio is empty.")
        
        total_frames = int(duration * FPS)
        num_chunks = math.ceil(total_frames / CHUNK_LENGTH) if total_frames > 0 else 1

        local_ui_values['s2v_chain'] = {
            "num_chunks": num_chunks,
            "seed": current_seed,
        }
    
    recipe_path = UI_INFO["workflow_recipe"]
    module_path = os.path.dirname(os.path.abspath(__file__))
    assembler = WorkflowAssembler(recipe_path, base_path=module_path)
    workflow = assembler.assemble(local_ui_values)
    
    return workflow, None

def run_generation(ui_values):
    final_video = None
    try:
        batch_count = int(ui_values.get('batch_count', 1))
        original_seed = int(ui_values.get('seed', -1))
        
        for i in range(batch_count):
            current_seed = original_seed + i if original_seed != -1 else None
            batch_msg = f" (Batch {i + 1}/{batch_count})" if batch_count > 1 else ""
            
            yield (f"Status: Preparing{batch_msg}...", None)
            
            workflow, extra_data = process_inputs(ui_values, seed_override=current_seed)
            workflow_package = (workflow, extra_data)
            
            for status, output_files in run_workflow_and_get_output(workflow_package):
                status_msg_with_batch = f"Status: {status.replace('Status: ', '')}{batch_msg}"
                if output_files and isinstance(output_files, list):
                    final_video = output_files[0]
                
                yield (status_msg_with_batch, final_video)

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield (f"Error: {e}", None)
        return
        
    yield ("Status: Loaded successfully!", final_video)