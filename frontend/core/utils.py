import os
import random
import shutil
import traceback
from PIL import Image
import numpy as np
from core.config import COMFYUI_INPUT_PATH
from core.comfy_api import run_workflow_and_get_output

def save_temp_image(img):
    if not isinstance(img, Image.Image): return None
    filename = f"temp_image_{random.randint(10000, 99999)}.png"
    filepath = os.path.join(COMFYUI_INPUT_PATH, filename)
    img.save(filepath, "PNG")
    return os.path.basename(filepath)

def save_temp_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        print(f"Warning: Audio path '{audio_path}' is invalid or does not exist. Cannot save temp audio.")
        return None
    
    ext = os.path.splitext(audio_path)[1] or ".wav"
    filename = f"temp_audio_{random.randint(10000, 99999)}{ext}"
    save_path = os.path.join(COMFYUI_INPUT_PATH, filename)
    shutil.copy(audio_path, save_path)
    print(f"Saved temporary audio file to: {save_path}")
    return os.path.basename(filename)

def save_temp_video(video_path):
    if not video_path or not os.path.exists(video_path):
        print(f"Warning: Video path '{video_path}' is invalid or does not exist. Cannot save temp video.")
        return None
    
    ext = os.path.splitext(video_path)[1] or ".mp4"
    filename = f"temp_video_{random.randint(10000, 99999)}{ext}"
    save_path = os.path.join(COMFYUI_INPUT_PATH, filename)
    shutil.copy(video_path, save_path)
    print(f"Saved temporary video file to: {save_path}")
    return os.path.basename(filename)

def create_mask_from_layer(image_editor_output):
    if not image_editor_output or image_editor_output.get('background') is None or not image_editor_output.get('layers'):
        return None

    background = image_editor_output['background']
    composite_mask = Image.new('RGBA', background.size, (0, 0, 0, 0))

    for layer_pil in image_editor_output['layers']:
        if layer_pil:
            composite_mask.paste(layer_pil, (0, 0), layer_pil)

    r, g, b, a = composite_mask.split()
    white_rgb_mask = Image.merge('RGBA', [Image.new('L', r.size, 255), Image.new('L', g.size, 255), Image.new('L', b.size, 255), a])
    
    return white_rgb_mask

def handle_seed(seed_value: int, max_val: int = 2**32 - 1) -> int:
    if seed_value == -1:
        return random.randint(0, max_val)
    return int(seed_value)

def create_simple_run_generation(process_inputs_func, get_ui_updates_func):
    def run_generation(ui_values):
        final_files = []
        try:
            yield get_ui_updates_func("Status: Preparing...", final_files)
            
            workflow, extra_data = process_inputs_func(ui_values)
            workflow_package = (workflow, extra_data)
            
            for status, output_files in run_workflow_and_get_output(workflow_package):
                if output_files and isinstance(output_files, list):
                    final_files = output_files
                
                yield get_ui_updates_func(status, final_files)

        except Exception as e:
            traceback.print_exc()
            yield get_ui_updates_func(f"Error: {e}", final_files)
            return

        yield get_ui_updates_func("Status: Loaded successfully!", final_files)
    
    return run_generation

def create_batched_run_generation(process_inputs_func, get_ui_updates_func):
    def run_generation(ui_values):
        all_output_files = []
        try:
            batch_count_key = 'batch_count'
            seed_key = 'seed'

            batch_count = int(ui_values.get(batch_count_key, 1))
            original_seed = int(ui_values.get(seed_key, -1))

            for i in range(batch_count):
                current_seed = original_seed + i if original_seed != -1 else None
                batch_msg = f" (Batch {i + 1}/{batch_count})" if batch_count > 1 else ""
                
                yield get_ui_updates_func(f"Status: Preparing{batch_msg}...", all_output_files)
                
                workflow, extra_data = process_inputs_func(ui_values, seed_override=current_seed)
                workflow_package = (workflow, extra_data)
                
                for status, output_path in run_workflow_and_get_output(workflow_package):
                    status_msg = f"Status: {status.replace('Status: ', '')}{batch_msg}"
                    
                    if output_path and isinstance(output_path, list):
                        new_files = [f for f in output_path if f not in all_output_files]
                        if new_files:
                            all_output_files.extend(new_files)

                    yield get_ui_updates_func(status_msg, all_output_files)

        except Exception as e:
            traceback.print_exc()
            yield get_ui_updates_func(f"Error: {e}", all_output_files)
            return

        yield get_ui_updates_func("Status: Loaded successfully!", all_output_files)
        
    return run_generation