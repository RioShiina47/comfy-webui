import os
import re
from collections import defaultdict
from typing import List, Dict, Any
from core.config import COMFYUI_OUTPUT_PATH

def scan_output_directory(limit: int = 200) -> List[Dict[str, Any]]:
    if not os.path.isdir(COMFYUI_OUTPUT_PATH):
        print(f"[History] Output directory not found: {COMFYUI_OUTPUT_PATH}")
        return []

    prefix_regex = re.compile(r"^(.*?)_(\d+)(_\.|\.)")
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
    video_extensions = {'.mp4', '.webm'}
    model_3d_extensions = {'.glb', '.obj'}
    audio_extensions = {'.mp3', '.wav', '.flac'}
    
    grouped_files = defaultdict(lambda: {'files': [], 'latest_timestamp': 0})

    for root, _, files in os.walk(COMFYUI_OUTPUT_PATH):
        for filename in files:
            match = prefix_regex.match(filename)
            if match:
                prefix = match.group(1)
                group_key = os.path.join(root, prefix)
            else:
                group_key = os.path.join(root, os.path.splitext(filename)[0])
            
            full_path = os.path.join(root, filename)
            
            try:
                mod_time = os.path.getmtime(full_path)
                
                group = grouped_files[group_key]
                group['files'].append(full_path)
                if mod_time > group['latest_timestamp']:
                    group['latest_timestamp'] = mod_time
                    
            except FileNotFoundError:
                continue

    history_items = []
    for group_key, data in grouped_files.items():
        files = sorted(data['files'])
        
        preview_file = None
        preview_priority = 99
        
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            current_priority = 99
            if ext in image_extensions: current_priority = 1
            elif ext in video_extensions: current_priority = 2
            elif ext in model_3d_extensions: current_priority = 3
            elif ext in audio_extensions: current_priority = 4

            if current_priority < preview_priority:
                preview_priority = current_priority
                preview_file = f

        history_items.append({
            "timestamp": data['latest_timestamp'],
            "files": files,
            "preview_file": preview_file
        })

    history_items.sort(key=lambda x: x['timestamp'], reverse=True)

    return history_items[:limit]