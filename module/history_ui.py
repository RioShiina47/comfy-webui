import os
import re
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any
import gradio as gr
from core.config import COMFYUI_OUTPUT_PATH

UI_INFO = {
    "main_tab": "History",
    "sub_tab": "History",
}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
VIDEO_EXTENSIONS = {'.mp4', '.webm'}
MODEL_3D_EXTENSIONS = {'.glb', '.obj'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac'}


def scan_output_directory(limit: int = 200) -> List[Dict[str, Any]]:
    """Scans COMFYUI_OUTPUT_PATH, groups files by job prefix, and determines preview files."""
    if not os.path.isdir(COMFYUI_OUTPUT_PATH):
        print(f"[History] Output directory not found: {COMFYUI_OUTPUT_PATH}")
        return []

    prefix_regex = re.compile(r"^(.*?)_(\d+)(_\.|\.)")
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
            if ext in IMAGE_EXTENSIONS: current_priority = 1
            elif ext in VIDEO_EXTENSIONS: current_priority = 2
            elif ext in MODEL_3D_EXTENSIONS: current_priority = 3
            elif ext in AUDIO_EXTENSIONS: current_priority = 4

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


def create_ui():
    """Creates the UI components for the History tab."""
    components = {}
    with gr.Column():
        gr.Markdown("## Generation History")
        gr.Markdown("💡 **Tip:** Click on a row in the table to see a preview on the right. Use the download button on the preview to save files.")
        
        components['refresh_button'] = gr.Button("🔄 Refresh History", variant="primary")
        
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                components['history_df'] = gr.DataFrame(
                    headers=["Type", "Time"],
                    datatype=["str", "str"],
                    label="Completed Jobs",
                    interactive=True,
                    row_count=20,
                    column_count=(2, "fixed"),
                    wrap=True
                )
            with gr.Column(scale=1):
                gr.Markdown("### Preview")
                components['preview_image'] = gr.Image(label="Image Preview", visible=False, interactive=False, height=400)
                components['preview_video'] = gr.Video(label="Video Preview", visible=False, interactive=False, height=400)
                components['preview_model3d'] = gr.Model3D(label="3D Model Preview", visible=False, interactive=False, height=400)
                components['preview_audio'] = gr.Audio(label="Audio Preview", visible=False, interactive=False)
        
    components['raw_history_state'] = gr.State([])

    return components

def get_main_output_components(components: dict):
    return []

def refresh_history():
    """Fetches completed jobs from the output folder and formats them for the UI."""
    history_items = scan_output_directory()
    
    if not history_items:
        return [], [["", "No files found."]], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    df_data = []
    for item in history_items:
        timestamp = datetime.fromtimestamp(item["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
        preview_file_path = item.get("preview_file")
        file_type = "Group"
        if preview_file_path:
            ext = os.path.splitext(preview_file_path)[1].lower()
            if ext in IMAGE_EXTENSIONS: file_type = "Image"
            elif ext in VIDEO_EXTENSIONS: file_type = "Video"
            elif ext in MODEL_3D_EXTENSIONS: file_type = "3D Model"
            elif ext in AUDIO_EXTENSIONS: file_type = "Audio"
            else: file_type = f"{ext.upper()} File"
        
        df_data.append([file_type, timestamp])
    
    return history_items, df_data, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def on_select_job(history_state: list, evt: gr.SelectData):
    """Handles row selection in the DataFrame to update the preview."""
    all_hidden = [gr.update(visible=False, value=None)] * 4
    
    if not history_state or not hasattr(evt, 'index') or evt.index is None:
        return tuple(all_hidden)
    
    row_index = evt.index[0]
    if row_index >= len(history_state):
        return tuple(all_hidden)
        
    selected_item = history_state[row_index]
    preview_path = selected_item.get("preview_file")

    if not preview_path or not os.path.exists(preview_path):
        return tuple(all_hidden)

    ext = os.path.splitext(preview_path)[1].lower()
    
    if ext in IMAGE_EXTENSIONS:
        return gr.update(value=preview_path, visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif ext in VIDEO_EXTENSIONS:
        return gr.update(visible=False), gr.update(value=preview_path, visible=True), gr.update(visible=False), gr.update(visible=False)
    elif ext in MODEL_3D_EXTENSIONS:
        return gr.update(visible=False), gr.update(visible=False), gr.update(value=preview_path, visible=True), gr.update(visible=False)
    elif ext in AUDIO_EXTENSIONS:
         return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(value=preview_path, visible=True)
    else:
        return tuple(all_hidden)

def create_event_handlers(components: dict, all_components: dict, demo: gr.Blocks):
    """Binds event handlers for the History UI module."""
    refresh_button = components['refresh_button']
    raw_history_state = components['raw_history_state']
    history_df = components['history_df']
    preview_image = components['preview_image']
    preview_video = components['preview_video']
    preview_model3d = components['preview_model3d']
    preview_audio = components['preview_audio']

    preview_outputs = [preview_image, preview_video, preview_model3d, preview_audio]

    refresh_button.click(
        fn=refresh_history,
        inputs=None,
        outputs=[raw_history_state, history_df] + preview_outputs,
        api_name=False
    )
    
    demo.load(
        fn=refresh_history,
        inputs=None,
        outputs=[raw_history_state, history_df] + preview_outputs,
        api_name=False
    )
    
    history_df.select(
        fn=on_select_job,
        inputs=[raw_history_state],
        outputs=preview_outputs,
        show_progress=False,
        api_name=False
    )