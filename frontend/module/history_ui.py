import gradio as gr
from datetime import datetime
from core.history_utils import scan_output_directory
import os

UI_INFO = {
    "main_tab": "History",
    "sub_tab": "History",
}

def create_ui():
    """Creates the UI components for the History tab."""
    components = {}
    with gr.Column():
        gr.Markdown("## Generation History")
        gr.HTML("""
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
        """, visible=False)
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
                    col_count=(2, "fixed"),
                    wrap=True
                )
            with gr.Column(scale=1):
                gr.Markdown("### Preview")
                components['preview_image'] = gr.Image(label="Image Preview", visible=False, interactive=False, height=400, show_download_button=True)
                components['preview_video'] = gr.Video(label="Video Preview", visible=False, interactive=False, height=400, show_share_button=False)
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
            if ext in ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp']: file_type = "Image"
            elif ext in ['.mp4', '.webm']: file_type = "Video"
            elif ext in ['.glb', '.obj']: file_type = "3D Model"
            elif ext in ['.mp3', '.wav', '.flac']: file_type = "Audio"
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
    
    image_ext = ['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp']
    video_ext = ['.mp4', '.webm']
    model_3d_ext = ['.glb', '.obj']
    audio_ext = ['.mp3', '.wav', '.flac']
    
    if ext in image_ext:
        return gr.update(value=preview_path, visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    elif ext in video_ext:
        return gr.update(visible=False), gr.update(value=preview_path, visible=True), gr.update(visible=False), gr.update(visible=False)
    elif ext in model_3d_ext:
        return gr.update(visible=False), gr.update(visible=False), gr.update(value=preview_path, visible=True), gr.update(visible=False)
    elif ext in audio_ext:
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
        show_api=False
    )
    
    demo.load(
        fn=refresh_history,
        inputs=None,
        outputs=[raw_history_state, history_df] + preview_outputs,
        show_api=False
    )
    
    history_df.select(
        fn=on_select_job,
        inputs=[raw_history_state],
        outputs=preview_outputs,
        show_progress=False,
        show_api=False
    )