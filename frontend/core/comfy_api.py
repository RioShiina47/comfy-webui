import requests
import json
import uuid
import urllib.parse
import websocket
import tempfile
import shutil
from pathlib import Path
import gradio as gr
import pyperclip
import os

from core.backend_manager import backend_manager
from core.config import DEV_COPY_WORKFLOW_TO_CLIPBOARD, DEV_SAVE_WORKFLOW_TO_JSON, JSON_SAVE_PATH
from core.workflow_utils import get_filename_prefix

def queue_prompt(prompt_workflow, client_id, extra_data=None):
    try:
        if DEV_COPY_WORKFLOW_TO_CLIPBOARD:
            try:
                workflow_str = json.dumps(prompt_workflow, indent=2)
                pyperclip.copy(workflow_str)
                print("[Dev Feature] Workflow JSON has been copied to the clipboard.")
            except Exception as e:
                print(f"[Dev Feature] Warning: Failed to copy workflow to clipboard: {e}")
        
        if DEV_SAVE_WORKFLOW_TO_JSON:
            try:
                filename = f"{get_filename_prefix()}_workflow.json"
                filepath = os.path.join(JSON_SAVE_PATH, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(prompt_workflow, f, indent=2)
                print(f"[Dev Feature] Workflow saved to: {filepath}")
            except Exception as e:
                print(f"[Dev Feature] Warning: Failed to save workflow to JSON file: {e}")


        payload = {"prompt": prompt_workflow, "client_id": client_id}
        if extra_data:
            payload.update(extra_data)
        
        active_url = backend_manager.get_active_backend_url()
        response = requests.post(f"{active_url}/prompt", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error queuing prompt: {e}")
        return None

def get_output_data(prompt_id, client_id):
    active_url = backend_manager.get_active_backend_url()
    ws_url = f"ws://{urllib.parse.urlparse(active_url).netloc}/ws?clientId={client_id}"
    ws = None
    try:
        ws = websocket.create_connection(ws_url)
        queue_remaining = -1

        while True:
            out = ws.recv()
            if not isinstance(out, str):
                continue
            
            message = json.loads(out)
            msg_type = message.get('type')

            if msg_type == 'status':
                data = message.get('data', {})
                status_info = data.get('status', {})
                queue_remaining = status_info.get('exec_info', {}).get('queue_remaining', -1)
                if queue_remaining == 0:
                    break

            elif msg_type == 'executed':
                data = message.get('data', {})
                if data.get('prompt_id') == prompt_id:
                    output_data = data.get('output', {})
                    has_output = any(
                        isinstance(v, list) and v and isinstance(v[0], dict) and 'filename' in v[0]
                        for v in output_data.values()
                    )
                    if has_output:
                        print(f"\nReceived node output for prompt {prompt_id}.")
                        yield output_data
            
            elif msg_type == 'progress':
                data = message.get('data', {})
                progress = f"Progress: {data.get('value')}/{data.get('max')}"
                print(progress, end='\r')
                yield progress
        
        print("\nWebSocket stream finished.")

    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        if ws:
            ws.close()

def download_file(filename, subfolder, file_type="output"):
    active_url = backend_manager.get_active_backend_url()
    url = f"{active_url}/view?filename={urllib.parse.quote_plus(filename)}&subfolder={urllib.parse.quote_plus(subfolder)}&type={file_type}"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            suffix = Path(filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                shutil.copyfileobj(r.raw, tmp_file)
                return tmp_file.name
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

def run_workflow_and_get_output(workflow_data):
    client_id = uuid.uuid4().hex
    
    prompt_workflow, extra_data = None, None
    if isinstance(workflow_data, tuple) and len(workflow_data) == 2:
        prompt_workflow, extra_data = workflow_data
    else:
        prompt_workflow = workflow_data

    yield "Status: Sending to ComfyUI...", None
    
    queue_data = queue_prompt(prompt_workflow, client_id, extra_data)
    if not queue_data or 'prompt_id' not in queue_data:
        active_url = backend_manager.get_active_backend_url()
        yield f"Error: Failed to send to ComfyUI backend at {active_url}. Please check if the service is running.", None
        return
        
    prompt_id = queue_data['prompt_id']
    yield f"Status: Workflow queued. Waiting for ComfyUI to process...", None
    
    all_local_file_paths = []
    
    for update in get_output_data(prompt_id, client_id):
        if isinstance(update, str):
            yield f"Status: {update}", None
        elif isinstance(update, dict):
            yield "Status: Node execution finished, downloading output...", None
            
            output_files_info = []
            for key, value in update.items():
                if isinstance(value, list) and value and isinstance(value[0], dict) and 'filename' in value[0]:
                    print(f"Found output files under key: '{key}'")
                    output_files_info.extend(value)

            for i, output_info in enumerate(output_files_info):
                yield f"Status: Downloading file {i+1}/{len(output_files_info)}...", None
                local_file_path = download_file(output_info['filename'], output_info['subfolder'], output_info['type'])
                if local_file_path:
                    all_local_file_paths.append(local_file_path)

            yield "Status: Download complete, waiting for the next node...", None

    if not all_local_file_paths:
        yield f"Error: Failed to receive any final output files from ComfyUI.", None
        return
    
    yield "Status: Loaded successfully!", all_local_file_paths