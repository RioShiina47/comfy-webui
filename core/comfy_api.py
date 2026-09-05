import requests
import json
import uuid
import urllib.parse
import websocket
import tempfile
import shutil
from pathlib import Path
import os

from core.backend_manager import backend_manager
from core.config import SAVE_WORKFLOW_TO_JSON, JSON_SAVE_PATH
from core.utils import get_filename_prefix


def queue_prompt(prompt_workflow, client_id, extra_data=None):
    """
    Queue a workflow prompt to the active ComfyUI backend.
    """
    try:
        if SAVE_WORKFLOW_TO_JSON:
            try:
                filename = f"{get_filename_prefix()}_workflow.json"
                filepath = os.path.join(JSON_SAVE_PATH, filename)
                os.makedirs(JSON_SAVE_PATH, exist_ok=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(prompt_workflow, f, indent=2)
                print(f"[Workflow] Saved to: {filepath}")
            except Exception as e:
                print(f"[Workflow] Warning: Failed to save workflow to JSON file: {e}")

        payload = {"prompt": prompt_workflow, "client_id": client_id}
        if extra_data:
            payload.update(extra_data)

        active_url = backend_manager.get_active_backend_url()
        response = requests.post(f"{active_url}/prompt", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error queuing prompt: {e}")
        return None


def get_history(prompt_id):
    """
    Fetch the execution history for a given prompt_id from ComfyUI /history/{prompt_id}.
    Returns a dict containing the prompt history data, or {} on failure.
    """
    active_url = backend_manager.get_active_backend_url()
    url = f"{active_url}/history/{prompt_id}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching history for prompt {prompt_id}: {e}")
        return {}


def interrupt_execution():
    """
    Interrupt current execution on the active ComfyUI backend by calling /interrupt.
    """
    active_url = backend_manager.get_active_backend_url()
    try:
        response = requests.post(f"{active_url}/interrupt", timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error interrupting execution: {e}")
        return False


def download_file(filename, subfolder="", file_type="output"):
    """
    Download a file from ComfyUI /view endpoint to a local temporary file.
    """
    active_url = backend_manager.get_active_backend_url()
    subfolder_param = subfolder or ""
    url = f"{active_url}/view?filename={urllib.parse.quote_plus(filename)}&subfolder={urllib.parse.quote_plus(subfolder_param)}&type={file_type}"
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            suffix = Path(filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                shutil.copyfileobj(r.raw, tmp_file)
                return tmp_file.name
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file {filename}: {e}")
        return None


def get_output_data(prompt_id, client_id):
    """
    Listen to WebSocket events for a specific prompt_id.
    Retained for backward compatibility.
    """
    active_url = backend_manager.get_active_backend_url()
    parsed_url = urllib.parse.urlparse(active_url)
    ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed_url.netloc}/ws?clientId={client_id}"
    ws = None
    try:
        ws = websocket.create_connection(ws_url, timeout=10)
        while True:
            out = ws.recv()
            if not isinstance(out, str):
                continue

            message = json.loads(out)
            msg_type = message.get('type')
            data = message.get('data', {})

            if msg_type == 'progress':
                progress = f"Progress: {data.get('value')}/{data.get('max')}"
                print(progress, end='\r')
                yield progress

            elif msg_type == 'executing':
                if data.get('prompt_id') == prompt_id:
                    if data.get('node') is None:
                        # Execution complete for this prompt
                        break

            elif msg_type == 'executed':
                if data.get('prompt_id') == prompt_id:
                    output_data = data.get('output', {})
                    has_output = any(
                        isinstance(v, list) and v and isinstance(v[0], dict) and 'filename' in v[0]
                        for v in output_data.values()
                    )
                    if has_output:
                        print(f"\nReceived node output for prompt {prompt_id}.")
                        yield output_data

            elif msg_type == 'execution_error':
                if data.get('prompt_id') == prompt_id:
                    err_msg = data.get('exception_message', 'Unknown error')
                    print(f"\nExecution error in prompt {prompt_id}: {err_msg}")
                    break

            elif msg_type == 'execution_interrupted':
                if data.get('prompt_id') == prompt_id:
                    print(f"\nExecution interrupted for prompt {prompt_id}.")
                    break

        print("\nWebSocket stream finished.")

    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass


def run_workflow_and_get_output(workflow_data):
    """
    High-level generator function used by WebUI modules.
    Follows official ComfyUI API best practices:
    1. Connect to WebSocket first to eliminate race conditions.
    2. Queue prompt with the matching client_id.
    3. Stream real-time progress and node execution status.
    4. Handle execution_error and execution_interrupted gracefully.
    5. Fetch final output files via /history/{prompt_id} (with fallback to executed events).
    6. Download and yield downloaded file paths.
    """
    client_id = uuid.uuid4().hex

    prompt_workflow, extra_data = None, None
    if isinstance(workflow_data, tuple) and len(workflow_data) == 2:
        prompt_workflow, extra_data = workflow_data
    else:
        prompt_workflow = workflow_data

    active_url = backend_manager.get_active_backend_url()
    parsed_url = urllib.parse.urlparse(active_url)
    ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed_url.netloc}/ws?clientId={client_id}"

    ws = None
    prompt_id = None
    executed_outputs = {}

    try:
        # Step 1: Connect WebSocket first to ensure no broadcast messages are missed
        yield "Status: Connecting to ComfyUI...", None
        try:
            ws = websocket.create_connection(ws_url, timeout=10)
        except Exception as e:
            yield f"Error: Failed to connect to ComfyUI WebSocket ({ws_url}): {e}", None
            return

        # Step 2: Queue the prompt
        yield "Status: Sending workflow to ComfyUI...", None
        queue_data = queue_prompt(prompt_workflow, client_id, extra_data)
        if not queue_data or 'prompt_id' not in queue_data:
            yield f"Error: Failed to queue workflow to ComfyUI backend at {active_url}. Please check if ComfyUI is running.", None
            return

        prompt_id = queue_data['prompt_id']
        yield f"Status: Queued (ID: {prompt_id[:8]}...). Waiting for execution...", None

        # Step 3: Listen for WebSocket events until completion or error
        while True:
            out = ws.recv()
            if not isinstance(out, str):
                # Binary frames (e.g. latent previews) can be handled or skipped
                continue

            message = json.loads(out)
            msg_type = message.get('type')
            data = message.get('data', {})

            if msg_type == 'progress':
                val = data.get('value', 0)
                max_val = data.get('max', 0)
                yield f"Status: Progress: {val}/{max_val}", None

            elif msg_type == 'executing':
                # executing with node=None signals that the entire workflow is done
                if data.get('prompt_id') == prompt_id:
                    node = data.get('node')
                    if node is None:
                        break
                    else:
                        yield f"Status: Executing node {node}...", None

            elif msg_type == 'executed':
                if data.get('prompt_id') == prompt_id:
                    node_id = data.get('node')
                    output_data = data.get('output', {})
                    if node_id and output_data:
                        executed_outputs[node_id] = output_data

            elif msg_type == 'execution_error':
                if data.get('prompt_id') == prompt_id:
                    node_id = data.get('node_id', 'Unknown')
                    err_msg = data.get('exception_message', 'Unknown error')
                    err_type = data.get('exception_type', 'Error')
                    yield f"Error: [{err_type}] Node {node_id}: {err_msg}", None
                    return

            elif msg_type == 'execution_interrupted':
                if data.get('prompt_id') == prompt_id:
                    yield "Status: Execution was interrupted.", None
                    return

    except Exception as e:
        yield f"Error: WebSocket communication error: {e}", None
        return
    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass

    if not prompt_id:
        return

    # Step 4: Extract output files via /history (authoritative) or fallback to WS executed events
    yield "Status: Execution complete, fetching outputs...", None

    outputs_to_download = []
    history = get_history(prompt_id)
    if history and prompt_id in history:
        history_outputs = history[prompt_id].get('outputs', {})
        for node_id, node_out in history_outputs.items():
            for key, val in node_out.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and 'filename' in item:
                            outputs_to_download.append(item)

    # Fallback to executed events if history endpoint had no output items
    if not outputs_to_download and executed_outputs:
        for node_id, node_out in executed_outputs.items():
            for key, val in node_out.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and 'filename' in item:
                            outputs_to_download.append(item)

    if not outputs_to_download:
        yield "Error: Failed to receive any output files from ComfyUI.", None
        return

    # Step 5: Download generated files
    all_local_file_paths = []
    total_files = len(outputs_to_download)
    for i, file_info in enumerate(outputs_to_download):
        fname = file_info.get('filename')
        subfolder = file_info.get('subfolder', '')
        file_type = file_info.get('type', 'output')
        yield f"Status: Downloading output {i+1}/{total_files} ({fname})...", None
        local_path = download_file(fname, subfolder, file_type)
        if local_path:
            all_local_file_paths.append(local_path)

    if not all_local_file_paths:
        yield "Error: Failed to download any output files from ComfyUI.", None
        return

    yield "Status: Loaded successfully!", all_local_file_paths


def execute_workflow_and_wait(workflow_data, timeout=300):
    """
    Synchronous helper function for backend and MCP tools.
    Executes a workflow and blocks until completion, returning a dict with:
    {
        'prompt_id': prompt_id,
        'files': [local_temp_paths...],
        'outputs': history_outputs_dict
    }
    Raises RuntimeError on failure.
    """
    client_id = uuid.uuid4().hex

    prompt_workflow, extra_data = None, None
    if isinstance(workflow_data, tuple) and len(workflow_data) == 2:
        prompt_workflow, extra_data = workflow_data
    else:
        prompt_workflow = workflow_data

    active_url = backend_manager.get_active_backend_url()
    parsed_url = urllib.parse.urlparse(active_url)
    ws_scheme = "wss" if parsed_url.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed_url.netloc}/ws?clientId={client_id}"

    ws = None
    try:
        ws = websocket.create_connection(ws_url, timeout=10)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to ComfyUI WebSocket ({ws_url}): {e}")

    prompt_id = None
    executed_outputs = {}

    try:
        queue_data = queue_prompt(prompt_workflow, client_id, extra_data)
        if not queue_data or 'prompt_id' not in queue_data:
            raise RuntimeError(f"Failed to queue workflow to ComfyUI backend at {active_url}.")

        prompt_id = queue_data['prompt_id']

        while True:
            out = ws.recv()
            if not isinstance(out, str):
                continue

            message = json.loads(out)
            msg_type = message.get('type')
            data = message.get('data', {})

            if msg_type == 'executing':
                if data.get('prompt_id') == prompt_id and data.get('node') is None:
                    break

            elif msg_type == 'executed':
                if data.get('prompt_id') == prompt_id:
                    node_id = data.get('node')
                    output_data = data.get('output', {})
                    if node_id and output_data:
                        executed_outputs[node_id] = output_data

            elif msg_type == 'execution_error':
                if data.get('prompt_id') == prompt_id:
                    err_msg = data.get('exception_message', 'Unknown error')
                    node_id = data.get('node_id', 'Unknown')
                    raise RuntimeError(f"ComfyUI execution error in node {node_id}: {err_msg}")

            elif msg_type == 'execution_interrupted':
                if data.get('prompt_id') == prompt_id:
                    raise RuntimeError("ComfyUI execution was interrupted.")

    finally:
        if ws:
            try:
                ws.close()
            except Exception:
                pass

    outputs_to_download = []
    history = get_history(prompt_id)
    history_outputs = {}
    if history and prompt_id in history:
        history_outputs = history[prompt_id].get('outputs', {})
        for node_id, node_out in history_outputs.items():
            for key, val in node_out.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and 'filename' in item:
                            outputs_to_download.append(item)

    if not outputs_to_download and executed_outputs:
        for node_id, node_out in executed_outputs.items():
            for key, val in node_out.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and 'filename' in item:
                            outputs_to_download.append(item)

    downloaded_files = []
    for item in outputs_to_download:
        fname = item.get('filename')
        subfolder = item.get('subfolder', '')
        file_type = item.get('type', 'output')
        local_path = download_file(fname, subfolder, file_type)
        if local_path:
            downloaded_files.append(local_path)

    return {
        'prompt_id': prompt_id,
        'files': downloaded_files,
        'outputs': history_outputs or executed_outputs,
        'output_files_info': outputs_to_download
    }


def format_gradio_file_url(file_path: str, request=None) -> str:
    """
    Format a local file path into a Gradio-accessible URL (/gradio_api/file=...).
    Takes request headers (host/x-forwarded-proto) into account if available.
    """
    server_port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860")))
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")

    if request and hasattr(request, "headers") and request.headers and "host" in request.headers:
        scheme = request.headers.get("x-forwarded-proto", "http")
        base_url = f"{scheme}://{request.headers['host']}"
    else:
        base_url = f"http://{server_name}:{server_port}"

    return f"{base_url}/gradio_api/file={urllib.parse.quote(file_path)}"


def resolve_output_file_url(file_info: dict, request=None, local_download_path: str = None) -> str:
    """
    Resolve a ComfyUI output file (dict with 'filename', 'subfolder') to a Gradio URL.
    Prefers the local COMFYUI_OUTPUT_PATH if available and exists, otherwise falls back to local_download_path.
    """
    from core.config import COMFYUI_OUTPUT_PATH

    filename = file_info.get('filename', '')
    subfolder = file_info.get('subfolder', '')
    target_path = os.path.join(COMFYUI_OUTPUT_PATH, subfolder, filename)

    if not os.path.exists(target_path) and local_download_path and os.path.exists(local_download_path):
        target_path = local_download_path

    return format_gradio_file_url(target_path, request)

