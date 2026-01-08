import gradio as gr
import time
import os
import sys
import importlib
from core.config import (
    SERVER_PORT, ENABLE_LOGIN, LOGIN_CREDENTIALS, SHARE_GRADIO, 
    COMFYUI_OUTPUT_PATH, AUTO_DOWNLOAD_MODELS, GRADIO_SERVER_NAME
)
from core.ui_loader import discover_ui_modules, load_ui_layout, load_ui_list
from core.ui_builder import build_gradio_ui

from core import job_manager, node_info_manager, backend_manager


js_shortcut_code = """
function() {
  document.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && event.ctrlKey) {
      event.preventDefault();
      const buttons = document.querySelectorAll('.run-shortcut');
      let visibleButton = null;
      for (const button of buttons) {
        if (button.offsetParent !== null && !button.disabled) {
          visibleButton = button;
          break;
        }
      }
      if (visibleButton) {
        visibleButton.click();
        console.log('Ctrl+Enter shortcut triggered.');
      }
    }
  });
}
"""

def discover_and_register_mcp_modules(app: gr.Blocks):
    print("="*50)
    print("Discovering and registering MCP modules...")
    
    if 'custom' not in sys.path:
        sys.path.insert(0, 'custom')

    ui_dirs_to_scan = ["module", "custom/module"]
    for ui_dir in ui_dirs_to_scan:
        if not os.path.isdir(ui_dir):
            continue
        for root, _, files in os.walk(ui_dir):
            for filename in files:
                if filename.endswith("_mcp.py"):
                    module_path_parts = os.path.normpath(os.path.join(root, filename[:-3])).split(os.sep)
                    module_name = ".".join(module_path_parts)
                    try:
                        module = importlib.import_module(module_name)
                        if hasattr(module, 'MCP_FUNCTIONS') and isinstance(module.MCP_FUNCTIONS, list):
                            for func in module.MCP_FUNCTIONS:
                                gr.api(func)
                                print(f"  ✅ Registered MCP tool: '{func.__name__}' from {module_name}")
                        else:
                            print(f"  ⚠️  Skipping MCP module (no MCP_FUNCTIONS list): {module_name}")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"  ❌ Error loading MCP module {module_name}: {e}")
    print("MCP module registration finished.")
    print("="*50)

def main():
    print("="*50)
    print("Initializing Backend Manager...")
    _ = backend_manager.backend_manager
    print("Backend Manager initialized.")
    print("="*50)

    MAX_RETRIES = 18
    RETRY_DELAY = 10
    node_info_initialized = False

    for i in range(MAX_RETRIES):
        try:
            print("="*50)
            print(f"Attempt {i+1}/{MAX_RETRIES}: Initializing node information from all ComfyUI backends...")
            node_info_manager.fetch_and_cache_object_info()
            print("Node information initialized successfully.")
            print("="*50)
            node_info_initialized = True
            break
        except ConnectionError as e:
            if i < MAX_RETRIES - 1:
                print(f"Warning (Attempt {i+1}/{MAX_RETRIES}): Could not connect to all backends. They might still be starting up.")
                print(f"Details: {e}")
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Application startup failed after {MAX_RETRIES} attempts: {e}")
                print("Please ensure your backend services (ComfyUI) are running and accessible from the frontend container.")
                return

    if not node_info_initialized:
        return

    if AUTO_DOWNLOAD_MODELS:
        try:
            print("="*50)
            print("Starting model check and download process...")
            from core.model_downloader import check_and_download_models
            check_and_download_models()
            print("Model check and download process finished.")
            print("="*50)
        except Exception as e:
            print(f"An error occurred during the model download process: {e}")
            print("Continuing with application startup...")
    else:
        print("="*50)
        print("Skipping automatic model check and download as per config.")
        print("="*50)

    ui_include_list = load_ui_list()
    ui_tree, ui_modules = discover_ui_modules(ui_include_list)
    layout_config = load_ui_layout()
    
    with gr.Blocks(js=js_shortcut_code, title="Comfy web UI") as demo:
        gr.Markdown("# Comfy web UI")
        
        all_components, module_component_map, modules_with_handlers = build_gradio_ui(
            demo, ui_tree, ui_modules, layout_config, SHARE_GRADIO
        )

        print("Binding custom event handlers...")
        for module in modules_with_handlers:
            try:
                module_components = module_component_map.get(module.__name__, {})
                module.create_event_handlers(module_components, all_components, demo)
                print(f"  - Successfully bound events for: {module.__name__}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  - Error binding events for {module.__name__}: {e}")
        
        discover_and_register_mcp_modules(demo)
    
    auth_credentials = None
    if ENABLE_LOGIN and LOGIN_CREDENTIALS:
        auth_credentials = [(cred['username'], cred['password']) for cred in LOGIN_CREDENTIALS]

    print("Launching Gradio interface...")
    demo.queue().launch(
        server_name=GRADIO_SERVER_NAME,
        server_port=SERVER_PORT, 
        mcp_server=True, 
        pwa=True,
        auth=auth_credentials,
        share=SHARE_GRADIO,
        allowed_paths=[COMFYUI_OUTPUT_PATH]
    )

if __name__ == "__main__":
    main()