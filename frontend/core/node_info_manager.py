import requests
from concurrent.futures import ThreadPoolExecutor

from core.backend_manager import backend_manager
from core.config import WAIT_FOR_ALL_BACKENDS

_node_info_cache = {}

def _fetch_info_from_backend(backend_name, backend_url):
    api_url = f"{backend_url}/object_info"
    try:
        response = requests.get(api_url, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def fetch_and_cache_object_info():
    global _node_info_cache
    if _node_info_cache:
        print("[NodeInfoManager] Node info already cached.")
        return

    all_backends = backend_manager.backends
    if not all_backends:
        raise ConnectionError("No backends configured in BackendManager.")

    all_results = {}
    
    print("[NodeInfoManager] Starting node info fetch from all backends...")
    with ThreadPoolExecutor(max_workers=len(all_backends)) as executor:
        future_to_backend = {executor.submit(_fetch_info_from_backend, name, url): name for name, url in all_backends.items()}
        
        for future in future_to_backend:
            backend_name = future_to_backend[future]
            try:
                result = future.result()
                all_results[backend_name] = result
            except Exception:
                all_results[backend_name] = None

    successful_backends = set()
    failed_backends = set()
    print("-" * 25)
    print("Backend Connection Status:")
    for backend_name, result in all_results.items():
        if result is not None:
            print(f"  ✅ SUCCESS: '{backend_name}'")
            successful_backends.add(backend_name)
        else:
            print(f"  ❌ FAILED:  '{backend_name}'")
            failed_backends.add(backend_name)
    print("-" * 25)

    if WAIT_FOR_ALL_BACKENDS:
        if failed_backends:
            error_message = (
                f"Strict mode enabled. Failed to connect to required backend(s): {', '.join(sorted(list(failed_backends)))}"
            )
            raise ConnectionError(error_message)
    else:
        if not successful_backends:
            error_message = (
                "Lenient mode enabled, but failed to connect to ANY backend."
            )
            raise ConnectionError(error_message)

    merged_info = {}
    for backend_name in successful_backends:
        info_dict = all_results.get(backend_name)
        if info_dict:
            merged_info.update(info_dict)
            print(f"[NodeInfoManager] Merged {len(info_dict)} nodes from '{backend_name}'.")

    _node_info_cache = merged_info
    print(f"[NodeInfoManager] Successfully initialized with nodes from {len(successful_backends)} backend(s).")

def get_node_info(class_type: str):
    return _node_info_cache.get(class_type)

def get_all_node_info():
    return _node_info_cache

def get_node_input_options(class_type: str, input_name: str) -> list:
    node_info = get_node_info(class_type)
    if not node_info:
        print(f"[NodeInfoManager] Warning: Could not find node info for class_type '{class_type}'")
        return []

    required_inputs = node_info.get("input", {}).get("required", {})
    if input_name in required_inputs:
        details = required_inputs[input_name]
        if isinstance(details, list) and len(details) > 0 and isinstance(details[0], list):
            return details[0]
            
    optional_inputs = node_info.get("input", {}).get("optional", {})
    if input_name in optional_inputs:
        details = optional_inputs[input_name]
        if isinstance(details, list) and len(details) > 0 and isinstance(details[0], list):
            return details[0]

    print(f"[NodeInfoManager] Warning: Could not find options for input '{input_name}' in node '{class_type}'")
    return []