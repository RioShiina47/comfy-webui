import requests
import threading
from concurrent.futures import ThreadPoolExecutor
from core.config import COMFYUI_BACKENDS

class BackendManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(BackendManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.backends = COMFYUI_BACKENDS
        self.active_backend_name = "default"
        self._initialized = True
        print(f"[BackendManager] Initialized with default backend '{self.active_backend_name}'.")

    def get_active_backend_url(self):
        return self.backends.get(self.active_backend_name)

    def get_all_backend_urls(self):
        return list(self.backends.values())

    def _free_backend_memory(self, backend_name, backend_url):
        try:
            print(f"[BackendManager] Sending /free request to {backend_name} ({backend_url})...")
            response = requests.post(
                f"{backend_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=20
            )
            response.raise_for_status()
            print(f"[BackendManager] Successfully freed memory for {backend_name}.")
        except requests.exceptions.RequestException as e:
            print(f"[BackendManager] Warning: Could not free memory for backend '{backend_name}'. "
                  f"Is the backend running and does it support the /free endpoint? Error: {e}")

    def switch_backend(self, target_backend_name: str):
        if target_backend_name not in self.backends:
            print(f"[BackendManager] Error: Attempted to switch to an unknown backend '{target_backend_name}'. "
                  f"Falling back to 'default'.")
            target_backend_name = "default"
        
        if self.active_backend_name == target_backend_name:
            print(f"[BackendManager] Backend '{target_backend_name}' is already active. No switch needed.")
            return

        print(f"[BackendManager] Switching backend from '{self.active_backend_name}' to '{target_backend_name}'...")
        
        inactive_backends = {name: url for name, url in self.backends.items() if name != target_backend_name}
        
        if inactive_backends:
            print(f"[BackendManager] Freeing up resources on {len(inactive_backends)} inactive backend(s)...")
            with ThreadPoolExecutor(max_workers=len(inactive_backends)) as executor:
                futures = [executor.submit(self._free_backend_memory, name, url) for name, url in inactive_backends.items()]
                for future in futures:
                    future.result()

        self.active_backend_name = target_backend_name
        print(f"[BackendManager] Switch complete. Active backend is now '{self.active_backend_name}'.")

backend_manager = BackendManager()