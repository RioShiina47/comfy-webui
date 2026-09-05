import yaml
import os
from core.yaml_loader import load_and_merge_yaml

def _load_backends_from_env():
    backends = {}
    prefix = "COMFYUI_BACKEND_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            backend_name = key[len(prefix):].lower()
            if value:
                backends[backend_name] = value
    return backends

def load_config():
    config_data = load_and_merge_yaml("config.yaml")
    
    base_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yaml", "config.yaml")
    if not os.path.exists(base_config_path):
         raise FileNotFoundError(
            f"Error: Default configuration file 'config.yaml' not found in the 'yaml' directory!\n"
            f"Please copy 'config.yaml.example' to 'config.yaml' and fill in your 'comfyui_path'."
        )
        
    return config_data

config = load_config()

WAIT_FOR_ALL_BACKENDS = config.get("wait_for_all_backends", True)

env_backends = _load_backends_from_env()
if env_backends:
    COMFYUI_BACKENDS = env_backends
    print("[Config] Loaded ComfyUI backends from environment variables.")
else:
    COMFYUI_BACKENDS = config.get("comfyui_backends", {})
    print("[Config] Loaded ComfyUI backends from YAML configuration file.")

if not isinstance(COMFYUI_BACKENDS, dict) or "default" not in COMFYUI_BACKENDS:
    raise ValueError(
        "Error: 'comfyui_backends' configuration is missing in the config file or environment variables, or the 'default' backend is not defined.\n"
        "Please check your .env file or 'yaml/config.yaml' file and ensure it contains at least one backend URL named 'default'."
    )


COMFYUI_PATH = os.getenv("COMFYUI_PATH", config.get("comfyui_path"))
if not COMFYUI_PATH or not os.path.isdir(COMFYUI_PATH):
    checked_path = os.getenv("COMFYUI_PATH") or config.get("comfyui_path")
    raise ValueError(
        f"Error: 'comfyui_path' configuration is invalid or does not exist.\n"
        f"Please check your 'yaml/config.yaml' or set COMFYUI_PATH environment variable, "
        f"and ensure the path '{checked_path}' is a valid directory."
    )

SAVE_WORKFLOW_TO_JSON = config.get("save_workflow_to_json", True)

CIVITAI_API_KEY = os.getenv("CIVITAI_API_KEY", config.get("civitai_api_key", ""))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", config.get("huggingface_token", ""))

ENABLE_LOGIN = config.get("enable_login", False)
LOGIN_CREDENTIALS = config.get("login_credentials", [])

if ENABLE_LOGIN:
    if not LOGIN_CREDENTIALS or not isinstance(LOGIN_CREDENTIALS, list):
        raise ValueError(
            "Error: 'enable_login' is enabled, but 'login_credentials' is not set correctly in the configuration.\n"
            "Please add your username and password in 'yaml/config.yaml'."
        )
    
    for cred in LOGIN_CREDENTIALS:
        if not isinstance(cred, dict) or not cred.get('username') or not cred.get('password'):
            raise ValueError(
                "Error: 'enable_login' is enabled, but an entry in 'login_credentials' has an incorrect format.\n"
                "You must set a non-empty username and password for each user."
            )

AUTO_DOWNLOAD_MODELS = config.get("auto_download_models", True)

COMFYUI_INPUT_PATH = os.path.join(COMFYUI_PATH, "input")
COMFYUI_OUTPUT_PATH = os.path.join(COMFYUI_PATH, "output")
LORA_DIR = os.path.join(COMFYUI_PATH, "models", "loras")
EMBEDDING_DIR = os.path.join(COMFYUI_PATH, "models", "embeddings")
JSON_SAVE_PATH = os.path.join(COMFYUI_PATH, "JSON")

print("="*50)
print("Configuration Loaded:")
print(f"  Startup Policy: {'Wait for all backends' if WAIT_FOR_ALL_BACKENDS else 'Start with at least one backend'}")
print(f"  ComfyUI Path: {COMFYUI_PATH}")
print("  ComfyUI Backends:")
for name, url in COMFYUI_BACKENDS.items():
    print(f"    - {name}: {url}")
print(f"  Input Directory: {COMFYUI_INPUT_PATH}")
print(f"  Output Directory: {COMFYUI_OUTPUT_PATH}")
print(f"  LoRA Directory: {LORA_DIR}")
print(f"  Embedding Directory: {EMBEDDING_DIR}")
print(f"  JSON Save Directory: {JSON_SAVE_PATH}")
print(f"  Login Enabled: {ENABLE_LOGIN}")
if ENABLE_LOGIN:
    print(f"  Login Users Found: {len(LOGIN_CREDENTIALS)}")
print(f"  Save Workflow to JSON: {SAVE_WORKFLOW_TO_JSON}")
print(f"  Auto Download Models: {AUTO_DOWNLOAD_MODELS}")
print(f"  Civitai API Key: {'Set' if CIVITAI_API_KEY else 'Not set'}")
print(f"  HuggingFace Token: {'Set' if HUGGINGFACE_TOKEN else 'Not set'}")
print("="*50)

os.makedirs(COMFYUI_INPUT_PATH, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(JSON_SAVE_PATH, exist_ok=True)