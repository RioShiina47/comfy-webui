import os
import yaml
import shutil
import subprocess
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from core.config import (
    COMFYUI_PATH, HF_CACHE_PATH, CIVITAI_API_KEY, HTTP_PROXY, HTTPS_PROXY,
    HUGGINGFACE_TOKEN
)
from core.yaml_loader import load_and_merge_yaml

def _get_proxies():
    proxies = {}
    if HTTP_PROXY: proxies['http'] = HTTP_PROXY
    if HTTPS_PROXY: proxies['https'] = HTTPS_PROXY
    return proxies or None

def _get_civitai_final_url(model_version_id: str):
    initial_url = f"https://civitai.com/api/download/models/{model_version_id}"
    headers = {}
    if CIVITAI_API_KEY:
        headers['Authorization'] = f'Bearer {CIVITAI_API_KEY}'
    
    try:
        with requests.get(initial_url, headers=headers, allow_redirects=False, stream=True, proxies=_get_proxies(), timeout=20) as r:
            if r.status_code in [301, 302, 307] and 'Location' in r.headers:
                return r.headers['Location']
            else:
                api_url = f"https://civitai.com/api/v1/model-versions/{model_version_id}"
                response = requests.get(api_url, timeout=10, proxies=_get_proxies(), headers=headers)
                response.raise_for_status()
                data = response.json()
                if data.get('files'):
                    return data['files'][0].get('downloadUrl')
                return None
    except requests.RequestException as e:
        print(f"  ❌ Error resolving Civitai URL: {e}")
        return None

def _download_with_hf(file_info, destination_path):
    try:
        cached_path = hf_hub_download(
            repo_id=file_info['repo_id'],
            filename=file_info['repository_file_path'],
            cache_dir=HF_CACHE_PATH,
            local_dir_use_symlinks=False,
            token=HUGGINGFACE_TOKEN
        )
        shutil.move(cached_path, destination_path)
        tqdm.write(f"  ✔ Successfully downloaded with Hugging Face Hub.")
        return True
    except Exception as e:
        tqdm.write(f"  ❌ [HF Download Error] {e}")
        return False

def _download_with_aria2(url, destination_path):
    command = [
        "aria2c",
        '--console-log-level=warn', '--summary-interval=0',
        '-x', '16', '-s', '16', '-k', '1M',
        '-d', os.path.dirname(destination_path),
        '-o', os.path.basename(destination_path),
        url
    ]
    
    env = os.environ.copy()
    if HTTP_PROXY: env['http_proxy'] = HTTP_PROXY
    if HTTPS_PROXY: env['https_proxy'] = HTTPS_PROXY

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', env=env)
        tqdm.write(f"  ✔ Successfully downloaded with Aria2.")
        return True
    except FileNotFoundError:
        tqdm.write("  ❌ [Aria2 Execution Error] 'aria2c' command not found. The 'aria2' wheel might not be installed correctly.")
        return False
    except subprocess.CalledProcessError as e:
        tqdm.write(f"  ❌ [Aria2 Download Error] Stderr: {e.stderr}")
        return False
    except Exception as e:
        tqdm.write(f"  ❌ [Aria2 Execution Error] {e}")
        return False

def _download_with_requests(url, destination_path):
    try:
        with requests.get(url, stream=True, proxies=_get_proxies(), timeout=20) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(destination_path, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True,
                desc=f"  Downloading {os.path.basename(destination_path)}",
                leave=False
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    pbar.update(len(chunk))
                    f.write(chunk)

        if total_size != 0 and os.path.getsize(destination_path) != total_size:
            raise IOError("Downloaded file size does not match expected size.")
            
        tqdm.write(f"  ✔ Successfully downloaded with Requests.")
        return True
    except Exception as e:
        tqdm.write(f"  ❌ [Requests Download Error] {e}")
        if os.path.exists(destination_path):
            os.remove(destination_path)
        return False

def check_and_download_models():
    global_file_config = load_and_merge_yaml("file_list.yaml")
    
    module_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "module"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "custom", "module")
    ]
    
    all_module_configs = []
    
    for module_dir in module_dirs:
        if not os.path.exists(module_dir):
            continue
            
        for root, dirs, files in os.walk(module_dir):
            for filename in files:
                if filename == "file_list.yaml":
                    module_config_path = os.path.join(root, filename)
                    try:
                        with open(module_config_path, 'r', encoding='utf-8') as f:
                            module_config = yaml.safe_load(f)
                            if module_config and isinstance(module_config, dict):
                                module_config['module_path'] = root
                                all_module_configs.append(module_config)
                    except Exception as e:
                        print(f"Warning: Could not load module file_list.yaml from {module_config_path}: {e}")
    
    all_files_to_check = []
    
    if 'file' in global_file_config:
        for category, files in global_file_config['file'].items():
            if isinstance(files, list):
                for file_info in files:
                    file_info = file_info.copy()
                    file_info['category'] = category
                    file_info['source_module'] = "global"
                    all_files_to_check.append(file_info)
    
    for module_config in all_module_configs:
        if 'file' in module_config:
            for category, files in module_config['file'].items():
                if isinstance(files, list):
                    for file_info in files:
                        file_info = file_info.copy()
                        file_info['category'] = category
                        file_info['source_module'] = module_config.get('module_path', 'unknown')
                        all_files_to_check.append(file_info)
    
    if not all_files_to_check:
        print("No files listed in any file_list.yaml or custom/yaml/file_list.yaml.")
        return

    pbar = tqdm(all_files_to_check, desc="Checking Models")
    for file_info in pbar:
        category, filename, source = file_info['category'], file_info['filename'], file_info['source']
        module_path = file_info.get('source_module', 'unknown')
        pbar.set_postfix_str(f"{category}/{filename}")

        destination_dir = os.path.join(COMFYUI_PATH, "models", category)
        destination_path = os.path.join(destination_dir, filename)
        
        try:
            os.makedirs(destination_dir, exist_ok=True)
        except OSError as e:
            tqdm.write(f"\nError: Could not create directory '{destination_dir}'. Skipping '{filename}'. Error: {e}")
            continue

        if os.path.exists(destination_path):
            continue

        tqdm.write(f"\nMissing file: {filename}. Starting download...")
        
        download_url = None
        if source == 'hf':
            download_url = f"https://huggingface.co/{file_info['repo_id']}/resolve/main/{file_info['repository_file_path']}"

        elif source == 'civitai':
            tqdm.write("  -> Resolving Civitai redirect URL...")
            download_url = _get_civitai_final_url(file_info.get('model_version_id'))
        
        if not download_url:
            tqdm.write(f"  ❌ Could not get a valid download URL for {filename}. Skipping.")
            continue

        download_successful = False

        if source == 'hf' and HUGGINGFACE_TOKEN and HF_CACHE_PATH:
            tqdm.write("  -> Attempting download with Hugging Face Hub library...")
            if _download_with_hf(file_info, destination_path):
                download_successful = True

        if not download_successful and shutil.which("aria2c"):
            tqdm.write("  -> Attempting download with Aria2 (Fallback)...")
            if _download_with_aria2(download_url, destination_path):
                download_successful = True

        if not download_successful:
            tqdm.write("  -> Attempting download with standard Python Requests (Final Fallback)...")
            if _download_with_requests(download_url, destination_path):
                download_successful = True
        
        if not download_successful:
            tqdm.write(f"  ❌ All download methods failed for {filename}.")
