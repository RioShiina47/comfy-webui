import os
import requests
import hashlib
import gradio as gr
from core.config import LORA_DIR, EMBEDDING_DIR, HTTP_PROXY, HTTPS_PROXY

os.makedirs(LORA_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

def _get_proxies():
    proxies = {}
    if HTTP_PROXY:
        proxies['http'] = HTTP_PROXY
    if HTTPS_PROXY:
        proxies['https'] = HTTPS_PROXY
    return proxies if proxies else None

def get_civitai_file_info(version_id):
    api_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    try:
        response = requests.get(api_url, timeout=10, proxies=_get_proxies())
        response.raise_for_status()
        data = response.json()
        
        for file_data in data.get('files', []):
            if file_data.get('type') == 'Model' and file_data['name'].endswith(('.safetensors', '.pt')):
                return file_data
        
        if data.get('files'):
            return data['files'][0]
            
    except Exception as e:
        print(f"Error getting Civitai info for version {version_id}: {e}")
        return None

def download_file(url, save_path, api_key=None, progress=None, desc=""):
    if os.path.exists(save_path):
        return f"File already exists: {os.path.basename(save_path)}"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    headers = {'Authorization': f'Bearer {api_key}'} if api_key and api_key.strip() else {}
    try:
        if progress: progress(0, desc=desc)
        response = requests.get(url, stream=True, headers=headers, timeout=15, proxies=_get_proxies())
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                if progress and total_size > 0:
                    downloaded += len(chunk)
                    progress(downloaded / total_size, desc=desc)
                    
        return f"Successfully downloaded: {os.path.basename(save_path)}"
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return f"Download failed for {os.path.basename(save_path)}: {e}"

def get_lora_path(source, id_or_url, civitai_key, progress=None):
    if not id_or_url or not id_or_url.strip():
        return None, "No ID or URL provided."
        
    id_or_url = id_or_url.strip()
    file_info = None
    api_key_to_use = None
    source_name = ""
    
    local_path = None
    relative_path = None

    file_ext = ".safetensors"
    if source == "Custom URL" and id_or_url.lower().endswith('.pt'):
        file_ext = ".pt"
        
    if source == "Civitai":
        subdir = "civitai"
        file_info = get_civitai_file_info(id_or_url)
        if file_info and file_info['name'].lower().endswith('.pt'):
            file_ext = ".pt"
        filename = f"{id_or_url}{file_ext}"
        relative_path = os.path.join(subdir, filename)
        local_path = os.path.join(LORA_DIR, subdir, filename)
        api_key_to_use = civitai_key
        source_name = f"LoRA Civitai ID {id_or_url}"
    elif source == "Custom URL":
        subdir = "custom"
        url_hash = hashlib.md5(id_or_url.encode()).hexdigest()
        filename = f"{url_hash}{file_ext}"
        relative_path = os.path.join(subdir, filename)
        local_path = os.path.join(LORA_DIR, subdir, filename)
        file_info = {'downloadUrl': id_or_url}
        api_key_to_use = None
        source_name = f"LoRA URL {id_or_url[:30]}..."
    else:
        return None, "Invalid source."

    if os.path.exists(local_path):
        return relative_path, "File already exists."

    if not file_info or not file_info.get('downloadUrl'):
        return None, f"Could not get download link for {source_name}."

    status = download_file(file_info['downloadUrl'], local_path, api_key_to_use, progress=progress, desc=f"Downloading {source_name}")
    
    if "Successfully" in status or "already exists" in status:
        return relative_path, status
    else:
        return None, status

def get_embedding_path(source, id_or_url, civitai_key, progress=None):
    if not id_or_url or not id_or_url.strip():
        return None, "No ID or URL provided."
        
    id_or_url = id_or_url.strip()
    file_info = None
    api_key_to_use = None
    source_name = ""
    
    local_path = None
    relative_path = None

    file_ext = ".safetensors"
    if source == "Custom URL" and id_or_url.lower().endswith('.pt'):
        file_ext = ".pt"

    if source == "Civitai":
        subdir = "civitai"
        file_info = get_civitai_file_info(id_or_url)
        if file_info and file_info['name'].lower().endswith('.pt'):
            file_ext = ".pt"
        filename = f"{id_or_url}{file_ext}"
        relative_path = os.path.join(subdir, filename)
        local_path = os.path.join(EMBEDDING_DIR, subdir, filename)
        api_key_to_use = civitai_key
        source_name = f"Embedding Civitai ID {id_or_url}"
    elif source == "Custom URL":
        subdir = "custom"
        url_hash = hashlib.md5(id_or_url.encode()).hexdigest()
        filename = f"{url_hash}{file_ext}"
        relative_path = os.path.join(subdir, filename)
        local_path = os.path.join(EMBEDDING_DIR, subdir, filename)
        file_info = {'downloadUrl': id_or_url}
        api_key_to_use = None
        source_name = f"Embedding URL {id_or_url[:30]}..."
    else:
        return None, "Invalid source."

    if os.path.exists(local_path):
        return relative_path, "File already exists."

    if not file_info or not file_info.get('downloadUrl'):
        return None, f"Could not get download link for {source_name}."

    status = download_file(file_info['downloadUrl'], local_path, api_key_to_use, progress=progress, desc=f"Downloading {source_name}")
    
    if "Successfully" in status or "already exists" in status:
        return relative_path, status
    else:
        return None, status