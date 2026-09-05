import os
import requests
import hashlib
import gradio as gr
from typing import Optional

def get_civitai_file_info(version_id):
    api_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for file_data in data.get('files', []):
            if file_data.get('type') == 'Model' and file_data['name'].endswith(('.safetensors', '.pt', '.bin', '.ckpt')):
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
        response = requests.get(url, stream=True, headers=headers, timeout=15)
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

def resolve_and_download_asset(
    source: str,
    id_or_url: str,
    target_dir: str,
    api_key: Optional[str] = None,
    hf_token: Optional[str] = None,
    allowed_extensions: tuple = ('.safetensors', '.pt', '.bin', '.ckpt'),
    progress=None,
    desc_prefix: str = "Asset"
):
    """
    Stateless generic asset resolver and downloader.
    Resolves Civitai ID, Hugging Face repo path, or direct URL, caches into target_dir/{civitai|huggingface|custom}/,
    and returns (relative_path, status_message).
    """
    if not id_or_url or not id_or_url.strip():
        return None, "No ID or URL provided."
        
    id_or_url = id_or_url.strip()
    file_info = None
    api_key_to_use = None
    source_name = ""
    file_ext = ".safetensors"
    repo_id = None
    repo_file_path = None

    if source == "Civitai":
        subdir = "civitai"
        file_info = get_civitai_file_info(id_or_url)
        if file_info and file_info.get('name'):
            ext = os.path.splitext(file_info['name'].lower())[1]
            if ext in allowed_extensions:
                file_ext = ext
        filename = f"{id_or_url}{file_ext}"
        api_key_to_use = api_key
        source_name = f"{desc_prefix} Civitai ID {id_or_url}"
    elif source == "Hugging Face":
        subdir = "huggingface"
        parts = id_or_url.split('/')
        if len(parts) < 3:
            return None, "Invalid Hugging Face path. Format: repo_owner/repo_name/filename (or repo_owner/repo_name/subpath/filename)"
        repo_id = f"{parts[0]}/{parts[1]}"
        repo_file_path = "/".join(parts[2:])
        filename = parts[-1]
        source_name = f"{desc_prefix} HF {repo_file_path}"
    elif source == "Custom URL":
        subdir = "custom"
        ext = os.path.splitext(id_or_url.lower())[1]
        if ext in allowed_extensions:
            file_ext = ext
        url_hash = hashlib.md5(id_or_url.encode()).hexdigest()
        filename = f"{url_hash}{file_ext}"
        file_info = {'downloadUrl': id_or_url}
        api_key_to_use = None
        source_name = f"{desc_prefix} URL {id_or_url[:30]}..."
    else:
        return None, f"Invalid or unsupported source: {source}."

    os.makedirs(target_dir, exist_ok=True)
    relative_path = os.path.join(subdir, filename)
    local_path = os.path.join(target_dir, subdir, filename)

    if os.path.exists(local_path):
        return relative_path, "File already exists."

    if source == "Hugging Face":
        import shutil
        from huggingface_hub import hf_hub_download
        token_to_use = hf_token
        if token_to_use is None:
            from core.config import HUGGINGFACE_TOKEN
            token_to_use = HUGGINGFACE_TOKEN or os.getenv("HF_TOKEN") or None
        try:
            if progress: progress(0, desc=f"Downloading {source_name}")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            cached_path = hf_hub_download(repo_id=repo_id, filename=repo_file_path, token=token_to_use)
            try:
                if os.path.lexists(local_path):
                    os.remove(local_path)
                os.symlink(cached_path, local_path)
            except (OSError, NotImplementedError):
                shutil.copy2(cached_path, local_path)
            if progress: progress(1.0, desc=f"Downloaded {source_name}")
            return relative_path, f"Successfully downloaded: {filename}"
        except Exception as e:
            return None, f"Hugging Face download failed for {filename}: {e}"

    if not file_info or not file_info.get('downloadUrl'):
        return None, f"Could not get download link for {source_name}."

    status = download_file(file_info['downloadUrl'], local_path, api_key_to_use, progress=progress, desc=f"Downloading {source_name}")
    
    if "Successfully" in status or "already exists" in status:
        return relative_path, status
    else:
        return None, status