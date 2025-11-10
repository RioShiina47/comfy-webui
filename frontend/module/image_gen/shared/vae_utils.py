import os
import shutil
import gradio as gr
from core.config import COMFYUI_PATH, CIVITAI_API_KEY
from core.download_utils import get_civitai_file_info, download_file
import hashlib
from .config_loader import load_constants_config

VAE_DIR = os.path.join(COMFYUI_PATH, "models", "vae")
os.makedirs(VAE_DIR, exist_ok=True)

def create_vae_override_ui(components, prefix):
    key = lambda name: f"{prefix}_{name}"
    constants = load_constants_config()
    source_choices = ["None"] + constants.get('LORA_SOURCE_CHOICES', [])

    with gr.Accordion("VAE Settings (Override)", open=False) as vae_accordion:
        components[key('vae_accordion')] = vae_accordion
        with gr.Row():
            components[key('vae_source')] = gr.Dropdown(
                label="VAE Source", 
                choices=source_choices, 
                value="None", 
                scale=1, 
                interactive=True
            )
            components[key('vae_id')] = gr.Textbox(
                label="ID/URL/File", 
                placeholder="e.g., 293549", 
                scale=3, 
                interactive=True
            )
            upload_btn = gr.UploadButton(
                "Upload", 
                file_types=[".safetensors", ".pt", ".bin"], 
                scale=1
            )
            components[key('vae_file')] = gr.State(None)
            upload_btn.upload(
                fn=on_vae_upload, 
                inputs=[upload_btn], 
                outputs=[
                    components[key('vae_id')], 
                    components[key('vae_source')], 
                    components[key('vae_file')]
                ],
                show_api=False
            )

def get_vae_path(source, id_or_url, civitai_key, progress=None):
    if not id_or_url or not id_or_url.strip():
        return None, "No ID or URL provided."
        
    id_or_url = id_or_url.strip()
    file_info = None
    api_key_to_use = None
    source_name = ""
    
    local_path = None
    relative_path = None

    file_ext = ".safetensors"
    if source == "Custom URL" and id_or_url.lower().endswith(('.pt', '.bin')):
        file_ext = os.path.splitext(id_or_url)[1]
        
    if source == "Civitai":
        subdir = "civitai"
        file_info = get_civitai_file_info(id_or_url)
        if file_info and file_info['name'].lower().endswith(('.pt', '.bin')):
            file_ext = os.path.splitext(file_info['name'])[1]
        filename = f"{id_or_url}{file_ext}"
        relative_path = os.path.join(subdir, filename)
        local_path = os.path.join(VAE_DIR, subdir, filename)
        api_key_to_use = civitai_key
        source_name = f"VAE Civitai ID {id_or_url}"
    elif source == "Custom URL":
        subdir = "custom"
        url_hash = hashlib.md5(id_or_url.encode()).hexdigest()
        filename = f"{url_hash}{file_ext}"
        relative_path = os.path.join(subdir, filename)
        local_path = os.path.join(VAE_DIR, subdir, filename)
        file_info = {'downloadUrl': id_or_url}
        api_key_to_use = None
        source_name = f"VAE URL {id_or_url[:30]}..."
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

def on_vae_upload(file_obj):
    if file_obj is None: return gr.update(), gr.update(), None
    
    upload_subdir = "file"
    vae_upload_dir = os.path.join(VAE_DIR, upload_subdir)
    os.makedirs(vae_upload_dir, exist_ok=True)
    
    basename = os.path.basename(file_obj.name)
    new_path = os.path.join(vae_upload_dir, basename)
    shutil.copy(file_obj.name, new_path)
    
    relative_path = os.path.join(upload_subdir, basename)
    
    return relative_path, "File", relative_path

def process_vae_override_input(vals):
    source = vals.get('vae_source')
    id_val = vals.get('vae_id')
    
    if not source or not id_val or source == "None":
        return None

    name = None
    if source == "File":
        name = id_val
    elif source in ["Civitai", "Custom URL"]:
        path, status_msg = get_vae_path(source, id_val, CIVITAI_API_KEY)
        if path is None:
            raise gr.Error(f"VAE '{id_val}' failed to download: {status_msg}")
        name = path
    
    return name