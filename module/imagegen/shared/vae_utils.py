import os
import shutil
import gradio as gr
from core.config import COMFYUI_PATH, CIVITAI_API_KEY
from core.download_utils import resolve_and_download_asset
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
                api_name=False
            )

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
    elif source in ["Civitai", "Hugging Face", "Custom URL"]:
        path, status_msg = resolve_and_download_asset(
            source=source,
            id_or_url=id_val,
            target_dir=VAE_DIR,
            api_key=CIVITAI_API_KEY,
            desc_prefix="VAE"
        )
        if path is None:
            raise gr.Error(f"VAE '{id_val}' failed to download: {status_msg}")
        name = path
    
    return name