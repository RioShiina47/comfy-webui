import gradio as gr
import os
from .config import CIVITAI_API_KEY, LORA_DIR
from .download_utils import resolve_and_download_asset

def process_lora_inputs(all_ui_values: dict, prefix: str):
    """
    Processes LoRA-related UI values from a given prefix.
    Handles file downloads for Civitai/URL sources and constructs the lora_chain list.
    """
    key = lambda name: f"{prefix}_{name}"
    
    loras = []
    lora_sources = all_ui_values.get(key('loras_sources'), [])
    if not lora_sources:
        return []
        
    lora_ids_txt = all_ui_values.get(key('loras_ids'), [])
    lora_ids_dd = all_ui_values.get(key('loras_file_dropdowns'), [])
    lora_scales = all_ui_values.get(key('loras_scales'), [])
    
    for i in range(len(lora_sources)):
        scale = lora_scales[i] if i < len(lora_scales) else 1.0
        if scale is not None and scale != 0:
            name = None
            src = lora_sources[i] if i < len(lora_sources) else None

            id_val = None
            if src == "File":
                id_val = lora_ids_dd[i] if i < len(lora_ids_dd) else None
            else:
                id_val = lora_ids_txt[i] if i < len(lora_ids_txt) else None

            if src == "Upload File" and id_val:
                name = id_val
            elif src == "File" and id_val:
                os_specific_subpath = id_val.replace("/", os.sep)
                name = os.path.join("file", os_specific_subpath)
            elif src in ["Civitai", "Hugging Face", "Custom URL"] and id_val:
                path, status_msg = resolve_and_download_asset(
                    source=src,
                    id_or_url=id_val,
                    target_dir=LORA_DIR,
                    api_key=CIVITAI_API_KEY,
                    desc_prefix="LoRA"
                )
                if path is None:
                    raise gr.Error(f"LoRA '{id_val}' failed to download: {status_msg}")
                name = path
            
            if name:
                loras.append({"lora_name": name, "strength_model": scale, "strength_clip": scale})
    return loras