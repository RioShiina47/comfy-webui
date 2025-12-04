import os
import re
import shutil
import random
from PIL import Image
import numpy as np
from core.config import LORA_DIR, COMFYUI_INPUT_PATH
from .config_loader import load_model_config, load_constants_config

_constants = None
def get_constants():
    global _constants
    if _constants is None:
        _constants = load_constants_config()
    return _constants

_model_path_cache = {}
def get_model_path(display_name):
    global _model_path_cache
    if not display_name: return None
    if display_name in _model_path_cache: return _model_path_cache[display_name]
    
    model_config = load_model_config()
    checkpoints = model_config.get("Checkpoints", {})

    for arch_name, arch_data in checkpoints.items():
        for model in arch_data.get("models", []):
            if model.get("display_name") == display_name:
                path_or_components = model.get("path") or model.get("components")
                _model_path_cache[display_name] = path_or_components
                return path_or_components
            
    return None

def get_model_type(selected_model_name: str, model_config: dict) -> str:
    checkpoints = model_config.get("Checkpoints", {})
    for arch_name, arch_data in checkpoints.items():
        for model in arch_data.get("models", []):
            if model.get("display_name") == selected_model_name:
                return arch_name.lower().replace(" ", "-").replace(".", "")
        
    return "sdxl"

def get_latent_type_for_model(selected_model_name: str) -> str:
    model_config = load_model_config()
    checkpoints = model_config.get("Checkpoints", {})
    for arch_name, arch_data in checkpoints.items():
        for model in arch_data.get("models", []):
            if model.get("display_name") == selected_model_name:
                return arch_data.get("latent_type", "latent")
    return "latent"

def parse_generation_parameters_for_ui(full_prompt_text: str):
    if not full_prompt_text: return {}
    data = {}
    neg_prompt_keyword = "Negative prompt:"
    parts = re.split(neg_prompt_keyword, full_prompt_text, flags=re.IGNORECASE)
    data['positive_prompt'] = parts[0].strip()
    params_text = ""
    if len(parts) > 1:
        remaining_lines = parts[1].strip().split('\n'); data['negative_prompt'] = remaining_lines[0].strip()
        params_text = "\n".join(remaining_lines[1:])
    else:
        prompt_lines = data['positive_prompt'].split('\n')
        if len(prompt_lines) > 1:
            data['positive_prompt'] = prompt_lines[0].strip()
            potential_params_text = "\n".join(prompt_lines[1:])
            
            if "Negative prompt:" in potential_params_text:
                 parts_again = re.split(neg_prompt_keyword, potential_params_text, flags=re.IGNORECASE)
                 data['negative_prompt'] = parts_again[1].strip().split('\n')[0]
                 params_text = parts_again[1]
            else:
                 params_text = potential_params_text

    param_map = {
        'steps': (r"\b(Steps|steps): ([^,]+)", int), 'sampler_name': (r"\b(Sampler|sampler_name): ([^,]+)", str),
        'scheduler': (r"\b(Scheduler|scheduler): ([^,]+)", str), 'cfg': (r"\b(CFG scale|cfg): ([^,]+)", float),
        'seed': (r"\b(Seed|seed): ([^,]+)", int), 'clip_skip': (r"\b(Clip skip|clip_skip): ([^,]+)", int),
    }
    sampler_map = get_constants().get('SAMPLER_MAP', {})
    for key, (pattern, cast_type) in param_map.items():
        match = re.search(pattern, params_text, re.IGNORECASE)
        if match:
            try:
                value = match.group(2).strip()
                if key == 'sampler_name': data[key] = sampler_map.get(value.lower(), value.lower())
                else: data[key] = cast_type(value)
            except (ValueError, TypeError): pass
    if (m := re.search(r"Size: (\d+)x(\d+)", params_text, re.IGNORECASE)): data.update({'width': int(m.group(1)), 'height': int(m.group(2))})
    if (m := re.search(r"\b(Model|model_name): ([^,]+)", params_text, re.IGNORECASE)):
        parsed_model = m.group(2).strip(); config = load_model_config()
        all_models = [
            model for arch_data in config.get("Checkpoints", {}).values()
            for model in arch_data.get("models", [])
        ]
        found = next((m['display_name'] for m in all_models if m['display_name'] == parsed_model), None)
        if not found: found = next((m['display_name'] for m in all_models if parsed_model in m['display_name']), None)
        if found: data['model_name'] = found
    return data

def get_model_generation_defaults(model_display_name: str, model_type: str, defaults_config: dict):
    final_defaults = {
        'steps': 25, 'cfg': 7.0, 'sampler_name': 'euler', 'scheduler': 'simple',
        'positive_prompt': '', 'negative_prompt': ''
    }

    if 'Default' in defaults_config:
        final_defaults.update(defaults_config['Default'])

    model_type_key = next((key for key in defaults_config if key.lower() == model_type.lower()), None)
    if model_type_key:
        model_type_config = defaults_config[model_type_key]
        if '_defaults' in model_type_config:
            final_defaults.update(model_type_config['_defaults'])
            
        if model_display_name in model_type_config:
            final_defaults.update(model_type_config[model_display_name])

    return final_defaults