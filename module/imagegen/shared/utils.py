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



def get_model_generation_defaults(model_display_name: str, model_type: str, defaults_config: dict):
    final_defaults = {
        'steps': 25, 'cfg': 7.0, 'sampler_name': 'euler', 'scheduler': 'simple',
        'positive_prompt': '', 'negative_prompt': ''
    }

    if 'Default' in defaults_config:
        final_defaults.update(defaults_config['Default'])

    model_type_key = next((key for key in defaults_config if key.lower().replace(" ", "-").replace(".", "") == model_type.lower()), None)
    if model_type_key:
        model_type_config = defaults_config[model_type_key]
        if '_defaults' in model_type_config:
            final_defaults.update(model_type_config['_defaults'])
            
        if model_display_name in model_type_config:
            final_defaults.update(model_type_config[model_display_name])

    return final_defaults