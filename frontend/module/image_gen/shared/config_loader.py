import os
import yaml
from core.yaml_loader import load_and_merge_yaml, deep_merge_dicts

_model_config = None
_model_defaults = None
_controlnet_models_config = None
_ipadapter_presets_config = None
_constants_config = None
_features_config = None

def _load_local_yaml(filename: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    base_config_path = os.path.join(base_dir, "..", "yaml", filename)
    
    custom_config_path = os.path.join(base_dir, "..", "..", "..", "custom", "module", "image_gen", "yaml", filename)
    
    base_config = {}
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r', encoding='utf-8') as f:
            try:
                base_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"Warning: Error parsing image_gen base config '{filename}': {e}")

    custom_config = {}
    if os.path.exists(custom_config_path):
        with open(custom_config_path, 'r', encoding='utf-8') as f:
            try:
                custom_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"Warning: Error parsing image_gen custom config '{filename}': {e}")

    return deep_merge_dicts(base_config, custom_config)

def load_constants_config():
    global _constants_config
    if _constants_config is None:
        _constants_config = _load_local_yaml("constants.yaml")
    return _constants_config

def load_model_config():
    global _model_config
    if _model_config is None:
        _model_config = _load_local_yaml("model_list.yaml")
    return _model_config

def load_model_defaults():
    global _model_defaults
    if _model_defaults is None:
        _model_defaults = _load_local_yaml("model_defaults.yaml")
    return _model_defaults

def load_controlnet_models():
    global _controlnet_models_config
    if _controlnet_models_config is None:
        config = _load_local_yaml("controlnet_models.yaml")
        _controlnet_models_config = config.get("ControlNet", {})
    return _controlnet_models_config

def load_ipadapter_presets():
    global _ipadapter_presets_config
    if _ipadapter_presets_config is None:
        _ipadapter_presets_config = _load_local_yaml("ipadapter.yaml")
    return _ipadapter_presets_config

def load_features_config():
    global _features_config
    if _features_config is None:
        _features_config = _load_local_yaml("image_gen_features.yaml")
    return _features_config