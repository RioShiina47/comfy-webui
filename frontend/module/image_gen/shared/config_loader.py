import os
import yaml
from core.yaml_loader import load_and_merge_yaml, deep_merge_dicts

_model_config = None
_model_defaults = None
_controlnet_models_config = None
_diffsynth_controlnet_models_config = None
_anima_controlnet_lllite_models_config = None
_krea2_controlnet_models_config = None
_ipadapter_presets_config = None
_constants_config = None
_features_config = None
_architectures_config = None
_pid_config = None

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
        global_constants = load_and_merge_yaml("ui_constants.yaml")
        local_constants = _load_local_yaml("constants.yaml")
        _constants_config = deep_merge_dicts(global_constants, local_constants)
    return _constants_config

def load_architectures_config():
    global _architectures_config
    if _architectures_config is None:
        _architectures_config = _load_local_yaml("model_architectures.yaml")
    return _architectures_config

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

def load_anima_controlnet_lllite_models():
    global _anima_controlnet_lllite_models_config
    if _anima_controlnet_lllite_models_config is None:
        _anima_controlnet_lllite_models_config = _load_local_yaml("anima_controlnet_lllite_models.yaml")
    return _anima_controlnet_lllite_models_config

def load_diffsynth_controlnet_models():
    global _diffsynth_controlnet_models_config
    if _diffsynth_controlnet_models_config is None:
        config = _load_local_yaml("diffsynth_controlnet_models.yaml")
        _diffsynth_controlnet_models_config = config.get("DiffSynth_ControlNet", {})
    return _diffsynth_controlnet_models_config

def load_krea2_controlnet_models():
    global _krea2_controlnet_models_config
    if _krea2_controlnet_models_config is None:
        config = _load_local_yaml("krea2_controlnet_models.yaml")
        _krea2_controlnet_models_config = config.get("Krea2_ControlNet", [])
    return _krea2_controlnet_models_config

def get_krea2_cn_defaults():
    cn_config = load_krea2_controlnet_models()
    if not cn_config:
        return [], None, [], None, "None"
        
    all_types = sorted(list(set(t for model in cn_config for t in model.get("Type", []))))
    default_type = all_types[0] if all_types else None
    
    series_choices = []
    if default_type:
        series_choices = sorted(list(set(model.get("Series", "Default") for model in cn_config if default_type in model.get("Type", []))))
    default_series = series_choices[0] if series_choices else None
    
    filepath = "None"
    if default_series and default_type:
        for model in cn_config:
            if model.get("Series") == default_series and default_type in model.get("Type", []):
                filepath = model.get("Filepath")
                break
                
    return all_types, default_type, series_choices, default_series, filepath


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

def load_pid_config():
    global _pid_config
    if _pid_config is None:
        _pid_config = _load_local_yaml("pid.yaml")
    return _pid_config