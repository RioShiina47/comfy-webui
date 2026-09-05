import os
import yaml
from collections.abc import Mapping

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
BASE_YAML_DIR = os.path.join(ROOT_DIR, "yaml")

_config_cache = {}

def deep_merge_dicts(base, custom):
    if not isinstance(base, Mapping) or not isinstance(custom, Mapping):
        return custom

    merged = base.copy()
    for key, custom_value in custom.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(custom_value, Mapping):
            merged[key] = deep_merge_dicts(base_value, custom_value)
        else:
            merged[key] = custom_value
    return merged

def load_and_merge_yaml(filename: str):
    if filename in _config_cache:
        return _config_cache[filename]

    base_config_path = os.path.join(BASE_YAML_DIR, filename)
    base_config = {}
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r', encoding='utf-8') as f:
            try:
                base_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"Warning: Error parsing configuration file {filename}: {e}")
                base_config = {}

    _config_cache[filename] = base_config
    return base_config

def load_and_merge_yaml_from_module(module_path: str, filename: str):
    module_yaml_dir = os.path.join(module_path, "yaml")
    base_config_path = os.path.join(module_yaml_dir, filename)
    base_config = {}
    if os.path.exists(base_config_path):
        with open(base_config_path, 'r', encoding='utf-8') as f:
            try:
                base_config = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                print(f"Warning: Error parsing module configuration file {filename}: {e}")
                base_config = {}

    return base_config
