"""
MCP Tool: get_feature_list
Get the list of supported advanced features along with their usage constraints and parameter schemas.
"""

import os
from copy import deepcopy
from .common import (
    _load_yaml,
    _CHAIN_FEATURES_PATH,
    _TASK_FEATURES_PATH,
    _YAML_DIR,
    _get_ipadapter_presets_by_arch,
)
from .error_schema import make_not_found_error


def _get_default_example_chain_item(chain_name: str, chain_data: dict) -> dict:
    if "example_chain_item" in chain_data and isinstance(chain_data["example_chain_item"], dict):
        return deepcopy(chain_data["example_chain_item"])

    if chain_name == "lora":
        return {
            "injector_type": "lora",
            "source": "Civitai",
            "lora_value": "12345",
            "scale": 1.0,
        }
    elif chain_name == "ipadapter":
        return {
            "injector_type": "ipadapter",
            "image": "https://example.com/reference_image.png",
            "weight": 1.0,
            "preset": "STANDARD (medium strength)",
        }
    elif chain_name in ("controlnet", "diffsynth_controlnet", "krea2_controlnet", "anima_controlnet_lllite"):
        return {
            "injector_type": chain_name,
            "type": "Depth",
            "series": "SDXL" if chain_name == "controlnet" else "Patil",
            "image": "https://example.com/control_depth_map.png",
            "strength": 1.0,
        }
    elif chain_name == "conditioning":
        return {
            "injector_type": "conditioning",
            "prompt": "blue sky with soft clouds",
            "x": 0,
            "y": 0,
            "width": 1024,
            "height": 512,
            "strength": 1.0,
        }
    elif chain_name == "vae":
        return {
            "injector_type": "vae",
            "source": "Civitai",
            "vae_value": "12345",
        }
    elif chain_name == "embedding":
        return {
            "injector_type": "embedding",
            "source": "Civitai",
            "embedding_value": "12345",
        }
    elif chain_name == "pid":
        return {
            "injector_type": "pid",
            "enabled": True,
        }

    injector_type = chain_data.get("chains")
    if isinstance(injector_type, list):
        injector_type = injector_type[0]
    elif not isinstance(injector_type, str):
        injector_type = chain_name

    item = {"injector_type": injector_type}
    schema = chain_data.get("parameters_schema", {})
    props = schema.get("properties", {})
    for p_name, p_info in props.items():
        if p_name == "image":
            item["image"] = "https://example.com/reference.png"
        elif p_name == "source":
            item["source"] = "Civitai"
        elif p_name in ("weight", "strength", "scale", "factor"):
            item[p_name] = p_info.get("default", 1.0)
        elif p_name == "enabled":
            item["enabled"] = True
    return item


def _get_supported_tasks_for_feature(chain_name: str, chain_data: dict, task_features: dict) -> list:
    feat_chains = chain_data.get("chains", chain_name)
    if isinstance(feat_chains, str):
        target_chains = {feat_chains, chain_name}
    elif isinstance(feat_chains, (list, tuple, set)):
        target_chains = set(feat_chains) | {chain_name}
    else:
        target_chains = {chain_name}

    supported = []
    for task_name, task_data in task_features.items():
        if not isinstance(task_data, dict):
            continue
        enabled_chains = task_data.get("enabled_chains", [])
        if any(c in enabled_chains for c in target_chains):
            supported.append(task_name)
    return supported


def _build_feature_entry(chain_name: str, chain_data: dict, task_features: dict = None, include_schema: bool = False) -> dict:
    if task_features is None:
        task_features = _load_yaml(_TASK_FEATURES_PATH)
    example_item = _get_default_example_chain_item(chain_name, chain_data)
    entry = {
        "feature_name": chain_name,
        "chains": chain_data.get("chains", chain_name),
        "display_name": chain_data.get("display_name", chain_name),
        "description": chain_data.get("description", ""),
        "supported_tasks": _get_supported_tasks_for_feature(chain_name, chain_data, task_features),
        "max_count": chain_data.get("max_count", 1),
        "usage_guideline": chain_data.get("usage_guideline", ""),
        "example_chain_item": example_item,
        "example_json_params": {
            "task_type": "txt2img",
            "model": "stabilityai/SDXL-Base-1.0",
            "prompt": "A majestic lion jumping from a big stone at night",
            "width": 1024,
            "height": 1024,
            "chain": [example_item],
        },
    }
    if include_schema:
        schema = deepcopy(chain_data.get("parameters_schema", {}))
        if chain_name in ("krea2_controlnet", "diffsynth_controlnet", "controlnet", "anima_controlnet_lllite"):
            config_key = (
                "Krea2_ControlNet" if chain_name == "krea2_controlnet"
                else "DiffSynth_ControlNet" if chain_name == "diffsynth_controlnet"
                else "Anima_ControlNet_Lllite" if chain_name == "anima_controlnet_lllite"
                else "ControlNet"
            )
            yaml_filename = f"{chain_name}_models.yaml"
            model_path = os.path.join(_YAML_DIR, yaml_filename)
            raw_models = _load_yaml(model_path).get(config_key, [])
            models_list = []
            if isinstance(raw_models, dict):
                for val in raw_models.values():
                    if isinstance(val, list):
                        models_list.extend(val)
                    elif isinstance(val, dict):
                        models_list.append(val)
            elif isinstance(raw_models, list):
                models_list = raw_models

            types_set = set()
            for m in models_list:
                t_val = m.get("Type", [])
                if isinstance(t_val, list):
                    types_set.update(t_val)
                elif isinstance(t_val, str):
                    types_set.add(t_val)
            types = sorted(list(types_set))
            series = sorted(list(set(m.get("Series") for m in models_list if m.get("Series"))))
            if "properties" in schema:
                if "type" in schema["properties"] and types:
                    schema["properties"]["type"]["enum"] = types
                if "series" in schema["properties"] and series:
                    schema["properties"]["series"]["enum"] = series
            if chain_name == "controlnet" and isinstance(raw_models, dict):
                schema["architectures"] = raw_models
        elif chain_name == "ipadapter":
            presets_by_arch = _get_ipadapter_presets_by_arch()
            schema["presets_by_architecture"] = presets_by_arch
            all_presets = sorted(list(set(presets_by_arch.get("SD1.5", []) + presets_by_arch.get("SDXL", []))))
            if "properties" in schema and "preset" in schema["properties"]:
                schema["properties"]["preset"]["enum"] = all_presets
        entry["parameters_schema"] = schema
    return entry


def ImageGen_get_feature_list(feature_name: str | list[str] = "") -> list | dict:
    """
    Dynamically load supported advanced features from chain_features.yaml.

    - If feature_name is empty: returns a summary list of ALL features (excluding parameters_schema)
      to optimize response size and token usage.
    - If feature_name is specified (single feature name, comma-separated string, or list of strings):
      returns complete feature details INCLUDING parameters_schema for the requested feature(s).
    """
    chain_features = _load_yaml(_CHAIN_FEATURES_PATH)
    task_features = _load_yaml(_TASK_FEATURES_PATH)

    targets = []
    is_single_string_query = False

    if isinstance(feature_name, list):
        targets = [str(x).strip() for x in feature_name if str(x).strip()]
    elif isinstance(feature_name, str) and feature_name.strip():
        raw_str = feature_name.strip()
        parts = [x.strip() for x in raw_str.split(",") if x.strip()]
        targets = parts
        if len(parts) == 1 and "," not in raw_str:
            is_single_string_query = True

    # Case 1: Empty input -> return summary list of all features (without parameters_schema)
    if not targets:
        return [
            _build_feature_entry(name, data, task_features, include_schema=False)
            for name, data in chain_features.items()
        ]

    # Helper function to resolve feature target by key or chains alias
    def _resolve_target(target_name: str) -> str | None:
        if target_name in chain_features:
            return target_name
        for feat_key, feat_data in chain_features.items():
            feat_chains = feat_data.get("chains")
            if isinstance(feat_chains, str) and feat_chains == target_name:
                return feat_key
            elif isinstance(feat_chains, list) and target_name in feat_chains:
                return feat_key
        return None

    resolved_targets = []
    # Case 2: Specific feature(s) requested -> validate existence
    for target in targets:
        resolved = _resolve_target(target)
        if not resolved:
            return make_not_found_error("feature_name", target)
        resolved_targets.append(resolved)

    # Case 3: Return full info including parameters_schema
    results = [
        _build_feature_entry(target, chain_features[target], task_features, include_schema=True)
        for target in resolved_targets
    ]

    if is_single_string_query and len(results) == 1:
        return results[0]

    return results
