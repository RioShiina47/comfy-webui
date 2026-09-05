"""
MCP Tool: get_model_features
Query metadata for a specified model, including supported task types, extended features, and default inference parameters.
"""

from .common import (
    _load_yaml,
    _MODEL_LIST_PATH,
    _MODEL_DEFAULTS_PATH,
    _IMAGE_GEN_FEATURES_PATH,
    _MODEL_ARCHITECTURES_PATH,
    _CHAIN_FEATURES_PATH,
    _TASK_DEFINITIONS,
)
from .error_schema import make_validation_error, make_not_found_error


def ImageGen_get_model_features(model: str) -> dict:
    """Query metadata for a specified model: supported task types, extended features, and default inference parameters."""
    if not model:
        return make_validation_error(
            "Parameter 'model' is required.",
            missing_fields=["model"],
        )

    model_list = _load_yaml(_MODEL_LIST_PATH)
    model_defaults = _load_yaml(_MODEL_DEFAULTS_PATH)
    features_config = _load_yaml(_IMAGE_GEN_FEATURES_PATH)
    arch_config = _load_yaml(_MODEL_ARCHITECTURES_PATH)
    chain_features = _load_yaml(_CHAIN_FEATURES_PATH)

    found_arch = None
    checkpoints = model_list.get("Checkpoint", {}) or model_list.get("Checkpoints", {})
    for arch_name, arch_data in checkpoints.items():
        if not isinstance(arch_data, dict):
            continue
        for m in arch_data.get("models", []):
            if m.get("display_name") == model:
                found_arch = arch_name
                break
        if found_arch:
            break

    if not found_arch:
        return make_not_found_error("model", model)

    architectures = arch_config.get("architectures", {})
    arch_info = architectures.get(found_arch, {})
    model_type = arch_info.get("model_type", found_arch.lower())

    arch_features = features_config.get(model_type, features_config.get("default", {}))
    enabled_chains = arch_features.get("enabled_chains", [])

    supported_features = []
    for chain_name in enabled_chains:
        if chain_name in chain_features:
            chain_data = chain_features[chain_name]
            visibility = chain_data.get("visibility", "public")
            if visibility == "public":
                supported_features.append(chain_name)
            else:
                generic_mapping = {
                    "krea2_controlnet": "controlnet",
                    "anima_controlnet_lllite": "controlnet",
                    "controlnet_model_patch": "controlnet",
                    "flux1_ipadapter": "ipadapter",
                    "sd3_ipadapter": "ipadapter",
                    "hidream_o1_reference": "reference_latent",
                }
                generic_name = generic_mapping.get(chain_name)
                if generic_name and generic_name not in supported_features:
                    supported_features.append(generic_name)

    arch_defaults_section = model_defaults.get(found_arch, {})
    arch_level_defaults = arch_defaults_section.get("_defaults", {})
    model_specific_defaults = arch_defaults_section.get(model, {})
    global_defaults = model_defaults.get("Default", {})

    merged_defaults = {**global_defaults, **arch_level_defaults, **model_specific_defaults}

    default_parameter = {
        "sampler": merged_defaults.get("sampler_name", "euler"),
        "scheduler": merged_defaults.get("scheduler", "simple"),
        "steps": merged_defaults.get("steps", 20),
        "cfg": merged_defaults.get("cfg", 1.0),
    }

    supported_tasks = [t["task_type"] for t in _TASK_DEFINITIONS]

    result = {
        "name": model,
        "model_architecture": found_arch,
        "supported_tasks": supported_tasks,
        "supported_features": supported_features,
        "default_parameter": default_parameter,
    }

    default_pos = model_specific_defaults.get(
        "positive_prompt",
        arch_level_defaults.get("positive_prompt", ""),
    )
    default_neg = model_specific_defaults.get(
        "negative_prompt",
        arch_level_defaults.get("negative_prompt", ""),
    )
    if default_pos:
        result["default_positive_prompt"] = default_pos
    if default_neg:
        result["default_negative_prompt"] = default_neg

    return result
