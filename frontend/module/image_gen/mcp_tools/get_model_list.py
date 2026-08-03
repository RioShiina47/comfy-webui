"""
MCP Tool: get_model_list
Query the list of available image generation models, with optional filtering by model architecture.
"""

from .common import _load_yaml, _MODEL_LIST_PATH, _MODEL_DEFAULTS_PATH, _MODEL_ARCHITECTURES_PATH
from .error_schema import make_not_found_error


def ImageGen_get_model_list(model_architecture: str = None) -> list | dict:
    """Dynamically load the list of available image generation models from model_list.yaml."""
    model_list = _load_yaml(_MODEL_LIST_PATH)
    model_defaults = _load_yaml(_MODEL_DEFAULTS_PATH)
    arch_config = _load_yaml(_MODEL_ARCHITECTURES_PATH)
    valid_architectures = set(arch_config.get("architectures", {}).keys())

    if model_architecture and model_architecture not in valid_architectures:
        return make_not_found_error("architecture", model_architecture)

    result = []
    checkpoints = model_list.get("Checkpoint", {}) or model_list.get("Checkpoints", {})

    for arch_name, arch_data in checkpoints.items():
        if model_architecture and arch_name != model_architecture:
            continue
        if not isinstance(arch_data, dict):
            continue

        models = arch_data.get("models", [])
        if not isinstance(models, list):
            continue

        arch_defaults = model_defaults.get(arch_name, {})
        arch_level_defaults = arch_defaults.get("_defaults", {})

        for model in models:
            display_name = model.get("display_name", "")
            category = model.get("category", None)

            model_specific_defaults = arch_defaults.get(display_name, {})

            default_pos = model_specific_defaults.get(
                "positive_prompt",
                arch_level_defaults.get("positive_prompt", ""),
            )
            default_neg = model_specific_defaults.get(
                "negative_prompt",
                arch_level_defaults.get("negative_prompt", ""),
            )

            entry = {
                "name": display_name,
                "model_architecture": arch_name,
            }
            if category:
                entry["category"] = category
            if default_pos:
                entry["default_positive_prompt"] = default_pos
            if default_neg:
                entry["default_negative_prompt"] = default_neg

            result.append(entry)

    return result
