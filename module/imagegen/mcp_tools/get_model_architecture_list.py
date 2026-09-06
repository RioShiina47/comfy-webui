"""
MCP Tool: get_model_architecture_list
Get all supported model architectures, their corresponding default resolutions, and available aspect ratios.
"""

from .common import _load_yaml, _MODEL_ARCHITECTURES_PATH, _CONSTANTS_PATH


def ImageGen_get_model_architecture_list() -> list:
    """Dynamically load all supported model architectures from model_architectures.yaml along with available aspect ratios and resolutions."""
    arch_config = _load_yaml(_MODEL_ARCHITECTURES_PATH)
    constants = _load_yaml(_CONSTANTS_PATH)
    resolution_map = constants.get("RESOLUTION_MAP", {})
    architectures = arch_config.get("architectures", {})
    architecture_order = arch_config.get("architecture_order", list(architectures.keys()))

    result = []
    for arch_name in architecture_order:
        if arch_name not in architectures:
            continue
        arch_data = architectures[arch_name]
        model_type = arch_data.get("model_type", arch_name.lower())

        default_res = [1024, 1024]
        resolutions_dict = {}
        if model_type in resolution_map:
            resolutions_dict = resolution_map[model_type]
            if resolutions_dict:
                first_key = next(iter(resolutions_dict))
                default_res = resolutions_dict[first_key]

        entry = {
            "model_architecture": arch_name,
            "default_resolution": default_res,
        }
        if resolutions_dict:
            entry["available_resolutions"] = resolutions_dict

        result.append(entry)

    return result
