"""
MCP Tool: get_feature_list
Get the list of supported advanced features along with their usage constraints and parameter schemas.
"""

from .common import _load_yaml, _CHAIN_FEATURES_PATH


def ImageGen_get_feature_list() -> list:
    """Dynamically load the list of supported advanced features from chain_features.yaml."""
    chain_features = _load_yaml(_CHAIN_FEATURES_PATH)
    result = []

    for chain_name, chain_data in chain_features.items():
        if chain_data.get("visibility", "public") != "public":
            continue

        entry = {
            "feature_name": chain_name,
            "display_name": chain_data.get("display_name", chain_name),
            "description": chain_data.get("description", ""),
            "supported_tasks": chain_data.get("supported_tasks", []),
            "max_count": chain_data.get("max_count", 1),
            "usage_guideline": chain_data.get("usage_guideline", ""),
            "parameters_schema": chain_data.get("parameters_schema", {}),
        }
        result.append(entry)

    return result
