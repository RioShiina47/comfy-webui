"""
MCP Tool: get_chain_schema
Get the complete parameter schema and usage guide for a specified chain/injector type.
"""

from .common import _load_yaml, _CHAIN_FEATURES_PATH
from .error_schema import make_validation_error, make_not_found_error


def ImageGen_get_chain_schema(chain_type: str) -> dict:
    """Get the complete parameter schema and usage guide for a specified chain/injector type."""
    if not chain_type:
        return make_validation_error(
            "Parameter 'chain_type' is required.",
            missing_fields=["chain_type"],
        )

    chain_features = _load_yaml(_CHAIN_FEATURES_PATH)

    if chain_type not in chain_features:
        return make_not_found_error("chain_type", chain_type)

    chain_data = chain_features[chain_type]
    return {
        "feature_name": chain_type,
        "display_name": chain_data.get("display_name", chain_type),
        "description": chain_data.get("description", ""),
        "supported_tasks": chain_data.get("supported_tasks", []),
        "max_count": chain_data.get("max_count", 1),
        "usage_guideline": chain_data.get("usage_guideline", ""),
        "parameters_schema": chain_data.get("parameters_schema", {}),
    }
