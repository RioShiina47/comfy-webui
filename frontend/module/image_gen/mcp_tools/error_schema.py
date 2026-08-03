"""
Unified MCP tool error response format.

Error code enumeration:
  - INVALID_PARAMS:         Parameter validation failed (missing required fields, type errors, value out of range)
  - MODEL_NOT_FOUND:        The specified model name does not exist
  - ARCHITECTURE_NOT_FOUND: The specified architecture name does not exist
  - CHAIN_TYPE_NOT_FOUND:   The specified chain/injector type is invalid
  - FEATURE_NOT_SUPPORTED:  The current model does not support the requested feature
  - TASK_NOT_FOUND:         The async task ID does not exist
  - MODEL_OOM:              GPU out of memory
  - INTERNAL_ERROR:         Internal server error
"""


def make_error(code: str, message: str, details: dict = None) -> dict:
    """
    Construct a unified MCP tool error response.

    Args:
        code:    Error code (UPPER_SNAKE_CASE format)
        message: Human-readable error description
        details: Optional details dictionary

    Returns:
        Standardized error response dictionary
    """
    error = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    if details:
        error["error"]["details"] = details
    return error


def make_validation_error(
    message: str = "Request validation failed.",
    missing_fields: list = None,
    invalid_fields: dict = None,
) -> dict:
    """
    Construct a parameter validation failure error response.

    Args:
        message:        Error description
        missing_fields: List of missing required field names
        invalid_fields: Key-value pairs of invalid fields, key=field name, value=reason description

    Returns:
        Standardized INVALID_PARAMS error response
    """
    details = {}
    if missing_fields:
        details["missing_fields"] = missing_fields
    if invalid_fields:
        details["invalid_fields"] = invalid_fields
    return make_error("INVALID_PARAMS", message, details if details else None)


def make_not_found_error(resource_type: str, resource_id: str) -> dict:
    """
    Construct a resource-not-found error response.

    Args:
        resource_type: Resource type (e.g., "model", "architecture", "chain_type", "task")
        resource_id:   Resource identifier

    Returns:
        Standardized *_NOT_FOUND error response
    """
    code_map = {
        "model": "MODEL_NOT_FOUND",
        "architecture": "ARCHITECTURE_NOT_FOUND",
        "chain_type": "CHAIN_TYPE_NOT_FOUND",
        "task": "TASK_NOT_FOUND",
    }
    code = code_map.get(resource_type, f"{resource_type.upper()}_NOT_FOUND")
    return make_error(
        code,
        f"The specified {resource_type} '{resource_id}' was not found.",
        {"resource_type": resource_type, "resource_id": resource_id},
    )
