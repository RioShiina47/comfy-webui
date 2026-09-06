"""
Unified MCP tool error response format for MiniMax-H3.
"""

def make_error(code: str, message: str, details: dict = None) -> dict:
    """Construct a unified MCP tool error response."""
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
    """Construct a parameter validation failure error response."""
    details = {}
    if missing_fields:
        details["missing_fields"] = missing_fields
    if invalid_fields:
        details["invalid_fields"] = invalid_fields
    return make_error("INVALID_PARAMS", message, details if details else None)


def make_not_found_error(resource_type: str, resource_id: str) -> dict:
    """Construct a resource-not-found error response."""
    code_map = {
        "task": "TASK_NOT_FOUND",
        "model": "MODEL_NOT_FOUND",
    }
    code = code_map.get(resource_type, f"{resource_type.upper()}_NOT_FOUND")
    return make_error(
        code,
        f"The specified {resource_type} '{resource_id}' was not found.",
        {"resource_type": resource_type, "resource_id": resource_id},
    )
