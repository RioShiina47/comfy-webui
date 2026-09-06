"""
Unified MCP tool error response format for Lightricks LTX-2.5.
"""

from typing import Optional, Dict, Any


def make_error(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct a unified MCP tool error response."""
    error_payload: Dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if details:
        error_payload["details"] = details
    return {"error": error_payload}


def make_validation_error(
    message: str,
    missing_fields: Optional[list] = None,
    invalid_fields: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Construct an MCP parameter validation error response."""
    details: Dict[str, Any] = {}
    if missing_fields:
        details["missing_fields"] = missing_fields
    if invalid_fields:
        details["invalid_fields"] = invalid_fields
    return make_error("VALIDATION_ERROR", message, details if details else None)


def make_not_found_error(resource_type: str, resource_id: str) -> Dict[str, Any]:
    """Construct an MCP resource not found error response."""
    return make_error(
        "NOT_FOUND",
        f"{resource_type.capitalize()} '{resource_id}' was not found.",
        {"resource_type": resource_type, "resource_id": resource_id},
    )
