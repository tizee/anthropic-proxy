"""
Gemini/Code Assist JSON schema sanitizer.

Based on antigravity-auth's schema-sanitizer.ts.
"""

from __future__ import annotations

from typing import Any

EMPTY_SCHEMA_PLACEHOLDER_NAME = "_placeholder"
EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION = "Placeholder. Always pass true."

UNSUPPORTED_CONSTRAINTS = {
    "minLength",
    "maxLength",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "pattern",
    "minItems",
    "maxItems",
    "format",
    "default",
    "examples",
}

UNSUPPORTED_KEYWORDS = {
    *UNSUPPORTED_CONSTRAINTS,
    "$schema",
    "$defs",
    "definitions",
    "const",
    "$ref",
    "additionalProperties",
    "propertyNames",
    "title",
    "$id",
    "$comment",
}


def _append_description_hint(description: str | None, hint: str) -> str:
    if description:
        return f"{description} ({hint})"
    return hint


def _sanitize_schema(schema: Any) -> Any:
    if schema is None:
        return None

    if isinstance(schema, list):
        return [_sanitize_schema(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    schema_type = schema.get("type")
    cleaned: dict[str, Any] = {}

    base_description = schema.get("description")
    if isinstance(base_description, str) and base_description:
        cleaned["description"] = base_description

    for key, value in schema.items():
        if key == "description":
            continue

        if key in UNSUPPORTED_CONSTRAINTS:
            if value is not None and not isinstance(value, (dict, list)):
                cleaned["description"] = _append_description_hint(
                    cleaned.get("description"), f"{key}: {value}"
                )
            continue

        if key in UNSUPPORTED_KEYWORDS:
            continue

        if key == "enum" and schema_type in {"integer", "number"}:
            cleaned["enum"] = [str(item) for item in value] if value else []
            cleaned["type"] = "string"
            continue

        if isinstance(value, dict) or isinstance(value, list):
            if key == "properties" and isinstance(value, dict):
                cleaned["properties"] = {
                    prop: _sanitize_schema(prop_schema)
                    for prop, prop_schema in value.items()
                }
            else:
                cleaned[key] = _sanitize_schema(value)
        else:
            cleaned[key] = value

    if schema_type == "array" and "items" not in cleaned:
        cleaned["items"] = {}

    if "required" in cleaned and "properties" in cleaned:
        required = cleaned.get("required")
        properties = cleaned.get("properties", {})
        if isinstance(required, list) and isinstance(properties, dict):
            cleaned["required"] = [item for item in required if item in properties]

    if cleaned.get("type") == "object":
        properties = cleaned.get("properties")
        has_properties = isinstance(properties, dict) and len(properties) > 0
        if not has_properties:
            cleaned["properties"] = {
                EMPTY_SCHEMA_PLACEHOLDER_NAME: {
                    "type": "boolean",
                    "description": EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION,
                }
            }
            cleaned["required"] = [EMPTY_SCHEMA_PLACEHOLDER_NAME]

    return cleaned


def clean_gemini_schema(schema: Any) -> Any:
    """Sanitize JSON schema for Gemini/Code Assist compatibility."""
    return _sanitize_schema(schema)
