"""
Format conversion facade for Anthropic proxy.

OpenAI-specific conversion lives in openai_converter.py. This module keeps
Gemini schema helpers and re-exports OpenAI converters for compatibility.
"""

from typing import Any

from .openai_converter import (
    convert_openai_response_to_anthropic,
    extract_usage_from_openai_response,
    parse_function_calls_from_thinking,
)
from .types import ClaudeUsage


def extract_usage_from_claude_response(
    claude_usage_dict: dict[str, Any],
) -> ClaudeUsage:
    """Extract usage data from Claude API response format and convert to enhanced ClaudeUsage."""
    from .types import CacheCreation, ServerToolUse

    if not claude_usage_dict:
        return ClaudeUsage(input_tokens=0, output_tokens=0)

    # Core fields
    input_tokens = claude_usage_dict.get("input_tokens", 0)
    output_tokens = claude_usage_dict.get("output_tokens", 0)
    cache_creation_input_tokens = claude_usage_dict.get(
        "cache_creation_input_tokens", 0
    )
    cache_read_input_tokens = claude_usage_dict.get("cache_read_input_tokens", 0)

    # Handle cache_creation object
    cache_creation = None
    if "cache_creation" in claude_usage_dict and claude_usage_dict["cache_creation"]:
        cache_data = claude_usage_dict["cache_creation"]
        cache_creation = CacheCreation(
            ephemeral_1h_input_tokens=cache_data.get("ephemeral_1h_input_tokens", 0),
            ephemeral_5m_input_tokens=cache_data.get("ephemeral_5m_input_tokens", 0),
        )

    # Handle server_tool_use object
    server_tool_use = None
    if "server_tool_use" in claude_usage_dict and claude_usage_dict["server_tool_use"]:
        tool_data = claude_usage_dict["server_tool_use"]
        server_tool_use = ServerToolUse(
            web_search_requests=tool_data.get("web_search_requests", 0)
        )

    service_tier = claude_usage_dict.get("service_tier")

    return ClaudeUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_creation=cache_creation,
        server_tool_use=server_tool_use,
        service_tier=service_tier,
    )


def clean_gemini_schema(schema: Any) -> Any:
    """
    Recursively clean and validate JSON schema for Gemini OpenAI API compatibility.
    """
    if schema is None:
        return None

    if isinstance(schema, dict):
        schema_type = schema.get("type")
        cleaned_schema = {}
        for key, value in schema.items():
            # Skip unsupported keywords
            if key in [
                "$schema",
                "$id",
                "$ref",
                "$defs",
                "default",  # Can cause validation issues
                "examples",  # Not supported
                "definitions",  # Complex references not supported
                "title",  # Can cause issues in nested objects
            ]:
                continue
            if key == "enum" and schema_type in {"integer", "number"}:
                cleaned_schema["enum"] = [str(item) for item in value] if value else []
                cleaned_schema["type"] = "string"
                continue

            # Clean nested schemas
            if isinstance(value, dict):
                cleaned_schema[key] = clean_gemini_schema(value)
            elif isinstance(value, list):
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_list.append(clean_gemini_schema(item))
                    else:
                        cleaned_list.append(item)
                cleaned_schema[key] = cleaned_list
            else:
                cleaned_schema[key] = value

        if schema_type == "array" and "items" not in cleaned_schema:
            cleaned_schema["items"] = {}

        if "required" in cleaned_schema and "properties" in cleaned_schema:
            required = cleaned_schema.get("required")
            properties = cleaned_schema.get("properties", {})
            if isinstance(required, list) and isinstance(properties, dict):
                cleaned_schema["required"] = [item for item in required if item in properties]

        return cleaned_schema

    if isinstance(schema, list):
        cleaned_list = []
        for item in schema:
            if isinstance(item, dict):
                cleaned_list.append(clean_gemini_schema(item))
            else:
                cleaned_list.append(item)
        return cleaned_list

    return schema


def validate_gemini_function_schema(tool_def: dict) -> tuple[bool, str]:
    """
    Validate Gemini function schema for compatibility.

    Returns:
        (is_valid, error_message)
    """
    try:
        function = tool_def.get("function", {})
        params = function.get("parameters", {})

        if not isinstance(params, dict):
            return False, "Parameters must be a JSON schema object"

        # Check for required fields
        if "type" not in params or params.get("type") != "object":
            return False, "Parameters must have type: object"

        # Check for unsupported keywords
        unsupported_keywords = [
            "$schema",
            "$id",
            "$ref",
            "$defs",
            "default",
            "examples",
            "definitions",
        ]

        def check_nested_object(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in unsupported_keywords:
                        return False, f"Unsupported keyword '{key}' found at {path}"
                    if not check_nested_object(value, f"{path}.{key}" if path else key):
                        return False, f"Unsupported keyword found in nested object"
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if not check_nested_object(item, f"{path}[{i}]"):
                        return False, f"Unsupported keyword found in list"
            return True, ""

        valid, message = check_nested_object(params)
        if not valid:
            return False, message

        return True, ""

    except Exception as e:
        return False, str(e)


__all__ = [
    "convert_openai_response_to_anthropic",
    "extract_usage_from_openai_response",
    "parse_function_calls_from_thinking",
    "extract_usage_from_claude_response",
    "clean_gemini_schema",
    "validate_gemini_function_schema",
]
