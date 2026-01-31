"""
Format conversion facade for Anthropic proxy.

OpenAI-specific conversion lives in openai_converter.py. This module keeps
Gemini schema helpers and re-exports OpenAI converters for compatibility.
"""

from typing import Any

from .converters import (
    convert_openai_response_to_anthropic,
    extract_usage_from_openai_response,
    parse_function_calls_from_thinking,
)
from .gemini_schema_sanitizer import clean_gemini_schema
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
            "const",
            "additionalProperties",
            "propertyNames",
            "title",
            "$comment",
            "minLength",
            "maxLength",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "pattern",
            "minItems",
            "maxItems",
            "format",
        ]

        def check_nested_object(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in unsupported_keywords:
                        return False, f"Unsupported keyword '{key}' found at {path}"
                    if not check_nested_object(value, f"{path}.{key}" if path else key):
                        return False, "Unsupported keyword found in nested object"
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if not check_nested_object(item, f"{path}[{i}]"):
                        return False, "Unsupported keyword found in list"
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
