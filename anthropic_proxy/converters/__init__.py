"""
Converters package for format conversion.

Provides converters between different API formats using Anthropic Messages
as the internal pivot format.

Supported formats:
- Anthropic Messages (pivot format)
- OpenAI Chat Completions
- Gemini GenerateContent
"""

from .base import BaseConverter, BaseStreamingConverter
from .anthropic import AnthropicConverter
from .anthropic import AnthropicStreamingConverter as AnthropicIdentityStreamingConverter
from .openai import OpenAIConverter, OpenAIToAnthropicStreamingConverter
from .gemini import GeminiConverter, GeminiStreamingConverter

# Re-export implementation functions for backward compatibility
from ._openai_impl import (
    convert_openai_response_to_anthropic,
    convert_openai_streaming_response_to_anthropic,
    AnthropicStreamingConverter,  # Legacy class, takes original_request arg
    extract_usage_from_openai_response,
    parse_function_calls_from_thinking,
)
from ._gemini_impl import (
    anthropic_to_gemini_request,
    anthropic_to_gemini_sdk_params,
    ensure_tool_ids,
    _clean_malformed_parts,
)
from ._gemini_streaming import (
    convert_gemini_streaming_response_to_anthropic,
    GeminiStreamingConverter as GeminiStreamingConverterImpl,
)
from ..gemini_schema_sanitizer import clean_gemini_schema

# Format identifiers
FORMAT_ANTHROPIC = "anthropic"
FORMAT_OPENAI = "openai"
FORMAT_GEMINI = "gemini"


def get_converter(format_type: str, **kwargs) -> BaseConverter:
    """
    Factory function to get the appropriate converter for a format.

    Args:
        format_type: One of "anthropic", "openai", "gemini"
        **kwargs: Additional arguments passed to converter constructor

    Returns:
        Appropriate converter instance
    """
    if format_type == FORMAT_ANTHROPIC:
        return AnthropicConverter()
    elif format_type == FORMAT_OPENAI:
        return OpenAIConverter()
    elif format_type == FORMAT_GEMINI:
        return GeminiConverter(**kwargs)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def get_streaming_converter(format_type: str, **kwargs) -> BaseStreamingConverter:
    """
    Factory function to get the appropriate streaming converter for a format.

    Args:
        format_type: One of "anthropic", "openai", "gemini"
        **kwargs: Additional arguments passed to converter constructor

    Returns:
        Appropriate streaming converter instance
    """
    if format_type == FORMAT_ANTHROPIC:
        return AnthropicIdentityStreamingConverter()
    elif format_type == FORMAT_OPENAI:
        return OpenAIToAnthropicStreamingConverter()
    elif format_type == FORMAT_GEMINI:
        return GeminiStreamingConverter(**kwargs)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


__all__ = [
    # Base classes
    "BaseConverter",
    "BaseStreamingConverter",
    # Converters
    "AnthropicConverter",
    "AnthropicIdentityStreamingConverter",
    "OpenAIConverter",
    "OpenAIToAnthropicStreamingConverter",
    "GeminiConverter",
    "GeminiStreamingConverter",
    # Factory functions
    "get_converter",
    "get_streaming_converter",
    # Format constants
    "FORMAT_ANTHROPIC",
    "FORMAT_OPENAI",
    "FORMAT_GEMINI",
    # OpenAI implementation exports (legacy)
    "convert_openai_response_to_anthropic",
    "convert_openai_streaming_response_to_anthropic",
    "AnthropicStreamingConverter",  # Legacy OpenAI->Anthropic streaming converter
    "extract_usage_from_openai_response",
    "parse_function_calls_from_thinking",
    # Gemini implementation exports
    "anthropic_to_gemini_request",
    "anthropic_to_gemini_sdk_params",
    "ensure_tool_ids",
    "convert_gemini_streaming_response_to_anthropic",
    "clean_gemini_schema",
]
