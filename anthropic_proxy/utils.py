"""
Utility functions for the anthropic_proxy package.
This module contains token counting, validation, debugging, and usage tracking helpers.
"""

import base64
import json
import logging
import struct
from typing import Any

import tiktoken

from .types import ClaudeUsage, global_usage_stats

logger = logging.getLogger(__name__)


# Default token estimate for images when dimensions cannot be determined
DEFAULT_IMAGE_TOKENS = 1500

# Anthropic's approximate formula: tokens = (width * height) / 750
ANTHROPIC_IMAGE_TOKEN_DIVISOR = 750


def get_image_dimensions_from_base64(
    base64_data: str, media_type: str
) -> tuple[int, int] | None:
    """
    Extract image dimensions from base64-encoded image data without external dependencies.

    Parses image headers directly to avoid loading full image into memory.
    Supports JPEG, PNG, GIF, and WebP formats.

    Args:
        base64_data: Base64-encoded image data
        media_type: MIME type (image/jpeg, image/png, image/gif, image/webp)

    Returns:
        Tuple of (width, height) or None if dimensions cannot be determined
    """
    try:
        # Decode just enough bytes to read the header
        # Most image headers are within first 32 bytes, but PNG/JPEG may need more
        header_bytes = base64.b64decode(base64_data[:1024])

        if media_type == "image/png":
            return _get_png_dimensions(header_bytes)
        elif media_type == "image/jpeg":
            return _get_jpeg_dimensions(base64_data)
        elif media_type == "image/gif":
            return _get_gif_dimensions(header_bytes)
        elif media_type == "image/webp":
            return _get_webp_dimensions(header_bytes)
        else:
            logger.debug(f"Unsupported image media type: {media_type}")
            return None
    except Exception as e:
        logger.debug(f"Failed to extract image dimensions: {e}")
        return None


def _get_png_dimensions(data: bytes) -> tuple[int, int] | None:
    """Extract dimensions from PNG header (IHDR chunk)."""
    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    # IHDR chunk starts at byte 8, contains width (4 bytes) and height (4 bytes)
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = struct.unpack(">I", data[16:20])[0]
    height = struct.unpack(">I", data[20:24])[0]
    return (width, height)


def _get_jpeg_dimensions(base64_data: str) -> tuple[int, int] | None:
    """Extract dimensions from JPEG by scanning for SOF markers."""
    try:
        # JPEG dimensions are in SOF (Start of Frame) markers
        # Need to scan through the file to find them
        data = base64.b64decode(base64_data[:65536])  # Decode more for JPEG

        if len(data) < 2 or data[0:2] != b"\xff\xd8":
            return None

        offset = 2
        while offset < len(data) - 8:
            if data[offset] != 0xFF:
                offset += 1
                continue

            marker = data[offset + 1]

            # SOF markers (Start of Frame) contain dimensions
            # SOF0 (0xC0) through SOF3 (0xC3), SOF5-SOF7, SOF9-SOF11, SOF13-SOF15
            if marker in (
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            ):
                # SOF structure: FF Cx LL LL PP HH HH WW WW
                # LL LL = length, PP = precision, HH HH = height, WW WW = width
                height = struct.unpack(">H", data[offset + 5 : offset + 7])[0]
                width = struct.unpack(">H", data[offset + 7 : offset + 9])[0]
                return (width, height)

            # Skip to next marker
            if marker == 0xD8 or marker == 0xD9:  # SOI or EOI
                offset += 2
            elif marker == 0x00:  # Stuffed byte
                offset += 1
            else:
                # Read segment length and skip
                if offset + 4 > len(data):
                    break
                segment_len = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
                offset += 2 + segment_len

        return None
    except Exception:
        return None


def _get_gif_dimensions(data: bytes) -> tuple[int, int] | None:
    """Extract dimensions from GIF header."""
    # GIF header: GIF87a or GIF89a, then 2 bytes width, 2 bytes height (little-endian)
    if len(data) < 10 or data[:3] != b"GIF":
        return None
    width = struct.unpack("<H", data[6:8])[0]
    height = struct.unpack("<H", data[8:10])[0]
    return (width, height)


def _get_webp_dimensions(data: bytes) -> tuple[int, int] | None:
    """Extract dimensions from WebP header."""
    # WebP: RIFF....WEBP, then VP8/VP8L/VP8X chunk
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None

    chunk_type = data[12:16]

    if chunk_type == b"VP8 ":
        # Lossy WebP: dimensions at bytes 26-30
        if len(data) < 30:
            return None
        # VP8 bitstream: skip to frame header
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return (width, height)
    elif chunk_type == b"VP8L":
        # Lossless WebP: signature byte + packed width/height
        if len(data) < 25:
            return None
        # Bits: 14 bits width-1, 14 bits height-1
        b1, b2, b3, b4 = data[21:25]
        width = ((b2 & 0x3F) << 8 | b1) + 1
        height = ((b4 & 0x0F) << 10 | b3 << 2 | (b2 >> 6)) + 1
        return (width, height)
    elif chunk_type == b"VP8X":
        # Extended WebP: canvas size at bytes 24-30
        if len(data) < 30:
            return None
        width = struct.unpack("<I", data[24:27] + b"\x00")[0] + 1
        height = struct.unpack("<I", data[27:30] + b"\x00")[0] + 1
        return (width, height)

    return None


def estimate_image_tokens(width: int | None = None, height: int | None = None) -> int:
    """
    Estimate token count for an image based on dimensions.

    Uses Anthropic's approximate formula: tokens = (width * height) / 750

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Estimated token count
    """
    if width is None or height is None or width <= 0 or height <= 0:
        return DEFAULT_IMAGE_TOKENS

    tokens = (width * height) // ANTHROPIC_IMAGE_TOKEN_DIVISOR
    # Ensure minimum token count
    return max(tokens, 85)


def estimate_image_tokens_from_base64(base64_data: str, media_type: str) -> int:
    """
    Estimate token count for a base64-encoded image.

    Args:
        base64_data: Base64-encoded image data
        media_type: MIME type of the image

    Returns:
        Estimated token count
    """
    dimensions = get_image_dimensions_from_base64(base64_data, media_type)
    if dimensions:
        width, height = dimensions
        tokens = estimate_image_tokens(width, height)
        logger.debug(f"Image {width}x{height} estimated at {tokens} tokens")
        return tokens

    logger.debug(
        f"Could not determine image dimensions, using default {DEFAULT_IMAGE_TOKENS} tokens"
    )
    return DEFAULT_IMAGE_TOKENS


def _get_tiktoken_encoding():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(f"Failed to get encoding, using fallback: {e}")
        return tiktoken.get_encoding("p50k_base")


def _normalize_payload(payload: Any) -> Any:
    if payload is None:
        return None
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if isinstance(payload, dict):
        return {key: _normalize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list | tuple):
        return [_normalize_payload(value) for value in payload]
    return payload


def _count_tokens_for_payload(payload: Any, encoding) -> int:
    if payload is None:
        return 0
    payload_str = json.dumps(_normalize_payload(payload))
    return len(encoding.encode(payload_str))


def count_tokens_in_response(
    response_content: str = "",
    thinking_content: str = "",
    tool_calls: list | None = None,
) -> int:
    """Count tokens in response content using tiktoken."""
    if tool_calls is None:
        tool_calls = []

    encoding = _get_tiktoken_encoding()

    content = response_content + thinking_content
    if tool_calls:
        content += json.dumps(_normalize_payload(tool_calls))

    return len(encoding.encode(content))


def _count_tokens_for_content_block(block: Any, encoding) -> int:
    """
    Count tokens for a single content block, with special handling for images.

    For image blocks, uses dimension-based estimation instead of tiktoken
    to avoid counting the base64 data as text tokens.
    """
    if not isinstance(block, dict):
        # Pydantic model - convert to dict
        if hasattr(block, "model_dump"):
            block = block.model_dump()
        else:
            return _count_tokens_for_payload(block, encoding)

    block_type = block.get("type")

    if block_type == "image":
        # Image block - estimate tokens based on dimensions, not base64 string length
        source = block.get("source", {})
        if isinstance(source, dict):
            source_type = source.get("type")
            if source_type == "base64":
                media_type = source.get("media_type", "image/jpeg")
                base64_data = source.get("data", "")
                if base64_data:
                    return estimate_image_tokens_from_base64(base64_data, media_type)
            elif source_type == "url":
                # URL-based image - can't determine dimensions, use default
                return DEFAULT_IMAGE_TOKENS

        # Fallback for unknown image format
        return DEFAULT_IMAGE_TOKENS

    # Non-image block - use normal tiktoken counting
    return _count_tokens_for_payload(block, encoding)


def _count_tokens_for_message(message: Any, encoding) -> int:
    """
    Count tokens for a single message, handling both string and structured content.

    For messages with image content blocks, uses dimension-based estimation
    for images instead of counting base64 data as text.
    """
    if not isinstance(message, dict):
        # Pydantic model - convert to dict
        if hasattr(message, "model_dump"):
            message = message.model_dump()
        else:
            return _count_tokens_for_payload(message, encoding)

    content = message.get("content")

    # Simple string content - count directly
    if isinstance(content, str):
        return _count_tokens_for_payload(message, encoding)

    # Array of content blocks - count each block appropriately
    if isinstance(content, list):
        total = 0
        # Count role and other message metadata
        message_without_content = {k: v for k, v in message.items() if k != "content"}
        total += _count_tokens_for_payload(message_without_content, encoding)

        # Count each content block
        for block in content:
            total += _count_tokens_for_content_block(block, encoding)

        return total

    # Other content types - use default counting
    return _count_tokens_for_payload(message, encoding)


def count_tokens_in_messages(messages: list, model: str) -> int:
    """
    Count tokens in messages using tiktoken, with special handling for images.

    For image content blocks, uses dimension-based estimation (Anthropic's formula:
    tokens = width * height / 750) instead of counting base64 data as text tokens.

    Args:
        messages: List of messages (Claude or dict format)
        model: Model name (currently unused but kept for API compatibility)

    Returns:
        Estimated token count
    """
    encoding = _get_tiktoken_encoding()

    total_tokens = 0
    for message in messages:
        total_tokens += _count_tokens_for_message(message, encoding)

    return total_tokens


def count_tokens_in_payload(payload: Any) -> int:
    """Count tokens in any payload using tiktoken."""
    encoding = _get_tiktoken_encoding()
    return _count_tokens_for_payload(payload, encoding)


def add_session_stats(
    model: str,
    input_tokens: int,
    output_tokens: int,
    model_id: str = "",
):
    """Add usage statistics to session tracking (cost calculation removed)."""
    try:
        # Create ClaudeUsage object for the existing global usage stats system
        usage = ClaudeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        update_global_usage_stats(
            usage=usage,
            model=model,
            context=f"session_stats_{model_id or model}",
        )
        logger.debug(
            f"Added session stats: {model}, input={input_tokens}, output={output_tokens}"
        )
    except Exception as e:
        logger.error(f"Error adding session stats: {e}")


def update_global_usage_stats(usage: ClaudeUsage, model: str, context: str = ""):
    """Update global usage statistics and log the usage information."""
    # Update the global stats
    global_usage_stats.update_usage(usage, model)

    # Log current usage
    logger.info(
        f"📊 USAGE UPDATE [{context}]: Model={model}, Input={usage.input_tokens}t, Output={usage.output_tokens}t"
    )

    # Log cache-related tokens if present
    cache_info = []
    if usage.cache_read_input_tokens and usage.cache_read_input_tokens > 0:
        cache_info.append(f"CacheRead={usage.cache_read_input_tokens}t")
    if usage.cache_creation_input_tokens and usage.cache_creation_input_tokens > 0:
        cache_info.append(f"CacheCreate={usage.cache_creation_input_tokens}t")
    if usage.prompt_cache_hit_tokens and usage.prompt_cache_hit_tokens > 0:
        cache_info.append(f"CacheHit={usage.prompt_cache_hit_tokens}t")
    if usage.prompt_cache_miss_tokens and usage.prompt_cache_miss_tokens > 0:
        cache_info.append(f"CacheMiss={usage.prompt_cache_miss_tokens}t")

    if cache_info:
        logger.info(f"💾 CACHE USAGE: {', '.join(cache_info)}")

    # Log reasoning tokens if present
    if (
        usage.completion_tokens_details
        and usage.completion_tokens_details.reasoning_tokens
    ):
        logger.info(
            f"🧠 REASONING TOKENS: {usage.completion_tokens_details.reasoning_tokens}t"
        )

    # Log session totals
    summary = global_usage_stats.get_session_summary()
    logger.info(
        f"📈 SESSION TOTALS: Requests={summary['total_requests']}, Input={summary['total_input_tokens']}t, Output={summary['total_output_tokens']}t, Total={summary['total_tokens']}t"
    )

    # Log reasoning and cache totals if significant
    if summary["total_reasoning_tokens"] > 0:
        logger.info(f"🧠 SESSION REASONING: {summary['total_reasoning_tokens']}t")

    total_cache = (
        summary["total_cache_hit_tokens"]
        + summary["total_cache_miss_tokens"]
        + summary["total_cache_read_tokens"]
    )
    if total_cache > 0:
        logger.info(
            f"💾 SESSION CACHE: Hit={summary['total_cache_hit_tokens']}t, Miss={summary['total_cache_miss_tokens']}t, Read={summary['total_cache_read_tokens']}t"
        )


def _compare_response_data(openai_response, claude_response):
    """Compare OpenAI response with converted Claude response and log differences."""
    try:
        # Extract OpenAI response data
        openai_content_blocks = 0
        openai_tool_calls = 0
        openai_finish_reason = None

        if openai_response.choices and len(openai_response.choices) > 0:
            choice = openai_response.choices[0]
            if hasattr(choice, "message") and choice.message:
                if hasattr(choice.message, "content") and choice.message.content:
                    openai_content_blocks = 1  # OpenAI has single content field
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    openai_tool_calls = len(choice.message.tool_calls)
            if hasattr(choice, "finish_reason"):
                openai_finish_reason = choice.finish_reason

        # Count Claude response data
        claude_content_blocks = (
            len(claude_response.content) if claude_response.content else 0
        )
        claude_tool_use_blocks = (
            sum(
                1
                for block in claude_response.content
                if hasattr(block, "type") and block.type == "tool_use"
            )
            if claude_response.content
            else 0
        )
        claude_stop_reason = claude_response.stop_reason

        # Log comparison
        logger.info("RESPONSE CONVERSION COMPARISON:")
        logger.info(
            f"  OpenAI -> Claude Content Blocks: {openai_content_blocks} -> {claude_content_blocks}"
        )
        logger.info(
            f"  OpenAI -> Claude Tool Calls/Use: {openai_tool_calls} -> {claude_tool_use_blocks}"
        )
        logger.info(
            f"  OpenAI -> Claude Finish/Stop Reason: {openai_finish_reason} -> {claude_stop_reason}"
        )

    except Exception as e:
        logger.warning(f"Error in response data comparison: {e}")


def _debug_openai_message_sequence(openai_messages: list, context: str):
    """Debug and validate OpenAI message sequence for tool call ordering."""
    try:
        logger.debug(
            f"🔍 DEBUG_MESSAGE_SEQUENCE [{context}]: {len(openai_messages)} messages"
        )

        for i, msg in enumerate(openai_messages):
            role = msg.get("role", "unknown")
            has_tool_calls = bool(msg.get("tool_calls"))
            has_tool_call_id = bool(msg.get("tool_call_id"))
            content_preview = (
                str(msg.get("content", ""))[:50] + "..."
                if msg.get("content")
                else "None"
            )

            logger.debug(
                f"🔍   Message {i}: role={role}, tool_calls={has_tool_calls}, tool_call_id={has_tool_call_id}, content={content_preview}"
            )

            # Validate tool call sequence
            if role == "assistant" and has_tool_calls:
                tool_calls = msg.get("tool_calls", [])
                logger.debug(f"🔍     Tool calls: {len(tool_calls)} calls")
                for j, tool_call in enumerate(tool_calls):
                    tool_name = (
                        tool_call.get("function", {}).get("name", "unknown")
                        if tool_call.get("function")
                        else "unknown"
                    )
                    logger.debug(f"🔍       Call {j}: {tool_name}")

            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "unknown")
                logger.debug(f"🔍     Tool result for: {tool_call_id}")

    except Exception as e:
        logger.warning(f"Error in debug message sequence: {e}")


def _compare_request_data(claude_request, openai_request):
    """Compare Claude request with converted OpenAI request and log differences."""
    try:
        # Count Claude request data
        claude_messages = len(claude_request.messages) if claude_request.messages else 0
        claude_tools = len(claude_request.tools) if claude_request.tools else 0
        claude_thinking = (
            claude_request.thinking.type if claude_request.thinking else "None"
        )

        # Count OpenAI request data
        openai_messages = len(openai_request.get("messages", []))
        openai_tools = len(openai_request.get("tools", []))

        # Log comparison
        logger.debug("REQUEST CONVERSION COMPARISON:")
        logger.debug(
            f"  Claude -> OpenAI Messages: {claude_messages} -> {openai_messages}"
        )
        logger.debug(f"  Claude -> OpenAI Tools: {claude_tools} -> {openai_tools}")
        logger.debug(f"  Claude Thinking: {claude_thinking}")

        # Check for message expansion (tool_result splitting)
        if openai_messages > claude_messages:
            logger.debug(
                f"  Message expansion detected: +{openai_messages - claude_messages} messages (likely tool_result splitting)"
            )

    except Exception as e:
        logger.warning(f"Error in request data comparison: {e}")


def _compare_streaming_with_non_streaming(
    original_request,
    accumulated_text: str,
    accumulated_thinking: str,
    current_content_blocks: list,
    output_tokens: int,
    openai_chunks_received: int,
):
    """Compare streaming conversion results with what non-streaming would look like."""
    try:
        logger.debug("🔍 STREAMING_VS_NON_STREAMING_COMPARISON:")
        logger.debug(f"  Streaming chunks processed: {openai_chunks_received}")
        logger.debug(f"  Accumulated text: {len(accumulated_text)} chars")
        logger.debug(f"  Accumulated thinking: {len(accumulated_thinking)} chars")
        logger.debug(f"  Content blocks: {len(current_content_blocks)}")
        logger.debug(f"  Output tokens: {output_tokens}")

        # Analyze content block types
        block_types = {}
        for block in current_content_blocks:
            block_type = block.get("type", "unknown")
            block_types[block_type] = block_types.get(block_type, 0) + 1

        logger.debug(f"  Block types: {block_types}")

    except Exception as e:
        logger.warning(f"Error in streaming comparison: {e}")


def validate_and_map_model(model: str) -> tuple[str, dict]:
    """
    Validate and map model name to provider configuration.

    Returns:
        tuple: (mapped_model_id, model_config)
    """
    # This is a placeholder function that would be implemented
    # based on the specific model mapping logic
    # For now, return the model as-is
    return model, {}


def setup_api_key_for_request(provider: str, request_params: dict) -> dict:
    """
    Setup API key for a specific provider request.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic')
        request_params: The request parameters to modify

    Returns:
        Modified request parameters with API key
    """
    # This is a placeholder function that would be implemented
    # based on the specific API key management logic
    return request_params


def process_openai_message_format(messages: list) -> list:
    """
    Process and validate OpenAI message format.

    Args:
        messages: List of OpenAI messages

    Returns:
        Processed and validated messages
    """
    # This is a placeholder function that would be implemented
    # based on the specific message processing logic
    return messages


def _extract_error_details(e: Exception) -> dict[str, Any]:
    """Extract comprehensive error details from an exception, ensuring all values are JSON serializable."""
    import traceback

    error_details = {
        "error": str(e),
        "type": type(e).__name__,
        "traceback": traceback.format_exc(),
    }

    # Special handling for JSON decode errors from empty API responses
    if type(e).__name__ == "JSONDecodeError":
        error_details["status_code"] = 500
        if hasattr(e, "doc"):
            error_details["doc"] = e.doc
        if hasattr(e, "pos"):
            error_details["pos"] = e.pos
        if hasattr(e, "lineno"):
            error_details["lineno"] = e.lineno
        if hasattr(e, "colno"):
            error_details["colno"] = e.colno
        if hasattr(e, "msg"):
            error_details["msg"] = e.msg
        # Add helpful context for empty response debugging
        if e.pos == 0 and e.msg == "Expecting value":
            error_details["likely_cause"] = (
                "API returned empty or invalid response body"
            )

    # Special handling for OpenAI API errors
    openai_error_types = [
        "APIError",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "AuthenticationError",
    ]

    if type(e).__name__ in openai_error_types:
        if hasattr(e, "status_code"):
            error_details["status_code"] = e.status_code
        if hasattr(e, "code"):
            error_details["code"] = e.code
        if hasattr(e, "param"):
            error_details["param"] = e.param

    # Combine attributes from the exception's dict and common API error attributes
    attrs_to_check = list(getattr(e, "__dict__", {}).keys())
    attrs_to_check.extend(
        ["message", "status_code", "response", "code", "param", "type"]
    )
    attrs_to_check = sorted(set(attrs_to_check))  # Get unique attributes

    for attr in attrs_to_check:
        if (
            hasattr(e, attr)
            and attr not in error_details
            and attr not in ["args", "__traceback__"]
        ):
            value = getattr(e, attr)

            if attr == "response":
                # The 'response' object from httpx/requests is not JSON serializable.
                # Extract its text content if possible.
                if hasattr(value, "text"):
                    error_details[attr] = value.text
                else:
                    error_details[attr] = str(value)
            elif isinstance(value, str | int | float | bool | list | dict | type(None)):
                # This value is already JSON serializable
                error_details[attr] = value
            else:
                # For any other non-serializable type, convert to string as a fallback.
                error_details[attr] = str(value)

    return error_details


def _format_error_message(e: Exception, error_details: dict[str, Any]) -> str:
    """Format error message for response."""
    error_message = f"Error: {str(e)}"
    if "message" in error_details and error_details["message"]:
        error_message += f"\nMessage: {error_details['message']}"
    if "response" in error_details and error_details["response"]:
        error_message += f"\nResponse: {error_details['response']}"
    return error_message


def log_openai_api_error(e: Exception, context: str = "") -> None:
    """Log detailed OpenAI API error information for debugging.

    Handles various OpenAI error types and extracts useful debugging information.

    Args:
        e: The exception to log
        context: Additional context string for the log messages
    """

    try:
        import openai
    except ImportError:
        logger.error(f"TOOL_DEBUG: {context} Error: {e}")
        return

    prefix = f"TOOL_DEBUG: {context}" if context else "TOOL_DEBUG:"

    if isinstance(e, openai.APIStatusError):
        logger.error(f"{prefix} APIStatusError - Status: {e.status_code}")
        _log_response_body(e.response, prefix)
    elif isinstance(e, openai.APIConnectionError):
        logger.error(f"{prefix} APIConnectionError - Cause: {e.__cause__}")
    elif isinstance(e, openai.APIError):
        logger.error(f"{prefix} APIError")
        if hasattr(e, "response") and e.response:
            _log_response_body(e.response, prefix)
    else:
        logger.error(f"{prefix} Non-OpenAI error: {type(e).__name__}")
        error_str = str(e)
        if "failed_generation" in error_str:
            _extract_json_from_error_string(error_str, prefix)


def _log_response_body(response, prefix: str) -> None:
    """Extract and log response body, attempting JSON parsing."""
    try:
        response_text = response.text
        logger.error(f"{prefix} Response body: {response_text}")
        try:
            response_json = json.loads(response_text)
            if "failed_generation" in response_json:
                logger.error(
                    f"{prefix} failed_generation: {response_json['failed_generation']}"
                )
            if "error" in response_json:
                logger.error(f"{prefix} error details: {response_json['error']}")
        except json.JSONDecodeError:
            pass
    except Exception as body_error:
        logger.error(f"{prefix} Could not read response body: {body_error}")


def _extract_json_from_error_string(error_str: str, prefix: str) -> None:
    """Try to extract JSON from an error message string."""
    import re

    json_match = re.search(r"\{.*\}", error_str, re.DOTALL)
    if json_match:
        try:
            error_json = json.loads(json_match.group())
            if "failed_generation" in error_json:
                logger.error(
                    f"{prefix} failed_generation: {error_json['failed_generation']}"
                )
        except json.JSONDecodeError:
            pass


def sanitize_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sanitize messages to comply with Anthropic API requirements.

    Uses single-pass validate-and-fix-on-the-fly approach:
    - Iterates through messages assuming they're valid
    - On first error, copies previous valid messages to result buffer
    - Continues with fix logic from that point
    - If no errors found, returns original list (zero-copy)

    Complexity: O(N) single pass, zero allocations for valid messages.

    Fixes the following issues:
    1. User messages incorrectly inserted between tool_use and tool_result
    2. Scattered tool_results across multiple user messages
    3. Empty content in messages (except final assistant message)
    4. Orphaned tool_use without corresponding tool_result

    Args:
        messages: List of message dicts in Anthropic format

    Returns:
        Original messages if valid, or sanitized copy if fixes were needed
    """
    if not messages:
        return messages

    # State for validation/fix
    pending_tool_use_ids: set[str] = set()

    # Fix mode state (only allocated when needed)
    result: list[dict[str, Any]] | None = None  # None = still validating
    buffered_tool_results: list[dict[str, Any]] = []
    deferred_user_messages: list[dict[str, Any]] = []

    def enter_fix_mode(error_index: int) -> None:
        """Switch to fix mode, copying all valid messages up to error_index."""
        nonlocal result
        if result is None:
            result = list(messages[:error_index])
            logger.warning(
                f"Sanitizing messages: entering fix mode at message {error_index}"
            )

    def flush_tool_results() -> None:
        if result is not None and buffered_tool_results:
            result.append({"role": "user", "content": list(buffered_tool_results)})
            buffered_tool_results.clear()

    def flush_deferred() -> None:
        if result is not None and deferred_user_messages:
            result.extend(deferred_user_messages)
            deferred_user_messages.clear()

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", [])
        is_final = i == len(messages) - 1

        # Check empty content
        if not content:
            if is_final and role == "assistant":
                if result is not None:
                    result.append(msg)
            else:
                enter_fix_mode(i)
            continue

        if role == "assistant":
            if pending_tool_use_ids:
                enter_fix_mode(i)
                for tid in pending_tool_use_ids:
                    buffered_tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tid,
                            "content": "No result provided (tool call was interrupted)",
                            "is_error": True,
                        }
                    )
                pending_tool_use_ids.clear()
                flush_tool_results()
                flush_deferred()

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_id = block.get("id")
                        if tool_id:
                            pending_tool_use_ids.add(tool_id)

            if result is not None:
                result.append(msg)

        elif role == "user":
            tool_result_ids: set[str] = set()
            tool_results: list[dict[str, Any]] = []
            other_content: list[dict[str, Any]] = []

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_results.append(block)
                        tid = block.get("tool_use_id")
                        if tid:
                            tool_result_ids.add(tid)
                    else:
                        other_content.append(block)
            elif isinstance(content, str):
                other_content.append({"type": "text", "text": content})

            if pending_tool_use_ids:
                if not tool_result_ids:
                    enter_fix_mode(i)
                    deferred_user_messages.append(msg)
                    continue

                if other_content and result is None:
                    enter_fix_mode(i)

                pending_tool_use_ids -= tool_result_ids

                if result is not None:
                    buffered_tool_results.extend(tool_results)
                    if other_content:
                        deferred_user_messages.append(
                            {"role": "user", "content": other_content}
                        )
                    if not pending_tool_use_ids:
                        flush_tool_results()
                        flush_deferred()
                else:
                    if pending_tool_use_ids:
                        enter_fix_mode(i)
                        buffered_tool_results.extend(tool_results)
                        if other_content:
                            deferred_user_messages.append(
                                {"role": "user", "content": other_content}
                            )
            else:
                if result is not None:
                    result.append(msg)

        else:
            if result is not None:
                result.append(msg)

    if pending_tool_use_ids:
        enter_fix_mode(len(messages))
        for tid in pending_tool_use_ids:
            buffered_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": "No result provided (tool call was interrupted or context was compacted)",
                    "is_error": True,
                }
            )
        flush_tool_results()
        flush_deferred()

    return messages if result is None else result


def sanitize_openai_request(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize OpenAI request payload by removing unsupported fields.

    The OpenAI SDK only accepts specific parameters. Clients sending requests
    with Anthropic-compatible fields (e.g., 'thinking') will fail if these
    are passed through to the OpenAI SDK.

    Args:
        payload: OpenAI request payload dict

    Returns:
        Sanitized payload with unsupported fields removed
    """
    # Fields not supported by OpenAI SDK
    unsupported_fields = ["thinking", "thinkingConfig"]

    for field in unsupported_fields:
        if field in payload:
            logger.debug(f"Removing unsupported field '{field}' from OpenAI request")
            del payload[field]

    return payload


def sanitize_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sanitize OpenAI messages to comply with OpenAI API requirements.

    Uses single-pass validate-and-fix-on-the-fly approach:
    - Iterates through messages assuming they're valid
    - On first error, copies previous valid messages to result buffer
    - Continues with fix logic from that point
    - If no errors found, returns original list (zero-copy)

    Complexity: O(N) single pass, zero allocations for valid messages.

    Fixes the following issues:
    1. User messages incorrectly inserted between tool_calls and tool results
    2. Scattered tool results across multiple messages
    3. Orphaned tool_calls without corresponding tool results

    OpenAI format:
    - Assistant: {role: "assistant", tool_calls: [{id: "call_xxx", ...}]}
    - Tool result: {role: "tool", tool_call_id: "call_xxx", content: "..."}

    Args:
        messages: List of message dicts in OpenAI format

    Returns:
        Original messages if valid, or sanitized copy if fixes were needed
    """
    if not messages:
        return messages

    # State for validation/fix
    pending_tool_call_ids: set[str] = set()

    # Fix mode state (only allocated when needed)
    result: list[dict[str, Any]] | None = None  # None = still validating
    buffered_tool_results: list[dict[str, Any]] = []
    deferred_messages: list[dict[str, Any]] = []

    def enter_fix_mode(error_index: int) -> None:
        """Switch to fix mode, copying all valid messages up to error_index."""
        nonlocal result
        if result is None:
            result = list(messages[:error_index])
            logger.warning(
                f"Sanitizing OpenAI messages: entering fix mode at message {error_index}"
            )

    def flush_tool_results() -> None:
        """Flush buffered tool results as a single user message."""
        if result is not None and buffered_tool_results:
            result.extend(buffered_tool_results)
            buffered_tool_results.clear()

    def flush_deferred() -> None:
        """Flush deferred non-tool messages."""
        if result is not None and deferred_messages:
            result.extend(deferred_messages)
            deferred_messages.clear()

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "assistant":
            # Check for tool_calls in assistant message
            tool_calls = msg.get("tool_calls", [])
            current_tool_ids = {tc.get("id") for tc in tool_calls if tc.get("id")}

            if pending_tool_call_ids:
                # We have pending tool calls from previous assistant - need to resolve them
                if result is None:
                    enter_fix_mode(i)

                # Add placeholders for any unresolved tool calls
                for tid in list(pending_tool_call_ids):
                    if tid not in {
                        t.get("tool_call_id") for t in buffered_tool_results
                    }:
                        buffered_tool_results.append(
                            {
                                "role": "tool",
                                "tool_call_id": tid,
                                "content": "No result provided (tool call was interrupted or context was compacted)",
                            }
                        )

                flush_tool_results()
                flush_deferred()
                pending_tool_call_ids.clear()

            if result is not None:
                result.append(msg)

            # Track new pending tool calls
            pending_tool_call_ids.update(current_tool_ids)

        elif role == "tool":
            # Tool result message
            tool_call_id = msg.get("tool_call_id")

            if not pending_tool_call_ids:
                # Orphaned tool result (no matching tool_call)
                if result is None:
                    enter_fix_mode(i)
                logger.warning(f"Orphaned tool result for {tool_call_id}, skipping")
                continue

            if tool_call_id in pending_tool_call_ids:
                if result is not None:
                    buffered_tool_results.append(msg)
                pending_tool_call_ids.discard(tool_call_id)

                if not pending_tool_call_ids:
                    flush_tool_results()
                    flush_deferred()
            else:
                # Tool result for unknown tool_call
                if result is None:
                    enter_fix_mode(i)
                logger.warning(
                    f"Tool result for unknown tool_call {tool_call_id}, skipping"
                )

        elif role == "user":
            if pending_tool_call_ids:
                # User message between tool_calls and tool results - defer it
                if result is None:
                    enter_fix_mode(i)
                deferred_messages.append(msg)
            else:
                if result is not None:
                    result.append(msg)

        else:
            # System or other messages
            if pending_tool_call_ids:
                # Non-tool message between tool_calls and tool results - defer it
                if result is None:
                    enter_fix_mode(i)
                deferred_messages.append(msg)
            else:
                if result is not None:
                    result.append(msg)

    # Handle any remaining pending tool calls
    if pending_tool_call_ids:
        enter_fix_mode(len(messages))
        # Add placeholder tool results for orphaned tool calls
        for tid in pending_tool_call_ids:
            buffered_tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": "No result provided (tool call was interrupted or context was compacted)",
                }
            )
        flush_tool_results()
        flush_deferred()

    return messages if result is None else result
