"""
Utility functions for the anthropic_proxy package.
This module contains token counting, validation, debugging, and usage tracking helpers.
"""

import json
import logging
from typing import Any

import tiktoken

from .types import ClaudeUsage, global_usage_stats

logger = logging.getLogger(__name__)


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
    if isinstance(payload, (list, tuple)):
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


def count_tokens_in_messages(messages: list, model: str) -> int:
    """Count tokens in messages using tiktoken."""
    encoding = _get_tiktoken_encoding()

    total_tokens = 0
    for message in messages:
        total_tokens += _count_tokens_for_payload(message, encoding)

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
            error_details["likely_cause"] = "API returned empty or invalid response body"

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
    import re

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
        if hasattr(e, 'response') and e.response:
            _log_response_body(e.response, prefix)
    else:
        logger.error(f"{prefix} Non-OpenAI error: {type(e).__name__}")
        error_str = str(e)
        if 'failed_generation' in error_str:
            _extract_json_from_error_string(error_str, prefix)


def _log_response_body(response, prefix: str) -> None:
    """Extract and log response body, attempting JSON parsing."""
    try:
        response_text = response.text
        logger.error(f"{prefix} Response body: {response_text}")
        try:
            response_json = json.loads(response_text)
            if 'failed_generation' in response_json:
                logger.error(f"{prefix} failed_generation: {response_json['failed_generation']}")
            if 'error' in response_json:
                logger.error(f"{prefix} error details: {response_json['error']}")
        except json.JSONDecodeError:
            pass
    except Exception as body_error:
        logger.error(f"{prefix} Could not read response body: {body_error}")


def _extract_json_from_error_string(error_str: str, prefix: str) -> None:
    """Try to extract JSON from an error message string."""
    import re

    json_match = re.search(r'\{.*\}', error_str, re.DOTALL)
    if json_match:
        try:
            error_json = json.loads(json_match.group())
            if 'failed_generation' in error_json:
                logger.error(f"{prefix} failed_generation: {error_json['failed_generation']}")
        except json.JSONDecodeError:
            pass
