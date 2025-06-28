"""
Format conversion functions between Claude and OpenAI APIs.
This module handles all the conversion logic between different API formats.
"""

import json
import logging
import re
import uuid
from typing import Any

from openai.types.chat import ChatCompletion

from .types import (
    ClaudeContentBlockText,
    ClaudeContentBlockThinking,
    ClaudeContentBlockToolUse,
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeUsage,
    Constants,
    generate_unique_id,
)

logger = logging.getLogger(__name__)


def parse_function_calls_from_thinking(thinking_content: str) -> tuple[str, list]:
    """Parse function calls from thinking content with custom markers.

    Returns:
        tuple: (cleaned_thinking_content, list_of_tool_calls)
    """
    # Pattern to match function call blocks (handles whitespace/newlines)
    pattern = r"<\|FunctionCallBegin\|>\s*\[(.*?)\]\s*<\|FunctionCallEnd\|>"

    tool_calls = []
    cleaned_content = thinking_content

    matches = re.findall(pattern, thinking_content, re.DOTALL)
    
    logger.debug(f"Found {len(matches)} function call matches in thinking content")

    for match in matches:
        try:
            # Parse the JSON array of function calls (brackets already captured)
            match_content = match.strip()
            logger.debug(f"Attempting to parse function call JSON: {match_content[:100]}...")
            function_call_data = json.loads(f"[{match_content}]")

            for call_data in function_call_data:
                if (
                    isinstance(call_data, dict)
                    and "name" in call_data
                    and "parameters" in call_data
                ):
                    # Create a tool call in OpenAI format
                    tool_call = {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": call_data["name"],
                            "arguments": json.dumps(call_data["parameters"]),
                        },
                    }
                    logger.debug(f"Added tool call: {tool_call}")
                    tool_calls.append(tool_call)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse function call from thinking content: {e}")
            logger.debug(f"Problematic content: {match[:200]}...")
            continue

    # Remove the function call markers from thinking content
    cleaned_content = re.sub(pattern, "", thinking_content, flags=re.DOTALL).strip()

    return cleaned_content, tool_calls


def _parse_tool_arguments(arguments_str: str) -> dict:
    """Parse tool arguments from string to dict."""
    try:
        return json.loads(arguments_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to parse tool arguments: {arguments_str}")
        return {}


def extract_usage_from_openai_response(openai_response) -> ClaudeUsage:
    """Extract usage data from OpenAI API response and convert to ClaudeUsage format."""
    from .types import CompletionTokensDetails, PromptTokensDetails

    usage = (
        openai_response.usage
        if hasattr(openai_response, "usage") and openai_response.usage
        else None
    )

    if not usage:
        return ClaudeUsage(input_tokens=0, output_tokens=0)

    # Core fields mapping
    input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)

    # Extract completion_tokens_details if present
    completion_details = None
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        details = usage.completion_tokens_details
        completion_details = CompletionTokensDetails(
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
            accepted_prediction_tokens=getattr(
                details, "accepted_prediction_tokens", None
            ),
            rejected_prediction_tokens=getattr(
                details, "rejected_prediction_tokens", None
            ),
        )

    # Extract prompt_tokens_details if present
    prompt_details = None
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        details = usage.prompt_tokens_details
        prompt_details = PromptTokensDetails(
            cached_tokens=getattr(details, "cached_tokens", None)
        )

    # Handle Deepseek-specific fields
    prompt_cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", None)
    prompt_cache_miss_tokens = getattr(usage, "prompt_cache_miss_tokens", None)

    return ClaudeUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
        completion_tokens_details=completion_details,
        prompt_tokens_details=prompt_details,
    )


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


def convert_openai_response_to_anthropic(
    openai_response: ChatCompletion, original_request: ClaudeMessagesRequest
) -> ClaudeMessagesResponse:
    """Convert OpenAI response back to Anthropic API format using OpenAI SDK type validation."""
    try:
        # Validate and extract response data using OpenAI SDK types
        response_id = f"msg_{uuid.uuid4()}"
        content_text = ""
        tool_calls = None
        finish_reason = "stop"
        thinking_content = ""

        logger.debug(f"Converting OpenAI response: {type(openai_response)}")

        choice = openai_response.choices[0]
        # Extract message content
        message = choice.message
        content_text = message.content or ""
        raw_message = message.model_dump()

        # Extract reasoning_content for thinking models (OpenAI format)
        if "reasoning_content" in raw_message:
            thinking_content = raw_message["reasoning_content"]
            logger.debug(
                f"Extracted reasoning_content: {len(thinking_content)} characters"
            )

        # Extract tool calls if present
        if message.tool_calls:
            tool_calls = message.tool_calls
        # Extract finish reason
        finish_reason = choice.finish_reason

        # Extract usage information
        usage = openai_response.usage
        if usage:
            logger.debug(f"token usage from response: {usage}")
        logger.debug(f"Raw content extracted: {len(content_text)} characters")
        logger.debug(f"Tool calls from response: {tool_calls}")

        # Enhanced debugging for Claude Code tool testing
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug("=== ENHANCED DEBUG INFO ===")
            if thinking_content:
                logger.debug(f"Thinking content preview: {thinking_content[:500]}...")
            if content_text:
                logger.debug(f"Raw content text: {repr(content_text)}")
            logger.debug("=== END DEBUG INFO ===")

        # Build content blocks
        content_blocks = []

        # Add thinking content first if present (for Claude Code display)
        if thinking_content:
            thinking_signature = generate_unique_id("thinking")
            content_blocks.append(
                ClaudeContentBlockThinking(
                    type="thinking",
                    thinking=thinking_content,
                    signature=thinking_signature,
                )
            )
            logger.debug(
                f"Added thinking content block: {len(thinking_content)} characters, signature: {thinking_signature}"
            )

        # Add text content if present
        if content_text:
            content_blocks.append(
                ClaudeContentBlockText(type=Constants.CONTENT_TEXT, text=content_text)
            )

        # Process tool calls
        # OpenAI tool calls -> Claude Tool Use
        if tool_calls:
            for tool_call in tool_calls:
                try:
                    arguments_dict = _parse_tool_arguments(tool_call.function.arguments)
                    content_blocks.append(
                        ClaudeContentBlockToolUse(
                            type=Constants.CONTENT_TOOL_USE,
                            id=generate_unique_id("toolu"),
                            name=tool_call.function.name,
                            input=arguments_dict,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error processing tool call: {e}")
                    continue

        # Map finish reason to Anthropic format
        if finish_reason == "length":
            stop_reason = Constants.STOP_MAX_TOKENS
        elif finish_reason == "tool_calls" or finish_reason is None and tool_calls:
            stop_reason = Constants.STOP_TOOL_USE
        else:
            stop_reason = Constants.STOP_END_TURN

        # Final debug info for Claude message creation
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug(
                f"Creating Claude response with {len(content_blocks)} content blocks:"
            )
            for i, block in enumerate(content_blocks):
                block_type = getattr(block, "type", "unknown")
                logger.debug(f"  Block {i}: {block_type}")
                if block_type == "thinking":
                    thinking_text = getattr(block, "thinking", "")
                    logger.debug(f"    Thinking length: {len(thinking_text)}")
                elif block_type == "text":
                    text = getattr(block, "text", "")
                    logger.debug(f"    Text: {repr(text[:200])}...")
                elif block_type == "tool_use":
                    name = getattr(block, "name", "unknown")
                    logger.debug(f"    Tool: {name}")

        # Extract comprehensive usage data from OpenAI response
        enhanced_usage = extract_usage_from_openai_response(openai_response)

        # Create Claude response
        claude_response = ClaudeMessagesResponse(
            id=response_id,
            model=original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=content_blocks,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=enhanced_usage,
        )

        # Compare response data and log any mismatches
        from .utils import _compare_response_data

        _compare_response_data(openai_response, claude_response)

        return claude_response

    except Exception as e:
        logger.error(f"Error converting response: {e}")
        return ClaudeMessagesResponse(
            id=f"msg_error_{uuid.uuid4()}",
            model=original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=[
                ClaudeContentBlockText(
                    type=Constants.CONTENT_TEXT, text="Response conversion error"
                )
            ],
            stop_reason=Constants.STOP_ERROR,
            usage=ClaudeUsage(input_tokens=0, output_tokens=0),
        )


def clean_gemini_schema(schema: Any) -> Any:
    """
    Recursively clean and validate JSON schema for Gemini OpenAI API compatibility.

    Gemini requires a strict subset of OpenAPI schema format:
    - Function names: descriptive, no spaces/special chars (underscores/camelCase OK)
    - Parameters: must be object type with properties structure
    - Required: array of mandatory parameter names
    - Enum: use arrays for fixed value sets
    """
    if isinstance(schema, dict):
        # Create a copy to avoid modifying the original
        cleaned_schema = {}

        for key, value in schema.items():
            # Skip problematic fields that cause MALFORMED_FUNCTION_CALL
            if key in [
                "additionalProperties",  # Gemini doesn't support this
                "default",  # Can cause validation issues
                "examples",  # Not part of core OpenAPI subset
                "title",  # Not needed for function parameters
                "$schema",  # Schema metadata not needed
                "$id",  # Schema metadata not needed
                "definitions",  # Complex references not supported
                "$ref",  # References not supported
                "allOf",
                "anyOf",
                "oneOf",  # Complex schema combinations
                "not",  # Negation schemas not supported
                "patternProperties",  # Advanced property patterns
                "dependencies",  # Complex dependency validation
                "const",  # Use enum instead
                "contains",  # Array contains validation
                "propertyNames",  # Property name validation
                "if",
                "then",
                "else",  # Conditional schemas
                "readOnly",
                "writeOnly",  # OpenAPI 3.0 specific fields
                "deprecated",  # OpenAPI metadata
                "external_docs",  # OpenAPI documentation
                "xml",  # XML-specific metadata
                "example",  # Use examples array instead
            ]:
                logger.debug(
                    f"Removing unsupported field '{key}' for Gemini compatibility"
                )
                continue

            # Handle format field - only allow very basic formats
            if key == "format":
                # Gemini only supports very limited formats
                allowed_formats = {"date-time", "email", "uri"}  # Minimal set
                if value not in allowed_formats:
                    logger.debug(
                        f"Removing unsupported format '{value}' for Gemini compatibility"
                    )
                    continue

            # Recursively clean nested structures
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

        return cleaned_schema

    elif isinstance(schema, list):
        # Handle arrays
        cleaned_list = []
        for item in schema:
            if isinstance(item, dict):
                cleaned_list.append(clean_gemini_schema(item))
            else:
                cleaned_list.append(item)
        return cleaned_list

    else:
        # For non-dict, non-list values, return as-is
        return schema


def validate_gemini_function_schema(tool_def: dict) -> tuple[bool, str]:
    """
    Validate function schema for Gemini compatibility.

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        function = tool_def.get("function", {})
        name = function.get("name", "")
        parameters = function.get("parameters", {})

        # Check function name
        if not name or not isinstance(name, str):
            return False, "Function name is required and must be a string"

        # Check for invalid characters in function name
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            return False, f"Function name '{name}' contains invalid characters"

        # Check parameters structure
        if not isinstance(parameters, dict):
            return False, "Parameters must be an object"

        # Check required fields in parameters
        if "type" not in parameters:
            return False, "Parameters must have a 'type' field"

        if parameters.get("type") != "object":
            return False, "Parameters type must be 'object'"

        # Validate properties if present
        properties = parameters.get("properties", {})
        if properties and not isinstance(properties, dict):
            return False, "Properties must be an object"

        # Check for problematic fields
        problematic_fields = [
            "additionalProperties",
            "default",
            "examples",
            "title",
            "$schema",
            "$id",
            "definitions",
            "$ref",
        ]

        def check_nested_object(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in problematic_fields:
                        return False, f"Unsupported field '{key}' found at {path}.{key}"
                    if isinstance(value, dict | list):
                        is_valid, error = check_nested_object(value, f"{path}.{key}")
                        if not is_valid:
                            return is_valid, error
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, dict | list):
                        is_valid, error = check_nested_object(item, f"{path}[{i}]")
                        if not is_valid:
                            return is_valid, error
            return True, ""

        is_valid, error = check_nested_object(parameters, "parameters")
        if not is_valid:
            return False, error

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"
