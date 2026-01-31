"""
Anthropic ↔ OpenAI conversion helpers (including streaming).

This module isolates OpenAI-format conversion and OpenAI SSE streaming handling
from Gemini-style conversion to avoid format confusion.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

import tiktoken
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ..types import (
    ClaudeContentBlockText,
    ClaudeContentBlockThinking,
    ClaudeContentBlockToolUse,
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeUsage,
    generate_unique_id,
)
from ..utils import log_openai_api_error

logger = logging.getLogger(__name__)

# Tokenizer for fallback token counting
_tokenizer = tiktoken.get_encoding("cl100k_base")


def parse_function_calls_from_thinking(thinking_content: str) -> tuple[str, list]:
    """Parse function calls from thinking content with custom markers.

    Returns:
        tuple: (cleaned_thinking_content, list_of_tool_calls)
    """
    # Pattern to match function call blocks (handles whitespace/newlines)
    pattern = r"<\|FunctionCallBegin\|>\s*(.*?)\s*<\|FunctionCallEnd\|>"

    tool_calls = []
    cleaned_content = thinking_content

    matches = re.findall(pattern, thinking_content, re.DOTALL)

    logger.debug(f"Found {len(matches)} function call matches in thinking content")

    for match in matches:
        match_content = match.strip()
        logger.debug(f"Raw function call content: {match_content[:200]}...")

        # Try multiple parsing approaches
        parsed_calls = _parse_function_call_content(match_content)

        for call_data in parsed_calls:
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
                logger.debug(f"Successfully created tool call: {tool_call['function']['name']}")
                tool_calls.append(tool_call)

    # Remove the function call markers from thinking content
    cleaned_content = re.sub(pattern, "", thinking_content, flags=re.DOTALL).strip()

    return cleaned_content, tool_calls


def _parse_function_call_content(content: str) -> list:
    """Parse function call content with multiple fallback approaches."""
    parsed_calls = []

    # Approach 1: Try direct JSON parsing (content should be a JSON array)
    try:
        logger.debug(f"Attempting direct JSON parse: {content[:100]}...")
        if content.strip().startswith("[") and content.strip().endswith("]"):
            function_call_data = json.loads(content)
            if isinstance(function_call_data, list):
                parsed_calls.extend(function_call_data)
                logger.debug(f"Direct parse successful, found {len(parsed_calls)} calls")
                return parsed_calls
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")

    # Approach 2: Try wrapping in brackets if not already wrapped
    try:
        logger.debug("Attempting parse with bracket wrapping...")
        if not content.strip().startswith("["):
            wrapped_content = f"[{content}]"
            function_call_data = json.loads(wrapped_content)
            if isinstance(function_call_data, list):
                parsed_calls.extend(function_call_data)
                logger.debug(f"Bracket-wrapped parse successful, found {len(parsed_calls)} calls")
                return parsed_calls
    except json.JSONDecodeError as e:
        logger.debug(f"Bracket-wrapped parse failed: {e}")

    # Approach 3: Try to fix common JSON formatting issues
    try:
        logger.debug("Attempting parse with JSON repair...")
        repaired_content = _repair_json_formatting(content)
        function_call_data = json.loads(repaired_content)
        if isinstance(function_call_data, list):
            parsed_calls.extend(function_call_data)
            logger.debug(f"Repaired JSON parse successful, found {len(parsed_calls)} calls")
            return parsed_calls
        elif isinstance(function_call_data, dict):
            parsed_calls.append(function_call_data)
            logger.debug("Repaired JSON parse successful, found 1 call (dict)")
            return parsed_calls
    except json.JSONDecodeError as e:
        logger.debug(f"Repaired JSON parse failed: {e}")

    # Approach 4: Extract individual function calls using regex
    try:
        logger.debug("Attempting regex-based extraction...")
        regex_calls = _extract_calls_with_regex(content)
        if regex_calls:
            parsed_calls.extend(regex_calls)
            logger.debug(f"Regex extraction successful, found {len(parsed_calls)} calls")
            return parsed_calls
    except Exception as e:
        logger.debug(f"Regex extraction failed: {e}")

    logger.warning(f"All parsing approaches failed for content: {content[:200]}...")
    return parsed_calls


def _repair_json_formatting(content: str) -> str:
    """Attempt to repair common JSON formatting issues."""
    repaired = content.strip()

    # Remove trailing commas before closing brackets/braces
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

    # Ensure proper array wrapping if content starts with object but not array
    if repaired.startswith("{") and not repaired.startswith("["):
        repaired = f"[{repaired}]"

    return repaired


def _extract_calls_with_regex(content: str) -> list:
    """Extract function calls using regex patterns as a last resort."""
    calls = []

    # Pattern to match individual function call objects
    call_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"parameters"\s*:\s*(\{.*?\})\s*\}'

    matches = re.findall(call_pattern, content, re.DOTALL)

    for name, params_str in matches:
        try:
            parameters = json.loads(params_str)
            call_data = {
                "name": name,
                "parameters": parameters
            }
            calls.append(call_data)
            logger.debug(f"Regex extracted call: {name}")
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse parameters for {name}: {e}")
            continue

    return calls


def _parse_tool_arguments(arguments_str: str) -> dict:
    """Parse tool arguments from string to dict."""
    logger.debug(f"🔧 TOOL_DEBUG: Parsing tool arguments: '{arguments_str}'")
    logger.debug(f"🔧 TOOL_DEBUG: Arguments type: {type(arguments_str)}")
    try:
        result = json.loads(arguments_str)
        logger.debug(f"🔧 TOOL_DEBUG: Successfully parsed arguments: {result}")
        return result
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"🔧 TOOL_DEBUG: Failed to parse tool arguments: {arguments_str}, error: {e}")
        return {}


def extract_usage_from_openai_response(openai_response) -> ClaudeUsage:
    """Extract usage data from OpenAI API response and convert to ClaudeUsage format."""
    from ..types import CompletionTokensDetails, PromptTokensDetails

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
            logger.debug(f"🔧 TOOL_DEBUG: message.tool_calls found: {tool_calls}")
            logger.debug(f"🔧 TOOL_DEBUG: tool_calls type: {type(tool_calls)}")
            for i, tc in enumerate(tool_calls):
                logger.debug(f"🔧 TOOL_DEBUG: tool_call {i}: {tc}")
                logger.debug(f"🔧 TOOL_DEBUG: tool_call {i} type: {type(tc)}")
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

        # Add main content text if present
        if content_text:
            content_blocks.append(
                ClaudeContentBlockText(type="text", text=content_text)
            )

        # If no content and no tool calls, add empty text block
        if not content_blocks and not tool_calls:
            content_blocks.append(ClaudeContentBlockText(type="text", text=""))

        # If tool calls present, add tool use blocks
        if tool_calls:
            logger.debug(f"🔧 TOOL_DEBUG: Processing {len(tool_calls)} tool calls")
            seen_tool_ids: set[str] = set()
            for i, tool_call in enumerate(tool_calls):
                try:
                    logger.debug(f"🔧 TOOL_DEBUG: Processing tool_call {i}: {tool_call}")
                    logger.debug(f"🔧 TOOL_DEBUG: tool_call.function: {tool_call.function}")
                    logger.debug(f"🔧 TOOL_DEBUG: tool_call.function.arguments: {tool_call.function.arguments}")

                    arguments_dict = _parse_tool_arguments(tool_call.function.arguments)

                    tool_id = getattr(tool_call, "id", None)
                    if (
                        not tool_id
                        or tool_id in seen_tool_ids
                        or (
                            str(tool_id).startswith("call_")
                            or str(tool_id).startswith("tool_")
                        )
                    ):
                        tool_id = generate_unique_id("toolu")
                    seen_tool_ids.add(tool_id)

                    # Create tool use block
                    tool_use_block = ClaudeContentBlockToolUse(
                        type="tool_use",
                        id=tool_id,
                        name=tool_call.function.name,
                        input=arguments_dict,
                    )

                    content_blocks.append(tool_use_block)
                    logger.debug(
                        f"🔧 TOOL_DEBUG: Successfully created tool_use block for {tool_call.function.name}"
                    )
                except Exception as e:
                    logger.error(f"🔧 TOOL_DEBUG: Failed tool_call data: {tool_call}")
                    logger.error(f"🔧 TOOL_DEBUG: Error: {e}")

        # Determine stop reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls" or (finish_reason is None and tool_calls):
            stop_reason = "tool_use"

        # Extract usage in Claude format
        enhanced_usage = extract_usage_from_openai_response(openai_response)

        # Create response object
        response = ClaudeMessagesResponse(
            id=response_id,
            type="message",
            role="assistant",
            model=original_request.model,
            content=content_blocks,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=enhanced_usage,
        )

        return response

    except Exception as e:
        logger.error(f"Error converting OpenAI response: {e}")
        logger.error(f"Error details: {type(e)}")
        raise


class AnthropicStreamingConverter:
    """Encapsulates state and logic for converting OpenAI streaming responses to Anthropic format."""

    def __init__(self, original_request: ClaudeMessagesRequest):
        self.original_request = original_request
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Content tracking
        self.content_block_index = 0
        self.current_content_blocks = []
        self.accumulated_text = ""
        self.accumulated_thinking = ""

        # Block state tracking
        self.text_block_started = False
        self.text_block_closed = False
        self.thinking_block_started = False
        self.thinking_block_closed = False
        self.is_tool_use = False
        self.tool_block_closed = False

        # Tool call state - support multiple simultaneous tool calls by index
        self.tool_calls = {}  # index -> {id, name, json_accumulator, content_block_index}
        self.active_tool_indices = set()  # Track which tool indices are active
        self.thinking_content_block_index = None

        # Response state
        self.has_sent_stop_reason = False
        # Calculate input tokens locally for message_start event
        self.input_tokens = original_request.calculate_tokens()
        self.completion_tokens = 0
        self.output_tokens = 0
        self.fallback_output_tokens = 0  # Tiktoken-based fallback counter
        self.has_server_usage = False  # Track if we received server usage data
        self.openai_chunks_received = 0

    def _send_message_start_event(self) -> str:
        """Send message_start event."""
        message_data = {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": 1  # Following Anthropic convention
                },
            },
        }
        event_str = f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: message_start - message_id: {self.message_id}, model: {self.original_request.model}"
        )
        return event_str

    def _send_ping_event(self) -> str:
        """Send ping event."""
        event_data = {"type": "ping"}
        event_str = f"event: ping\ndata: {json.dumps(event_data)}\n\n"
        logger.debug("STREAMING_EVENT: ping")
        return event_str

    def _send_content_block_start_event(self, block_type: str, **kwargs) -> str:
        """Send content_block_start event."""
        content_block = {"type": block_type, **kwargs}
        if block_type == "text":
            content_block["text"] = ""
        elif block_type == "tool_use":
            # Ensure tool_use blocks have required fields
            if "id" not in content_block:
                content_block["id"] = generate_unique_id("tool")
            if "name" not in content_block:
                content_block["name"] = ""
            if "input" not in content_block:
                content_block["input"] = {}
        event_data = {
            "type": "content_block_start",
            "index": self.content_block_index,
            "content_block": content_block,
        }
        event_str = f"event: content_block_start\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: content_block_start - index: {self.content_block_index}, block_type: {block_type}, kwargs: {kwargs}"
        )
        return event_str

    def _send_content_block_delta_event(self, delta_type: str, content: str) -> str:
        """Send content_block_delta event."""
        delta = {"type": delta_type}
        if delta_type == "text_delta":
            delta["text"] = content
        elif delta_type == "input_json_delta":
            delta["partial_json"] = content
        elif delta_type == "thinking_delta":
            delta["thinking"] = content
        elif delta_type == "signature_delta":
            delta["signature"] = content
        event_data = {
            "type": "content_block_delta",
            "index": self.content_block_index,
            "delta": delta,
        }
        event_str = f"event: content_block_delta\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: content_block_delta - index: {self.content_block_index}, delta_type: {delta_type}, content_len: {len(content)}"
        )
        return event_str

    def _send_content_block_stop_event(self) -> str:
        """Send content_block_stop event."""
        event_data = {"type": "content_block_stop", "index": self.content_block_index}
        event_str = f"event: content_block_stop\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: content_block_stop - index: {self.content_block_index}"
        )
        return event_str

    def _send_message_delta_event(self, stop_reason: str, output_tokens: int) -> str:
        """Send message_delta event with output token usage information.

        If server usage data is available, include input_tokens to correct
        the initial estimate sent in message_start.
        """
        usage_data: dict = {
            "output_tokens": output_tokens,
        }
        # Include input_tokens if we have server-reported usage
        # This corrects the locally-estimated value sent in message_start
        if self.has_server_usage and self.input_tokens > 0:
            usage_data["input_tokens"] = self.input_tokens

        event_data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage_data,
        }
        event_str = f"event: message_delta\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: message_delta - stop_reason: {stop_reason}, output_tokens: {output_tokens}, input_tokens: {usage_data.get('input_tokens', 'N/A')}"
        )
        return event_str

    def _send_message_stop_event(self) -> str:
        """Send message_stop event (no usage information)."""
        event_data = {
            "type": "message_stop",
        }
        event_str = f"event: message_stop\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            "STREAMING_EVENT: message_stop"
        )
        return event_str

    def _send_done_event(self) -> str:
        """Send done event."""
        event_data = {"type": "done"}
        event_str = f"event: done\ndata: {json.dumps(event_data)}\n\n"
        logger.debug("STREAMING_EVENT: done")
        return event_str

    def is_malformed_tool_json(self, json_str: str) -> bool:
        """Enhanced malformed tool JSON detection."""
        if not json_str or not isinstance(json_str, str):
            return True

        json_stripped = json_str.strip()

        # Empty or whitespace
        if not json_stripped:
            return True

        # Single characters that indicate malformed JSON
        malformed_singles = ["{", "}", "[", "]", ",", ":", '"', "'"]
        if json_stripped in malformed_singles:
            return True

        # Common malformed patterns
        malformed_patterns = [
            '{"',
            '"}',
            "[{",
            "}]",
            "{}",
            "[]",
            "null",
            '""',
            "''",
            " ",
            "",
            "{,",
            ",}",
            "[,",
            ",]",
        ]
        if json_stripped in malformed_patterns:
            return True

        # Incomplete JSON structures
        if (
            json_stripped.startswith("{")
            and not json_stripped.endswith("}")
            and len(json_stripped) < 15
        ):
            return True

        if (
            json_stripped.startswith("[")
            and not json_stripped.endswith("]")
            and len(json_stripped) < 10
        ):
            return True

        # Check for obviously broken JSON patterns
        if (
            json_stripped.count("{") != json_stripped.count("}")
            and len(json_stripped) < 20
        ):
            return True

        if (
            json_stripped.count("[") != json_stripped.count("]")
            and len(json_stripped) < 20
        ):
            return True

        # Check for trailing malformed characters
        if json_stripped.endswith("}]") or json_stripped.endswith("},]"):
            return True

        # Check for malformed JSON syntax patterns
        malformed_syntax_patterns = [
            ":}",  # Missing value before closing brace
            ":,",  # Missing value before comma
            ":{",  # Missing value, nested object
            ":]",  # Missing value before closing bracket
        ]

        return any(pattern in json_stripped for pattern in malformed_syntax_patterns)

    def try_repair_tool_json(self, json_str: str) -> tuple[dict, bool]:
        """Try to repair malformed tool JSON and return (parsed_json, was_repaired)."""
        if not json_str or not isinstance(json_str, str):
            return {}, False

        json_stripped = json_str.strip()

        # If already valid JSON, return as-is
        try:
            parsed = json.loads(json_stripped)
            if isinstance(parsed, dict):
                return parsed, False
        except json.JSONDecodeError:
            pass

        # Remove common trailing issues
        repaired = json_stripped
        repaired = re.sub(r",\s*\]", "]", repaired)
        repaired = re.sub(r",\s*\}", "}", repaired)
        repaired = re.sub(r",\s*$", "", repaired)

        # Attempt to fix unbalanced braces/brackets
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        while close_braces > open_braces and repaired.endswith("}"):
            repaired = repaired[:-1]
            close_braces -= 1
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)

        open_brackets = repaired.count("[")
        close_brackets = repaired.count("]")
        while close_brackets > open_brackets and repaired.endswith("]"):
            repaired = repaired[:-1]
            close_brackets -= 1
        if open_brackets > close_brackets:
            repaired += "]" * (open_brackets - close_brackets)

        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed, True
        except json.JSONDecodeError:
            pass

        return {}, False

    async def _close_thinking_block(self):
        """Close the current thinking block if open."""
        if self.thinking_block_started and not self.thinking_block_closed:
            self.thinking_block_closed = True
            yield self._send_content_block_stop_event()
            self.content_block_index += 1
            self.thinking_content_block_index = None

    async def _prepare_finalization(self, stop_reason: str):
        """Prepare for stream finalization by closing any open blocks."""
        # Close any open blocks
        if self.thinking_block_started and not self.thinking_block_closed:
            async for event in self._close_thinking_block():
                yield event

        if (
            self.text_block_started
            and not self.text_block_closed
            or self.is_tool_use
            and not getattr(self, "tool_block_closed", False)
        ):
            yield self._send_content_block_stop_event()

        # Finalize any pending tool calls
        for tool_index in list(self.tool_calls.keys()):
            tool_info = self.tool_calls[tool_index]
            if tool_info.get("json_accumulator"):
                json_acc = tool_info["json_accumulator"]
                if self.is_malformed_tool_json(json_acc):
                    repaired, was_repaired = self.try_repair_tool_json(json_acc)
                    if repaired or was_repaired:
                        if tool_info["content_block_index"] < len(self.current_content_blocks):
                            self.current_content_blocks[tool_info["content_block_index"]]["input"] = repaired
                        else:
                            logger.error(
                                "🔧 TOOL_DEBUG: content_block_index out of range during finalization (%s >= %s)",
                                tool_info["content_block_index"],
                                len(self.current_content_blocks),
                            )

        # Clear tool call state
        self.tool_calls = {}
        self.active_tool_indices = set()
        self.is_tool_use = False

        # Calculate final tokens: use server-reported if available, otherwise fallback
        if self.has_server_usage and self.output_tokens > 0:
            final_output_tokens = self.output_tokens
            logger.debug(
                f"Finalization - Using server-reported tokens: {final_output_tokens}"
            )
        else:
            final_output_tokens = self.fallback_output_tokens
            logger.debug(
                f"Finalization - Using tiktoken fallback: {final_output_tokens}"
            )

        yield self._send_message_delta_event(stop_reason, final_output_tokens)
        yield self._send_message_stop_event()
        yield self._send_done_event()

    async def _handle_text_delta(self, content: str):
        """Handle text content deltas."""
        if self.is_tool_use:
            yield self._send_content_block_stop_event()
            self.is_tool_use = False
            self.tool_block_closed = True

        if not self.text_block_started:
            # If a tool_use or thinking block is open, close it first
            if self.thinking_block_started and not self.thinking_block_closed:
                async for event in self._close_thinking_block():
                    yield event

            # Start a new text block
            yield self._send_content_block_start_event("text")
            self.text_block_started = True

        # Send text delta
        yield self._send_content_block_delta_event("text_delta", content)

        # Update accumulated text and fallback token count
        self.accumulated_text += content
        self.fallback_output_tokens += len(_tokenizer.encode(content))

    async def _handle_thinking_delta(self, content: str):
        """Handle thinking content deltas."""
        if not self.thinking_block_started:
            # Close text block if open
            if self.text_block_started and not self.text_block_closed:
                yield self._send_content_block_stop_event()
                self.text_block_closed = True

            # Start a new thinking block
            yield self._send_content_block_start_event("thinking")
            self.thinking_block_started = True
            if len(self.current_content_blocks) < self.content_block_index:
                logger.debug(
                    "🔧 TOOL_DEBUG: Padding content blocks before thinking block (index=%s, len=%s)",
                    self.content_block_index,
                    len(self.current_content_blocks),
                )
                while len(self.current_content_blocks) < self.content_block_index:
                    self.current_content_blocks.append({"type": "text", "text": ""})
            if len(self.current_content_blocks) == self.content_block_index:
                self.thinking_content_block_index = self.content_block_index
                self.current_content_blocks.append({"type": "thinking", "thinking": ""})

        # Send thinking delta
        yield self._send_content_block_delta_event("thinking_delta", content)

        # Update accumulated thinking and fallback token count
        self.accumulated_thinking += content
        self.fallback_output_tokens += len(_tokenizer.encode(content))
        if self.thinking_content_block_index is not None:
            self.current_content_blocks[self.thinking_content_block_index]["thinking"] += content

    async def _handle_tool_call_delta(self, tool_call):
        """Handle tool call deltas and manage tool_use blocks."""
        logger.debug(f"🔧 TOOL_DEBUG: Raw tool_call received: {tool_call}")
        logger.debug(f"🔧 TOOL_DEBUG: tool_call type: {type(tool_call)}")

        # Extract tool call index
        if isinstance(tool_call, dict):
            tool_index = tool_call.get("index")
            logger.debug(f"🔧 TOOL_DEBUG: Dict tool_call keys: {list(tool_call.keys())}")
        elif hasattr(tool_call, "index"):
            tool_index = tool_call.index
            logger.debug(f"🔧 TOOL_DEBUG: Object tool_call attributes: {dir(tool_call)}")
        else:
            logger.debug(f"🔧 TOOL_DEBUG: Unexpected tool_call type: {type(tool_call)}")
            return

        if tool_index is None:
            logger.debug("🔧 TOOL_DEBUG: tool_call missing index")
            return

        if tool_index not in self.tool_calls:
            async for event in self._initialize_new_tool_call(tool_call, tool_index):
                yield event

        async for event in self._process_tool_call_arguments(tool_call, tool_index):
            yield event

    async def _initialize_new_tool_call(self, tool_call, tool_index):
        """Initialize a new tool call block."""
        logger.debug(f"🔧 TOOL_DEBUG: Initializing tool call - raw data: {tool_call}")

        if isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            tool_call_id = tool_call.get("id")
            tool_name = function.get("name", "")
        else:
            function = getattr(tool_call, "function", None)
            tool_call_id = getattr(tool_call, "id", None)
            tool_name = getattr(function, "name", "") if function else ""

        # Ensure tool ID uniqueness
        if not tool_call_id or tool_call_id in self.active_tool_indices:
            tool_call_id = generate_unique_id("toolu")

        # Close any open text or thinking blocks
        if self.text_block_started and not self.text_block_closed:
            yield self._send_content_block_stop_event()
            self.text_block_closed = True

        if self.thinking_block_started and not self.thinking_block_closed:
            async for event in self._close_thinking_block():
                yield event

        # Initialize tool call state
        self.tool_calls[tool_index] = {
            "id": tool_call_id,
            "name": tool_name,
            "json_accumulator": "",
            "content_block_index": self.content_block_index,
        }
        self.active_tool_indices.add(tool_index)
        self.is_tool_use = True
        self.tool_block_closed = False

        # Send tool_use block start
        yield self._send_content_block_start_event(
            "tool_use", id=tool_call_id, name=tool_name
        )

        # Count tokens for tool name
        if tool_name:
            self.fallback_output_tokens += len(_tokenizer.encode(tool_name))

        # Add tool_use block to current content blocks for tracking
        self.current_content_blocks.append(
            {"type": "tool_use", "id": tool_call_id, "name": tool_name, "input": {}}
        )
        self.content_block_index += 1

    async def _process_tool_call_arguments(self, tool_call, tool_index):
        """Process tool call arguments and update JSON accumulator."""
        logger.debug(f"🔧 TOOL_DEBUG: tool_call data: {tool_call}")

        if isinstance(tool_call, dict) and "function" in tool_call:
            function = tool_call.get("function", {})
        elif hasattr(tool_call, "function"):
            function = getattr(tool_call, "function", None)
        else:
            logger.debug("🔧 TOOL_DEBUG: No function found in tool_call")
            return

        arguments = ""
        if isinstance(function, dict):
            arguments = function.get("arguments", "")
        else:
            arguments = getattr(function, "arguments", "")

        tool_info = self.tool_calls[tool_index]
        if arguments:
            arg_str = str(arguments)
            arg_stripped = arg_str.strip()
            if arg_stripped.startswith("{") and arg_stripped.endswith("}"):
                tool_info["json_accumulator"] = arg_str
            else:
                tool_info["json_accumulator"] += arg_str
            yield self._send_content_block_delta_event("input_json_delta", arg_str)
            # Count tokens for tool call arguments
            self.fallback_output_tokens += len(_tokenizer.encode(arg_str))

        # Try to parse partial JSON to update input
        if tool_info["json_accumulator"]:
            parsed_args, was_repaired = self.try_repair_tool_json(tool_info["json_accumulator"])
            if parsed_args:
                if tool_info["content_block_index"] < len(self.current_content_blocks):
                    self.current_content_blocks[tool_info["content_block_index"]]["input"] = parsed_args
                else:
                    logger.error(
                        "🔧 TOOL_DEBUG: content_block_index out of range (%s >= %s)",
                        tool_info["content_block_index"],
                        len(self.current_content_blocks),
                    )
                if was_repaired:
                    logger.debug("🔧 TOOL_DEBUG: Repaired malformed tool JSON")

    async def process_chunk(self, chunk: ChatCompletionChunk):
        """Process a single OpenAI stream chunk and yield Anthropic events."""
        self.openai_chunks_received += 1

        # Extract usage from chunk (OpenAI sends this in final chunk when stream_options.include_usage=true)
        if hasattr(chunk, "usage") and chunk.usage:
            usage = chunk.usage
            self.has_server_usage = True
            if hasattr(usage, "prompt_tokens") and usage.prompt_tokens:
                self.input_tokens = usage.prompt_tokens
            if hasattr(usage, "completion_tokens") and usage.completion_tokens:
                self.output_tokens = usage.completion_tokens
            # Store completion_tokens_details if available (for reasoning_tokens)
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                if hasattr(details, "reasoning_tokens") and details.reasoning_tokens:
                    self.completion_tokens = details.reasoning_tokens
            logger.debug(
                f"Extracted server usage: input={self.input_tokens}, output={self.output_tokens}, fallback={self.fallback_output_tokens}"
            )

        # Extract delta and finish reason
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason
        else:
            delta = None
            finish_reason = None

        # Handle tool calls
        if delta and hasattr(delta, "tool_calls") and delta.tool_calls:
            delta_tool_calls = delta.tool_calls
            if not isinstance(delta_tool_calls, list):
                delta_tool_calls = [delta_tool_calls]

            for tool_call in delta_tool_calls:
                try:
                    async for event in self._handle_tool_call_delta(tool_call):
                        yield event
                except Exception as e:
                    logger.error(f"🔧 TOOL_DEBUG: Error in tool_call_delta: {e}")

        # Handle text content
        if delta and delta.content:
            async for event in self._handle_text_delta(delta.content):
                yield event

        # Handle reasoning content (thinking)
        if delta and hasattr(delta, "reasoning_content") and delta.reasoning_content:
            if isinstance(delta.reasoning_content, str):
                async for event in self._handle_thinking_delta(delta.reasoning_content):
                    yield event

        # Handle finish reason
        if finish_reason:
            stop_reason = "end_turn"
            if finish_reason == "length":
                stop_reason = "max_tokens"
            elif finish_reason == "tool_calls":
                stop_reason = "tool_use"

            async for event in self._prepare_finalization(stop_reason):
                yield event
            self.has_sent_stop_reason = True


async def convert_openai_streaming_response_to_anthropic(
    response_generator: AsyncStream[ChatCompletionChunk],
    original_request: ClaudeMessagesRequest,
    model_id: str = "",
):
    """Handle streaming responses from OpenAI SDK and convert to Anthropic format.

    Optimized version using state management class to improve performance.
    """
    # Create converter instance with all state encapsulated
    converter = AnthropicStreamingConverter(original_request)

    # Enhanced error recovery tracking
    consecutive_errors = 0
    max_consecutive_errors = 5  # Max consecutive errors before aborting

    try:
        # Send initial events
        yield converter._send_message_start_event()
        yield converter._send_ping_event()

        logger.debug(f"🌊 Starting streaming for model: {original_request.model}")

        # Process each chunk directly with enhanced error handling
        chunk_count = 0
        try:
            async for chunk in response_generator:
                chunk_count += 1
                try:
                    # Process chunk and yield all events
                    async for event in converter.process_chunk(chunk):
                        yield event

                    # Reset consecutive errors on successful processing
                    consecutive_errors = 0

                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"🌊 ERROR_PROCESSING_CHUNK #{chunk_count}: {str(e)}")

                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"Too many consecutive errors ({consecutive_errors}), aborting stream."
                        )
                        break
                    continue
        except Exception as streaming_error:
            logger.error(f"TOOL_DEBUG: Streaming iteration error after {chunk_count} chunks: {streaming_error}")
            log_openai_api_error(streaming_error, "streaming_iteration")

            # Log tool-related context if applicable
            error_str = str(streaming_error).lower()
            if "function" in error_str or "tool" in error_str:
                logger.error(f"TOOL_DEBUG: Tool-related error for model {original_request.model}")
                if original_request.tools:
                    tool_names = [tool.name for tool in original_request.tools]
                    logger.error(f"TOOL_DEBUG: Tools in request: {tool_names}")

            raise streaming_error

        # Handle stream completion - ensure proper cleanup regardless of how stream ended
        if not converter.has_sent_stop_reason:
            logger.debug("Stream ended without finish_reason, performing cleanup")

            # Close any open blocks
            if converter.thinking_block_started and not converter.thinking_block_closed:
                async for event in converter._close_thinking_block():
                    yield event

            # If no blocks started, create empty text block
            if (
                not converter.text_block_started
                and not converter.is_tool_use
                and not converter.thinking_block_started
            ):
                text_block = {"type": "text", "text": ""}
                converter.current_content_blocks.append(text_block)
                yield converter._send_content_block_start_event("text")
                yield converter._send_content_block_stop_event()
            elif (
                converter.text_block_started
                and not converter.text_block_closed
                or converter.is_tool_use
                and not getattr(converter, "tool_block_closed", False)
            ):
                logger.debug("STREAMING_EVENT: content_block_stop - index: 0")
                yield converter._send_content_block_stop_event()

            # Calculate final tokens: use server-reported if available, otherwise fallback
            if converter.has_server_usage and converter.output_tokens > 0:
                final_output_tokens = converter.output_tokens
                logger.debug(
                    f"No finish reason - Using server-reported tokens: {final_output_tokens}"
                )
            else:
                final_output_tokens = converter.fallback_output_tokens
                logger.debug(
                    f"No finish reason - Using tiktoken fallback: {final_output_tokens}"
                )

            # Determine appropriate stop_reason based on content and pending finish_reason
            if (
                hasattr(converter, "pending_finish_reason")
                and converter.pending_finish_reason == "tool_calls"
            ) or converter.is_tool_use:
                stop_reason = "tool_use"
            else:
                stop_reason = "end_turn"
            yield converter._send_message_delta_event(stop_reason, final_output_tokens)
            yield converter._send_message_stop_event()
            yield converter._send_done_event()

    finally:
        # Log streaming completion
        _log_streaming_completion(converter, original_request, model_id)


def _log_streaming_completion(
    converter: AnthropicStreamingConverter,
    original_request: ClaudeMessagesRequest,
    model_id: str = "",
):
    """Log a detailed summary of the streaming completion."""
    try:
        # Use server-reported tokens if available, otherwise fallback
        if converter.has_server_usage and converter.output_tokens > 0:
            final_output_tokens = converter.output_tokens
        else:
            final_output_tokens = converter.fallback_output_tokens

        # Use server-reported input tokens if available
        input_tokens = converter.input_tokens if converter.has_server_usage else 0

        # Update global usage stats
        from ..types import ClaudeUsage, global_usage_stats

        usage = ClaudeUsage(
            input_tokens=input_tokens,
            output_tokens=final_output_tokens,
        )
        global_usage_stats.update_usage(usage, original_request.model)

        # Log detailed summary
        content_blocks_summary = []
        tool_calls_summary = []
        for i, block in enumerate(converter.current_content_blocks):
            if block.get("type") == "text":
                content_blocks_summary.append(
                    f"Block {i}: text ({len(block.get('text', ''))} chars)"
                )
            elif block.get("type") == "tool_use":
                tool_name = block.get("name", "unknown")
                tool_calls_summary.append(
                    {
                        "name": tool_name,
                        "id": block.get("id", "unknown"),
                        "input": block.get("input", {}),
                    }
                )
            elif block.get("type") == "thinking":
                content_blocks_summary.append(
                    f"Block {i}: thinking ({len(block.get('thinking', ''))} chars)"
                )

        if content_blocks_summary:
            logger.info(
                f"📝 STREAMING_SUMMARY: {', '.join(content_blocks_summary)}"
            )
        if tool_calls_summary:
            logger.info(
                f"🔧 STREAMING_TOOL_CALLS: {len(tool_calls_summary)} tool calls"
            )
            for tool_call in tool_calls_summary:
                logger.info(f"🔧   Tool: {tool_call['name']} (id: {tool_call['id']})")
                logger.info(f"🔧   Input: {json.dumps(tool_call['input'], indent=2)}")

        logger.info(
            f"🌊 STREAMING_COMPLETE: Model={original_request.model}, Text={len(converter.accumulated_text)} chars, Thinking={len(converter.accumulated_thinking)} chars, OutputTokens={final_output_tokens}"
        )

    except Exception as e:
        logger.error(f"Error logging streaming completion: {e}")
