"""
Streaming response processing for Claude to OpenAI API conversion.
This module contains the AnthropicStreamingConverter class and related streaming functions.
"""

import json
import logging
import uuid

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from .converter import parse_function_calls_from_thinking
from .types import (
    ClaudeMessagesRequest,
    generate_unique_id,
)
from .utils import log_openai_api_error

logger = logging.getLogger(__name__)


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

        # Response state
        self.has_sent_stop_reason = False
        # Calculate input tokens locally for message_start event
        self.input_tokens = original_request.calculate_tokens()
        self.completion_tokens = 0
        self.output_tokens = 0
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
        """Send message_delta event with output token usage information."""
        event_data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "output_tokens": output_tokens,
            },
        }
        event_str = f"event: message_delta\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: message_delta - stop_reason: {stop_reason}, output_tokens: {output_tokens}"
        )
        return event_str

    def _send_message_stop_event(self) -> str:
        """Send message_stop event (no usage information)."""
        event_data = {
            "type": "message_stop",
        }
        event_str = f"event: message_stop\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: message_stop"
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

        # Try parsing as-is first
        try:
            return json.loads(json_stripped), False
        except json.JSONDecodeError:
            pass

        # Try to find complete JSON objects in the string
        brace_count = 0
        start_pos = -1

        for i, char in enumerate(json_stripped):
            if char == "{":
                if start_pos == -1:
                    start_pos = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    # Found complete JSON object
                    json_candidate = json_stripped[start_pos : i + 1]
                    try:
                        parsed = json.loads(json_candidate)
                        return parsed, True
                    except json.JSONDecodeError:
                        continue

        # Try removing common trailing artifacts
        repair_attempts = [
            json_stripped.rstrip("]"),  # Remove trailing ]
            json_stripped.rstrip("},]"),  # Remove trailing },]
            json_stripped.rstrip("}]"),  # Remove trailing }]
            json_stripped.rstrip(","),  # Remove trailing comma
        ]

        for attempt in repair_attempts:
            if attempt != json_stripped:  # Only try if it's different
                try:
                    parsed = json.loads(attempt)
                    return parsed, True
                except json.JSONDecodeError:
                    continue

        # If all repair attempts fail, return empty dict
        return {}, False

    async def _close_thinking_block(self):
        """Close the current thinking block if open."""
        if self.thinking_block_started and not self.thinking_block_closed:
            # Parse function calls from thinking content before closing
            cleaned_thinking, function_calls = parse_function_calls_from_thinking(
                self.accumulated_thinking
            )

            # Update thinking content with cleaned version
            if self.content_block_index < len(self.current_content_blocks):
                self.current_content_blocks[self.content_block_index]["thinking"] = (
                    cleaned_thinking
                )

                # Generate signature for cleaned thinking content
                thinking_signature = generate_unique_id("thinking")
                self.current_content_blocks[self.content_block_index]["signature"] = (
                    thinking_signature
                )

                # Send signature delta before closing thinking block
                yield self._send_content_block_delta_event(
                    "signature_delta", thinking_signature
                )

            yield self._send_content_block_stop_event()
            self.thinking_block_closed = True
            self.content_block_index += 1

            # Add function call blocks if any were found
            for tool_call in function_calls:
                # Create tool use content block
                unique_tool_id = generate_unique_id("toolu")
                tool_block = {
                    "type": "tool_use",
                    "id": unique_tool_id,
                    "name": tool_call["function"]["name"],
                    "input": json.loads(tool_call["function"]["arguments"]),
                }
                self.current_content_blocks.append(tool_block)
                logger.debug(f"Adding tool call block: {tool_block}")

                # Send tool call events
                yield self._send_content_block_start_event(
                    "tool_use", id=unique_tool_id, name=tool_call["function"]["name"]
                )
                yield self._send_content_block_delta_event(
                    "input_json_delta", tool_call["function"]["arguments"]
                )
                yield self._send_content_block_stop_event()
                self.content_block_index += 1

    async def _handle_text_delta(self, delta_content: str):
        """Handle text content delta."""
        # If we have text content and we're currently in tool use mode, end the tool use first
        if self.is_tool_use:
            # Close all active tool calls
            for tool_index in sorted(self.active_tool_indices):
                tool_info = self.tool_calls[tool_index]
                content_block_idx = tool_info["content_block_index"]

                # Set content_block_index to the tool's index for the stop event
                original_index = self.content_block_index
                self.content_block_index = content_block_idx
                yield self._send_content_block_stop_event()
                self.content_block_index = original_index

            self.tool_block_closed = True
            self.is_tool_use = False

        self.accumulated_text += delta_content

        logger.debug(
            f"Added text content: +{len(delta_content)} chars, total: {len(self.accumulated_text)} chars"
        )

        # Start text block if not started
        if not self.text_block_started:
            text_block = {"type": "text", "text": ""}
            self.current_content_blocks.append(text_block)
            yield self._send_content_block_start_event("text")
            self.text_block_started = True

        # Send text delta
        yield self._send_content_block_delta_event("text_delta", delta_content)

        # Update content block
        if self.content_block_index < len(self.current_content_blocks):
            self.current_content_blocks[self.content_block_index]["text"] = (
                self.accumulated_text
            )

    async def _handle_thinking_delta(self, delta_reasoning: str):
        """Handle thinking/reasoning content delta."""
        self.accumulated_thinking += delta_reasoning
        logger.debug(
            f"Added thinking content: +{len(delta_reasoning)} chars, total: {len(self.accumulated_thinking)} chars"
        )

        # Start thinking block if not started
        if not self.thinking_block_started:
            thinking_block = {"type": "thinking", "thinking": ""}
            self.current_content_blocks.append(thinking_block)
            yield self._send_content_block_start_event("thinking")
            self.thinking_block_started = True

        # Send thinking delta
        yield self._send_content_block_delta_event("thinking_delta", delta_reasoning)

        # Update content block
        if self.content_block_index < len(self.current_content_blocks):
            self.current_content_blocks[self.content_block_index]["thinking"] = (
                self.accumulated_thinking
            )

    async def _handle_tool_call_delta(self, tool_call):
        """Handle tool call delta with support for multiple simultaneous tool calls."""
        # GROQ DEBUG: Log the raw tool call data
        logger.debug(f"🔧 TOOL_DEBUG: Raw tool_call received: {tool_call}")
        logger.debug(f"🔧 TOOL_DEBUG: tool_call type: {type(tool_call)}")

        # Extract tool call index - this is crucial for multiple tool calls
        tool_index = None
        if isinstance(tool_call, dict):
            tool_index = tool_call.get("index")
            logger.debug(f"🔧 TOOL_DEBUG: Dict tool_call keys: {list(tool_call.keys())}")
        elif hasattr(tool_call, "index"):
            tool_index = tool_call.index
            logger.debug(f"🔧 TOOL_DEBUG: Object tool_call attributes: {dir(tool_call)}")
        else:
            logger.debug(f"🔧 TOOL_DEBUG: Unexpected tool_call type: {type(tool_call)}")

        if tool_index is None:
            logger.warning(
                "🔧 TOOL_CALL_DELTA: Missing tool call index, defaulting to 0"
            )
            tool_index = 0

        logger.debug(
            f"🔧 TOOL_CALL_DELTA: Processing tool call index {tool_index}, active_tools={list(self.active_tool_indices)}"
        )

        # Handle text content before first tool call if needed
        if not self.is_tool_use and not self.active_tool_indices:
            async for event in self._handle_text_before_tools():
                yield event

        # Check if this is a new tool call (first time seeing this index)
        if tool_index not in self.tool_calls:
            async for event in self._initialize_new_tool_call(tool_call, tool_index):
                yield event

        # Handle arguments for this specific tool call
        async for event in self._process_tool_call_arguments(tool_call, tool_index):
            yield event

    async def _handle_text_before_tools(self):
        """Handle accumulated text before starting tool calls."""
        if self.accumulated_text and not self.text_block_started:
            logger.debug(
                f"🔧 TOOL_CALL_DELTA: Handling accumulated text first ({len(self.accumulated_text)} chars)"
            )
            text_block = {"type": "text", "text": ""}
            self.current_content_blocks.append(text_block)
            yield self._send_content_block_start_event("text")
            self.text_block_started = True

            yield self._send_content_block_delta_event(
                "text_delta", self.accumulated_text
            )

            self.current_content_blocks[self.content_block_index]["text"] = (
                self.accumulated_text
            )

            yield self._send_content_block_stop_event()
            self.text_block_closed = True
            self.content_block_index += 1
        elif self.text_block_started and not self.text_block_closed:
            logger.debug(
                "🔧 TOOL_CALL_DELTA: Closing open text block before starting tool"
            )
            yield self._send_content_block_stop_event()
            self.text_block_closed = True
            self.content_block_index += 1

    async def _initialize_new_tool_call(self, tool_call, tool_index):
        """Initialize a new tool call at the given index."""
        # GROQ DEBUG: Log tool call structure during initialization
        logger.debug(f"🔧 TOOL_DEBUG: Initializing tool call - raw data: {tool_call}")

        # Extract tool info
        if isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            tool_name = function.get("name", "") if isinstance(function, dict) else ""
            logger.debug(f"🔧 TOOL_DEBUG: Dict function data: {function}")
        else:
            function = getattr(tool_call, "function", None)
            tool_name = getattr(function, "name", "") if function else ""
            logger.debug(f"🔧 TOOL_DEBUG: Object function data: {function}")
            if function:
                logger.debug(f"🔧 TOOL_DEBUG: Function attributes: {dir(function)}")

        tool_id = generate_unique_id("toolu")

        logger.debug(f"🔧 TOOL_DEBUG: Extracted tool_name: '{tool_name}', tool_id: '{tool_id}'")

        # Create tool call entry
        self.tool_calls[tool_index] = {
            "id": tool_id,
            "name": tool_name,
            "json_accumulator": "",
            "content_block_index": self.content_block_index,
        }

        self.active_tool_indices.add(tool_index)
        self.is_tool_use = True

        logger.debug(
            f"🔧 TOOL_CALL_DELTA: Initialized new tool call - index: {tool_index}, name: {tool_name}, id: {tool_id}, block_index: {self.content_block_index}"
        )

        # Create tool use block
        tool_block = {
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": {},
        }
        self.current_content_blocks.append(tool_block)

        yield self._send_content_block_start_event(
            "tool_use", id=tool_id, name=tool_name
        )

        # Move to next content block index for next tool
        self.content_block_index += 1

    async def _process_tool_call_arguments(self, tool_call, tool_index):
        """Process arguments for a specific tool call."""
        # GROQ DEBUG: Log arguments extraction process
        logger.debug(f"🔧 TOOL_DEBUG: Processing arguments for tool {tool_index}")
        logger.debug(f"🔧 TOOL_DEBUG: tool_call data: {tool_call}")

        # Extract function arguments
        arguments = None
        if isinstance(tool_call, dict) and "function" in tool_call:
            function = tool_call.get("function", {})
            arguments = (
                function.get("arguments", "") if isinstance(function, dict) else ""
            )
            logger.debug(f"🔧 TOOL_DEBUG: Dict function arguments: '{arguments}'")
        elif hasattr(tool_call, "function"):
            function = getattr(tool_call, "function", None)
            arguments = getattr(function, "arguments", "") if function else ""
            logger.debug(f"🔧 TOOL_DEBUG: Object function arguments: '{arguments}'")
        else:
            logger.debug("🔧 TOOL_DEBUG: No function found in tool_call")

        if not arguments:
            logger.debug(f"🔧 TOOL_CALL_DELTA: No arguments for tool {tool_index}")
            return

        logger.debug(f"🔧 TOOL_DEBUG: Extracted arguments length: {len(arguments)}")

        tool_info = self.tool_calls[tool_index]
        accumulated_json = tool_info["json_accumulator"]
        content_block_idx = tool_info["content_block_index"]

        # Check if this contains new arguments or is a repetition
        if accumulated_json and arguments.startswith(accumulated_json):
            # This is cumulative - extract only the new part
            new_arguments = arguments[len(accumulated_json) :]
            if new_arguments:
                logger.debug(
                    f"🔧 TOOL_CALL_DELTA: Tool {tool_index} - extracting {len(new_arguments)} new chars from cumulative {len(arguments)}"
                )
                arguments = new_arguments
            else:
                logger.debug(
                    f"🔧 TOOL_CALL_DELTA: Tool {tool_index} - no new content in cumulative update, skipping"
                )
                return
        elif accumulated_json and arguments == accumulated_json:
            logger.debug(
                f"🔧 TOOL_CALL_DELTA: Tool {tool_index} - exact duplicate arguments, skipping"
            )
            return

        # Update accumulator
        tool_info["json_accumulator"] += arguments

        logger.debug(
            f"🔧 TOOL_CALL_DELTA: Tool {tool_index} - added {len(arguments)} chars, total: {len(tool_info['json_accumulator'])}"
        )

        # Try to parse JSON to update the content block
        try:
            parsed_json = json.loads(tool_info["json_accumulator"])
            self.current_content_blocks[content_block_idx]["input"] = parsed_json
            logger.debug(
                f"🔧 TOOL_CALL_DELTA: Tool {tool_index} - successfully parsed complete JSON"
            )
        except json.JSONDecodeError as e:
            logger.debug(
                f"🔧 TOOL_CALL_DELTA: Tool {tool_index} - JSON not complete yet (pos {e.pos}), continuing to accumulate"
            )

        # Send the delta - need to use the correct content block index
        logger.debug(
            f"🔧 TOOL_CALL_DELTA: Tool {tool_index} - sending input_json_delta with {len(arguments)} chars for block {content_block_idx}"
        )

        # Temporarily set content_block_index to the tool's index for the delta event
        original_index = self.content_block_index
        self.content_block_index = content_block_idx
        yield self._send_content_block_delta_event("input_json_delta", arguments)
        self.content_block_index = original_index

    async def _prepare_finalization(self, finish_reason: str):
        """Prepare for finalization by closing blocks, but don't send stop events yet."""
        logger.debug(f"🔚 PREPARE_FINALIZATION: finish_reason={finish_reason}")

        # Close thinking block if it was started
        if self.thinking_block_started and not self.thinking_block_closed:
            logger.debug("🔚 PREPARE_FINALIZATION: Closing thinking block")
            async for event in self._close_thinking_block():
                yield event

        # Handle tool use completion - finalize all active tool calls
        if self.is_tool_use and self.active_tool_indices:
            logger.debug(
                f"🔚 PREPARE_FINALIZATION: Finalizing {len(self.active_tool_indices)} tool calls"
            )

            # Process each active tool call
            for tool_index in sorted(self.active_tool_indices):
                tool_info = self.tool_calls[tool_index]
                tool_json = tool_info["json_accumulator"]
                content_block_idx = tool_info["content_block_index"]

                # Ensure tool JSON is complete and parsed
                if tool_json:
                    # Try to repair and parse the tool JSON
                    final_parsed_json, was_repaired = self.try_repair_tool_json(
                        tool_json
                    )

                    if final_parsed_json:
                        self.current_content_blocks[content_block_idx]["input"] = (
                            final_parsed_json
                        )
                    else:
                        logger.error(
                            f"🔚 PREPARE_FINALIZATION: Tool {tool_index} - Failed to parse or repair tool JSON"
                        )
                        # Set empty input as fallback
                        self.current_content_blocks[content_block_idx]["input"] = {}

                # Close the tool use block
                original_index = self.content_block_index
                self.content_block_index = content_block_idx
                yield self._send_content_block_stop_event()
                self.content_block_index = original_index

            self.tool_block_closed = True

        # Handle text block completion
        elif self.text_block_started and not self.text_block_closed:
            logger.debug("🔚 PREPARE_FINALIZATION: Closing open text block")
            yield self._send_content_block_stop_event()
            self.text_block_closed = True
            self.content_block_index += 1

        # If we haven't started any blocks yet, start and immediately close a text block
        elif (
            not self.text_block_started
            and not self.is_tool_use
            and not self.thinking_block_started
        ):
            logger.debug(
                "🔚 PREPARE_FINALIZATION: No blocks started, creating empty text block"
            )
            text_block = {"type": "text", "text": ""}
            self.current_content_blocks.append(text_block)
            yield self._send_content_block_start_event("text")
            yield self._send_content_block_stop_event()
            self.text_block_started = True
            self.text_block_closed = True
            self.content_block_index += 1

    async def _send_final_events(self):
        """Send final events after both finish_reason and usage have been processed."""
        if not hasattr(self, "pending_finish_reason") or self.has_sent_stop_reason:
            return

        finish_reason = self.pending_finish_reason
        logger.debug(f"Sending final events for finish_reason: {finish_reason}")

        # Determine stop reason
        stop_reason = _map_finish_reason_to_stop_reason(finish_reason)
        logger.debug(f"Mapped stop_reason: {stop_reason}")

        # Use the updated token counts
        final_output_tokens = self.output_tokens

        # Send message delta with final content and stop reason
        yield self._send_message_delta_event(stop_reason, final_output_tokens)

        # Send message stop and done
        yield self._send_message_stop_event()
        yield self._send_done_event()
        logger.debug("Streaming completed successfully")

        self.has_sent_stop_reason = True

    async def process_chunk(self, chunk: ChatCompletionChunk):
        """Process a single chunk from the OpenAI streaming response."""
        self.openai_chunks_received += 1

        # Pre-extract all data from Pydantic object to minimize attribute access
        chunk_data = {
            "chunk_id": chunk.id,
            "has_choices": len(chunk.choices) > 0,
            "has_usage": chunk.usage is not None,
            "usage": chunk.usage,
        }

        # Extract choice data if available
        if chunk_data["has_choices"]:
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # Pre-extract delta data to minimize model_dump() calls
            raw_delta = delta.model_dump() if hasattr(delta, "model_dump") else {}

            # GROQ DEBUG: Log detailed delta information
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                logger.debug(f"🔧 TOOL_DEBUG: Delta has tool_calls: {delta.tool_calls}")
                logger.debug(f"🔧 TOOL_DEBUG: raw_delta content: {raw_delta}")

            chunk_data.update(
                {
                    "delta": delta,
                    "finish_reason": finish_reason,
                    "delta_content": getattr(delta, "content", None),
                    "delta_tool_calls": getattr(delta, "tool_calls", None),
                    "delta_reasoning": raw_delta.get("reasoning_content")
                    or raw_delta.get("reasoning"),
                }
            )

        # Debug logging for streaming chunk analysis
        logger.debug(f"🔄 STREAMING_CHUNK #{self.openai_chunks_received}: processing")

        # GROQ DEBUG: Log raw chunk data structure
        if chunk_data["has_choices"] and chunk_data["delta_tool_calls"]:
            logger.debug(f"🔧 TOOL_DEBUG: Chunk has tool calls - raw chunk: {chunk}")
            logger.debug(f"🔧 TOOL_DEBUG: chunk.choices[0].delta attributes: {dir(chunk.choices[0].delta)}")
            if hasattr(chunk.choices[0].delta, 'tool_calls'):
                logger.debug(f"🔧 TOOL_DEBUG: tool_calls attribute value: {chunk.choices[0].delta.tool_calls}")
                logger.debug(f"🔧 TOOL_DEBUG: tool_calls type: {type(chunk.choices[0].delta.tool_calls)}")

        # GROQ DEBUG: Log ALL chunks with their complete data structure
        logger.debug(f"🔧 TOOL_DEBUG: Raw chunk: {chunk}")
        logger.debug(f"🔧 TOOL_DEBUG: Chunk data: {chunk_data}")

        # GROQ DEBUG: Special attention to error-related chunks
        if chunk_data["has_choices"] and chunk_data["finish_reason"] == "error":
            logger.error(f"🔧 TOOL_DEBUG: ERROR chunk received! Full chunk: {chunk}")
            logger.error(f"🔧 TOOL_DEBUG: Error chunk data: {chunk_data}")
            logger.error(f"🔧 TOOL_DEBUG: Delta content: {chunk_data['delta_content']}")
            logger.error(f"🔧 TOOL_DEBUG: Accumulated text so far: '{self.accumulated_text}'")

        # Handle usage data (final chunk)
        if chunk_data["has_usage"] and chunk_data["usage"]:
            usage = chunk_data["usage"]

            # Handle both object and dictionary usage formats
            if hasattr(usage, "prompt_tokens"):
                self.input_tokens = usage.prompt_tokens
                self.completion_tokens = usage.completion_tokens
            elif isinstance(usage, dict):
                self.input_tokens = usage.get("prompt_tokens", self.input_tokens)
                self.completion_tokens = usage.get("completion_tokens", self.output_tokens)
            else:
                # Fallback to getattr for any other format
                self.input_tokens = getattr(usage, "prompt_tokens", self.input_tokens)
                self.completion_tokens = getattr(usage, "completion_tokens", self.output_tokens)

            # Sync output_tokens with completion_tokens for consistency
            self.output_tokens = self.completion_tokens
            logger.debug(
                f"Usage chunk received - Input: {self.input_tokens}, Output: {self.output_tokens}"
            )

            # Now that we have usage data, send final events if finish_reason was already processed
            if hasattr(self, "pending_finish_reason") and not self.has_sent_stop_reason:
                async for event in self._send_final_events():
                    yield event

        # Process content if we have choices
        if chunk_data["has_choices"]:
            # Handle tool calls first
            if chunk_data["delta_tool_calls"]:
                # If reasoning was streamed, close the thinking block before starting tool_use blocks
                if self.thinking_block_started and not self.thinking_block_closed:
                    logger.debug(
                        "🔚 PRE-TOOL_CALL: Closing thinking block before tool calls"
                    )
                    async for event in self._close_thinking_block():
                        yield event

                delta_tool_calls = chunk_data["delta_tool_calls"]
                logger.debug(f"🔧 TOOL_DEBUG: Raw delta_tool_calls: {delta_tool_calls}")
                logger.debug(f"🔧 TOOL_DEBUG: delta_tool_calls type: {type(delta_tool_calls)}")

                if not isinstance(delta_tool_calls, list):
                    delta_tool_calls = [delta_tool_calls]
                    logger.debug(f"🔧 TOOL_DEBUG: Converted to list: {delta_tool_calls}")

                for i, tool_call in enumerate(delta_tool_calls):
                    logger.debug(f"🔧 TOOL_DEBUG: Processing tool_call {i}: {tool_call}")
                    try:
                        async for event in self._handle_tool_call_delta(tool_call):
                            yield event
                    except Exception as e:
                        logger.error(f"🔧 TOOL_DEBUG: Error in tool_call_delta for item {i}: {e}")
                        logger.error(f"🔧 TOOL_DEBUG: Failed tool_call data: {tool_call}")
                        # Re-raise the error to maintain existing error handling
                        raise

            # Handle thinking/reasoning content
            if chunk_data["delta_reasoning"]:
                async for event in self._handle_thinking_delta(
                    chunk_data["delta_reasoning"]
                ):
                    yield event
            elif (
                self.thinking_block_started
                and not self.thinking_block_closed
                and chunk_data["delta_content"]
            ):
                # If we have normal content coming and thinking was active, close thinking block
                async for event in self._close_thinking_block():
                    yield event

            # Handle text content
            if chunk_data["delta_content"] and not self.is_tool_use:
                async for event in self._handle_text_delta(chunk_data["delta_content"]):
                    yield event

            # Process finish_reason - but wait for usage chunk before finalizing
            if chunk_data["finish_reason"] and not self.has_sent_stop_reason:
                # GROQ DEBUG: Log finish_reason details
                logger.debug(f"🔧 TOOL_DEBUG: Processing finish_reason: {chunk_data['finish_reason']}")
                logger.debug(f"🔧 TOOL_DEBUG: Accumulated text: '{self.accumulated_text}'")
                logger.debug(f"🔧 TOOL_DEBUG: Current tokens: input={self.input_tokens}, output={self.output_tokens}")

                # Store finish_reason and prepare for finalization, but don't send stop events yet
                self.pending_finish_reason = chunk_data["finish_reason"]
                async for event in self._prepare_finalization(
                    chunk_data["finish_reason"]
                ):
                    yield event

                # If we already have usage data, send final events immediately
                if self.output_tokens > 0:  # Usage was already processed
                    async for event in self._send_final_events():
                        yield event


def _map_finish_reason_to_stop_reason(finish_reason: str) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    if finish_reason == "length":
        return "max_tokens"
    elif finish_reason == "tool_calls":
        return "tool_use"
    else:
        return "end_turn"


def _calculate_accurate_output_tokens(
    accumulated_text: str,
    accumulated_thinking: str,
    reported_tokens: int,
    context: str = "",
) -> int:
    """Use server-reported tokens for proxy service (no manual calculation needed)."""
    # For a proxy service, always use the authoritative usage data from the API
    final_tokens = reported_tokens

    logger.debug(
        f"{context} - Token usage - Reported: {reported_tokens}, Using: {final_tokens} (server-reported)"
    )
    return final_tokens


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

            # Calculate final tokens and send completion events
            final_output_tokens = _calculate_accurate_output_tokens(
                converter.accumulated_text,
                converter.accumulated_thinking,
                converter.output_tokens,
                "No finish reason received",
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
        # Calculate final tokens for tracking
        final_output_tokens = _calculate_accurate_output_tokens(
            converter.accumulated_text,
            converter.accumulated_thinking,
            converter.output_tokens,
            "Final streaming cleanup",
        )

        # Input token counting removed - rely on API usage data only
        input_tokens = 0

        # Update global usage stats
        from .types import ClaudeUsage, global_usage_stats

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
                tool_id = block.get("id", "unknown")
                input_data = block.get("input", {})
                tool_calls_summary.append(
                    {"name": tool_name, "id": tool_id, "input": input_data}
                )
                content_blocks_summary.append(
                    f"Block {i}: tool_use (name={tool_name}, input_keys={list(input_data.keys())})"
                )
            elif block.get("type") == "thinking":
                content_blocks_summary.append(
                    f"Block {i}: thinking ({len(block.get('thinking', ''))} chars)"
                )

        logger.info(
            f"STREAMING COMPLETE - Model: {original_request.model}, "
            f"Chunks: {converter.openai_chunks_received}, "
            f"Input tokens: {input_tokens}, Output tokens: {final_output_tokens}, "
            f"Text: {len(converter.accumulated_text)} chars, "
            f"Thinking: {len(converter.accumulated_thinking)} chars"
        )
        if content_blocks_summary:
            logger.info(
                f"📋 STREAMING_CONTENT_BLOCKS: {len(converter.current_content_blocks)} blocks"
            )
            for block_summary in content_blocks_summary:
                logger.info(f"📋   {block_summary}")
        if tool_calls_summary:
            logger.info(
                f"🔧 STREAMING_TOOL_CALLS: {len(tool_calls_summary)} tool calls"
            )
            for tool_call in tool_calls_summary:
                logger.info(f"🔧   Tool: {tool_call['name']} (id: {tool_call['id']})")
                logger.info(f"🔧   Input: {json.dumps(tool_call['input'], indent=2)}")
        else:
            logger.info("🔧 STREAMING_TOOL_CALLS: No tool calls")

    except Exception as cleanup_error:
        logger.error(f"Error in streaming cleanup logging: {cleanup_error}")
