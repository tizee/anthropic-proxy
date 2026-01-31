"""
OpenAI ↔ Anthropic converter.

Converts between OpenAI Chat Completions format and Anthropic Messages format.
Re-exports existing conversion logic from openai_converter.py with the new interface.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai.types.chat import ChatCompletion

from ..types import (
    ClaudeContentBlockImage,
    ClaudeContentBlockImageBase64Source,
    ClaudeContentBlockImageURLSource,
    ClaudeContentBlockText,
    ClaudeContentBlockThinking,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockToolUse,
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeTool,
    ClaudeToolChoiceAny,
    ClaudeToolChoiceAuto,
    ClaudeToolChoiceNone,
    ClaudeToolChoiceTool,
    generate_unique_id,
)
from ._openai_impl import (
    convert_openai_response_to_anthropic,
    convert_openai_streaming_response_to_anthropic,
)
from .base import BaseConverter, BaseStreamingConverter

logger = logging.getLogger(__name__)


def _parse_openai_tool_choice(tool_choice: Any) -> Any:
    """Convert OpenAI tool_choice to Claude tool_choice."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return ClaudeToolChoiceAuto()
        elif tool_choice == "required":
            return ClaudeToolChoiceAny()
        elif tool_choice == "none":
            return ClaudeToolChoiceNone()
    elif isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            function = tool_choice.get("function", {})
            name = function.get("name", "")
            if name:
                return ClaudeToolChoiceTool(name=name)
    return None


def _parse_openai_tools(tools: list[dict[str, Any]] | None) -> list[ClaudeTool] | None:
    """Convert OpenAI tools to Claude tools."""
    if not tools:
        return None

    claude_tools = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        function = tool.get("function", {})
        claude_tools.append(
            ClaudeTool(
                name=function.get("name", ""),
                description=function.get("description"),
                input_schema=function.get("parameters", {"type": "object", "properties": {}}),
            )
        )
    return claude_tools if claude_tools else None


def _parse_openai_message_content(content: Any) -> str | list[Any]:
    """Convert OpenAI message content to Claude content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    # Handle array content (multimodal)
    if isinstance(content, list):
        blocks = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                blocks.append(ClaudeContentBlockText(type="text", text=part.get("text", "")))
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url", "")
                if url.startswith("data:"):
                    # Parse data URL
                    try:
                        header, data = url.split(",", 1)
                        media_type = header.split(";")[0].replace("data:", "")
                        blocks.append(
                            ClaudeContentBlockImage(
                                type="image",
                                source=ClaudeContentBlockImageBase64Source(
                                    type="base64",
                                    media_type=media_type,
                                    data=data,
                                ),
                            )
                        )
                    except ValueError:
                        logger.warning(f"Failed to parse data URL: {url[:50]}...")
                else:
                    blocks.append(
                        ClaudeContentBlockImage(
                            type="image",
                            source=ClaudeContentBlockImageURLSource(type="url", url=url),
                        )
                    )
        return blocks if blocks else ""

    return str(content)


def _parse_openai_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[ClaudeMessage]]:
    """
    Convert OpenAI messages to Claude messages.

    Returns:
        tuple of (system_prompt, messages)
    """
    system_prompt = None
    claude_messages: list[ClaudeMessage] = []

    # Track tool calls for matching with tool results
    pending_tool_calls: dict[str, dict[str, Any]] = {}

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            # Accumulate system messages
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            if system_prompt:
                system_prompt += "\n" + content
            else:
                system_prompt = content

        elif role == "user":
            content = _parse_openai_message_content(msg.get("content"))
            claude_messages.append(ClaudeMessage(role="user", content=content))

        elif role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls", [])

            blocks: list[Any] = []

            # Handle reasoning_content (thinking) if present
            reasoning = msg.get("reasoning_content")
            if reasoning:
                blocks.append(
                    ClaudeContentBlockThinking(
                        type="thinking",
                        thinking=reasoning,
                        signature=generate_unique_id("thinking"),
                    )
                )

            # Handle text content
            if content:
                if isinstance(content, str):
                    blocks.append(ClaudeContentBlockText(type="text", text=content))
                elif isinstance(content, list):
                    parsed = _parse_openai_message_content(content)
                    if isinstance(parsed, list):
                        blocks.extend(parsed)
                    elif parsed:
                        blocks.append(ClaudeContentBlockText(type="text", text=parsed))

            # Handle tool calls
            for tc in tool_calls:
                tc_id = tc.get("id", generate_unique_id("toolu"))
                function = tc.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments", "{}")

                try:
                    args_dict = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    args_dict = {}

                blocks.append(
                    ClaudeContentBlockToolUse(
                        type="tool_use",
                        id=tc_id,
                        name=name,
                        input=args_dict,
                    )
                )
                pending_tool_calls[tc_id] = {"name": name}

            if blocks:
                claude_messages.append(ClaudeMessage(role="assistant", content=blocks))
            elif not content and not tool_calls:
                # Empty assistant message
                claude_messages.append(ClaudeMessage(role="assistant", content=""))

        elif role == "tool":
            # Tool result message
            tool_call_id = msg.get("tool_call_id", "")
            content = msg.get("content", "")

            # Need to add this as part of a user message with tool_result
            tool_result = ClaudeContentBlockToolResult(
                type="tool_result",
                tool_use_id=tool_call_id,
                content=content,
            )

            # Check if last message is a user message we can append to
            if claude_messages and claude_messages[-1].role == "user":
                last_content = claude_messages[-1].content
                if isinstance(last_content, list):
                    last_content.append(tool_result)
                else:
                    claude_messages[-1].content = [
                        ClaudeContentBlockText(type="text", text=last_content) if last_content else tool_result,
                        tool_result,
                    ] if last_content else [tool_result]
            else:
                # Create new user message with tool result
                claude_messages.append(ClaudeMessage(role="user", content=[tool_result]))

    return system_prompt, claude_messages


class OpenAIConverter(BaseConverter):
    """
    Converter between OpenAI Chat Completions and Anthropic Messages format.
    """

    def request_to_anthropic(self, payload: dict[str, Any]) -> ClaudeMessagesRequest:
        """Convert OpenAI Chat Completions request to Anthropic Messages format."""
        model = payload.get("model", "")
        messages = payload.get("messages", [])
        max_tokens = payload.get("max_tokens") or payload.get("max_completion_tokens", 4096)
        temperature = payload.get("temperature")
        top_p = payload.get("top_p")
        stop = payload.get("stop")
        stream = payload.get("stream", False)
        tools = payload.get("tools")
        tool_choice = payload.get("tool_choice")

        # Parse messages
        system_prompt, claude_messages = _parse_openai_messages(messages)

        # Build request
        request_data: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": claude_messages,
            "stream": stream,
        }

        if system_prompt:
            request_data["system"] = system_prompt
        if temperature is not None:
            request_data["temperature"] = temperature
        if top_p is not None:
            request_data["top_p"] = top_p
        if stop:
            request_data["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        # Convert tools
        claude_tools = _parse_openai_tools(tools)
        if claude_tools:
            request_data["tools"] = claude_tools

        # Convert tool_choice
        claude_tool_choice = _parse_openai_tool_choice(tool_choice)
        if claude_tool_choice:
            request_data["tool_choice"] = claude_tool_choice

        return ClaudeMessagesRequest.model_validate(request_data)

    def request_from_anthropic(
        self,
        request: ClaudeMessagesRequest,
        *,
        model_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert Anthropic Messages request to OpenAI format."""
        return request.to_openai_request()

    def response_to_anthropic(
        self,
        response: Any,
        original_request: ClaudeMessagesRequest,
    ) -> ClaudeMessagesResponse:
        """Convert OpenAI response to Anthropic format."""
        if isinstance(response, ChatCompletion):
            return convert_openai_response_to_anthropic(response, original_request)
        if isinstance(response, dict):
            # Parse dict as ChatCompletion
            completion = ChatCompletion.model_validate(response)
            return convert_openai_response_to_anthropic(completion, original_request)
        raise TypeError(f"Cannot convert response of type {type(response)}")

    def response_from_anthropic(
        self,
        response: ClaudeMessagesResponse,
    ) -> dict[str, Any]:
        """Convert Anthropic response to OpenAI format."""
        # Build OpenAI-style response
        content = None
        tool_calls = []
        reasoning_content = None

        for block in response.content:
            if isinstance(block, ClaudeContentBlockText):
                if content is None:
                    content = block.text
                else:
                    content += block.text
            elif isinstance(block, ClaudeContentBlockThinking):
                reasoning_content = block.thinking
            elif isinstance(block, ClaudeContentBlockToolUse):
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input, ensure_ascii=False),
                    },
                })

        # Determine finish_reason
        finish_reason = "stop"
        if response.stop_reason == "end_turn":
            finish_reason = "stop"
        elif response.stop_reason == "max_tokens":
            finish_reason = "length"
        elif response.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif response.stop_reason == "stop_sequence":
            finish_reason = "stop"

        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        choice = {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }

        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return {
            "id": response.id.replace("msg_", "chatcmpl-"),
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": response.model,
            "choices": [choice],
            "usage": usage,
        }


class OpenAIToAnthropicStreamingConverter(BaseStreamingConverter):
    """
    Streaming converter from OpenAI chunks to Anthropic SSE format.

    Wraps the existing AnthropicStreamingConverter from openai_converter.py.
    """

    async def stream_to_anthropic(
        self,
        stream: AsyncIterator[Any],
        original_request: ClaudeMessagesRequest,
    ) -> AsyncIterator[str]:
        """Convert OpenAI streaming response to Anthropic SSE format."""

        async for event in convert_openai_streaming_response_to_anthropic(
            stream, original_request
        ):
            yield event

    async def stream_from_anthropic(
        self,
        stream: AsyncIterator[str],
        model: str = "",
    ) -> AsyncIterator[str]:
        """
        Convert Anthropic SSE stream to OpenAI SSE format.

        Anthropic events:
        - message_start: {type: "message_start", message: {...}}
        - content_block_start: {type: "content_block_start", index: N, content_block: {...}}
        - content_block_delta: {type: "content_block_delta", index: N, delta: {...}}
        - content_block_stop: {type: "content_block_stop", index: N}
        - message_delta: {type: "message_delta", delta: {...}, usage: {...}}
        - message_stop: {type: "message_stop"}

        OpenAI events:
        - {id, object: "chat.completion.chunk", choices: [{index, delta: {...}, finish_reason}], ...}
        - data: [DONE]
        """
        import time

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        # Track current state
        current_tool_index = -1
        tool_call_ids: dict[int, str] = {}  # block index -> tool call id

        async for event_str in stream:
            # Parse SSE event
            if not event_str.strip():
                continue

            # Extract data from SSE format
            data_line = None
            for line in event_str.strip().split("\n"):
                if line.startswith("data: "):
                    data_line = line[6:]
                    break

            if not data_line or data_line == "[DONE]":
                yield "data: [DONE]\n\n"
                continue

            try:
                event = json.loads(data_line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "message_start":
                # Send initial chunk with role
                msg = event.get("message", {})
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model or msg.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            elif event_type == "content_block_start":
                block = event.get("content_block", {})
                block_type = block.get("type", "")
                block_index = event.get("index", 0)

                if block_type == "tool_use":
                    # Start a tool call
                    current_tool_index += 1
                    tool_id = block.get("id", f"call_{uuid.uuid4().hex[:24]}")
                    tool_call_ids[block_index] = tool_id

                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "tool_calls": [{
                                    "index": current_tool_index,
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": "",
                                    },
                                }]
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                elif block_type == "thinking":
                    # Thinking block - emit as reasoning_content in OpenAI format (if supported)
                    pass

            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type", "")
                block_index = event.get("index", 0)

                if delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                elif delta_type == "input_json_delta":
                    # Tool call argument delta
                    partial_json = delta.get("partial_json", "")
                    if partial_json and block_index in tool_call_ids:
                        # Find the tool call index for this block
                        tc_index = list(tool_call_ids.keys()).index(block_index)
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": tc_index,
                                        "function": {"arguments": partial_json},
                                    }]
                                },
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                elif delta_type == "thinking_delta":
                    # Thinking content - could emit as reasoning delta if format supports
                    pass

            elif event_type == "message_delta":
                # Final message delta with stop reason
                delta = event.get("delta", {})
                stop_reason = delta.get("stop_reason", "")

                finish_reason = "stop"
                if stop_reason == "end_turn":
                    finish_reason = "stop"
                elif stop_reason == "max_tokens":
                    finish_reason = "length"
                elif stop_reason == "tool_use":
                    finish_reason = "tool_calls"
                elif stop_reason == "stop_sequence":
                    finish_reason = "stop"

                # Send final chunk with finish_reason
                usage = event.get("usage", {})
                chunk: dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }],
                }
                if usage:
                    chunk["usage"] = {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    }
                yield f"data: {json.dumps(chunk)}\n\n"

            elif event_type == "message_stop":
                yield "data: [DONE]\n\n"
