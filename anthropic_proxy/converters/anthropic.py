"""
Anthropic identity converter.

This converter handles Anthropic-to-Anthropic pass-through with minimal transformation.
Used when both source and target formats are Anthropic Messages format.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import tiktoken

from ..types import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeUsage,
    global_usage_stats,
)
from .base import BaseConverter, BaseStreamingConverter

logger = logging.getLogger(__name__)

# Tokenizer for fallback token counting
_tokenizer = tiktoken.get_encoding("cl100k_base")


class AnthropicConverter(BaseConverter):
    """
    Identity converter for Anthropic Messages format.

    Since Anthropic is the pivot format, most conversions are pass-through.
    """

    def request_to_anthropic(self, payload: dict[str, Any]) -> ClaudeMessagesRequest:
        """Parse raw Anthropic request payload into ClaudeMessagesRequest."""
        return ClaudeMessagesRequest.model_validate(payload)

    def request_from_anthropic(
        self,
        request: ClaudeMessagesRequest,
        *,
        model_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert ClaudeMessagesRequest to dict for Anthropic API."""
        payload = request.model_dump(exclude_none=True, by_alias=True)
        if model_id:
            payload["model"] = model_id
        return payload

    def response_to_anthropic(
        self,
        response: Any,
        original_request: ClaudeMessagesRequest,
    ) -> ClaudeMessagesResponse:
        """Parse Anthropic API response into ClaudeMessagesResponse."""
        if isinstance(response, ClaudeMessagesResponse):
            return response
        if isinstance(response, dict):
            return ClaudeMessagesResponse.model_validate(response)
        # Handle httpx Response or similar
        if hasattr(response, "json"):
            data = response.json()
            return ClaudeMessagesResponse.model_validate(data)
        raise TypeError(f"Cannot convert response of type {type(response)}")

    def response_from_anthropic(
        self,
        response: ClaudeMessagesResponse,
    ) -> dict[str, Any]:
        """Convert ClaudeMessagesResponse to dict for client."""
        return response.model_dump(exclude_none=True, by_alias=True)


class AnthropicStreamingConverter(BaseStreamingConverter):
    """
    Identity streaming converter for Anthropic SSE format.

    Pass-through with minimal transformation.
    """

    async def stream_to_anthropic(
        self,
        stream: AsyncIterator[Any],
        original_request: ClaudeMessagesRequest,
    ) -> AsyncIterator[str]:
        """Pass through Anthropic SSE stream."""
        async for chunk in stream:
            if isinstance(chunk, str):
                yield chunk
            elif isinstance(chunk, bytes):
                yield chunk.decode("utf-8")
            elif hasattr(chunk, "data"):
                yield f"data: {chunk.data}\n\n"
            else:
                yield str(chunk)

    async def stream_from_anthropic(
        self,
        stream: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        """Pass through Anthropic SSE stream."""
        async for chunk in stream:
            yield chunk


async def convert_anthropic_streaming_with_usage_tracking(
    stream: AsyncIterator[str],
    original_request: ClaudeMessagesRequest,
    model_id: str = "",
) -> AsyncIterator[str]:
    """
    Wrap Anthropic SSE stream to extract and track usage.

    Handles raw chunks from aiter_text() that may contain multiple SSE events
    or partial events split across chunks.

    Parses SSE events to extract:
    - input_tokens, cache_creation_input_tokens, cache_read_input_tokens from message_start
    - output_tokens from message_delta (cumulative, final count)

    Uses tiktoken fallback for counting when server doesn't provide usage.
    """
    from ..types import SSEEventParser

    # Server-reported usage (full usage dict to preserve all fields)
    # message_delta contains cumulative final counts, so we use that
    server_usage: dict = {}
    has_server_usage = False

    # Fallback token counting
    fallback_output_tokens = 0

    # Buffer for accumulating partial SSE events
    parser = SSEEventParser()

    async for chunk in stream:
        # Pass through the chunk immediately
        yield chunk

        # Parse complete SSE events from buffer
        events = parser.feed(chunk)

        for event in events:
            event_type = event.type

            # message_start contains initial usage with input_tokens
            if event_type == "message_start" and event.message:
                usage = event.message.get("usage", {})
                if usage:
                    server_usage.update(usage)
                    has_server_usage = True
                    logger.debug(f"message_start usage: {usage}")

            # message_delta contains CUMULATIVE final usage (this is the accurate count)
            elif event_type == "message_delta" and event.usage:
                # message_delta.usage contains cumulative counts - use these
                server_usage.update(event.usage)
                has_server_usage = True
                logger.debug(f"message_delta usage (cumulative): {event.usage}")

            # content_block_delta - count tokens for fallback
            elif event_type == "content_block_delta" and event.delta:
                delta_type = event.delta.get("type")

                if delta_type == "text_delta":
                    text = event.delta.get("text", "")
                    fallback_output_tokens += len(_tokenizer.encode(text))
                elif delta_type == "thinking_delta":
                    thinking = event.delta.get("thinking", "")
                    fallback_output_tokens += len(_tokenizer.encode(thinking))
                elif delta_type == "input_json_delta":
                    partial_json = event.delta.get("partial_json", "")
                    fallback_output_tokens += len(_tokenizer.encode(partial_json))

            # content_block_start - count tool name tokens for fallback
            elif event_type == "content_block_start" and event.content_block:
                if event.content_block.get("type") in ("tool_use", "server_tool_use"):
                    tool_name = event.content_block.get("name", "")
                    if tool_name:
                        fallback_output_tokens += len(_tokenizer.encode(tool_name))

    # Build ClaudeUsage from server data or fallback
    if has_server_usage:
        usage = ClaudeUsage(
            input_tokens=server_usage.get("input_tokens", 0),
            output_tokens=server_usage.get("output_tokens", 0),
            cache_creation_input_tokens=server_usage.get("cache_creation_input_tokens"),
            cache_read_input_tokens=server_usage.get("cache_read_input_tokens"),
        )
        logger.debug(
            f"Anthropic streaming - Server usage: input={usage.input_tokens}, "
            f"output={usage.output_tokens}, cache_create={usage.cache_creation_input_tokens}, "
            f"cache_read={usage.cache_read_input_tokens}"
        )
    else:
        input_tokens = original_request.calculate_tokens()
        usage = ClaudeUsage(
            input_tokens=input_tokens,
            output_tokens=fallback_output_tokens,
        )
        logger.debug(
            f"Anthropic streaming - Tiktoken fallback: input={input_tokens}, output={fallback_output_tokens}"
        )

    global_usage_stats.update_usage(usage, model_id or original_request.model)
