"""
Anthropic identity converter.

This converter handles Anthropic-to-Anthropic pass-through with minimal transformation.
Used when both source and target formats are Anthropic Messages format.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from .base import BaseConverter, BaseStreamingConverter
from ..types import ClaudeMessagesRequest, ClaudeMessagesResponse


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
