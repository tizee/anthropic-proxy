"""
Base converter protocol for format conversion.

Defines the interface that all format converters must implement.
The Anthropic Messages format is used as the internal pivot format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from ..types import ClaudeMessagesRequest, ClaudeMessagesResponse


class BaseConverter(ABC):
    """
    Abstract base class for format converters.

    All converters use Anthropic Messages format as the pivot:
    - request_to_anthropic: Convert incoming request to Anthropic format
    - request_from_anthropic: Convert Anthropic request to target format
    - response_to_anthropic: Convert provider response to Anthropic format
    - response_from_anthropic: Convert Anthropic response to client format
    """

    @abstractmethod
    def request_to_anthropic(self, payload: dict[str, Any]) -> ClaudeMessagesRequest:
        """
        Convert incoming request from this format to Anthropic Messages format.

        Args:
            payload: Raw request payload in this converter's format

        Returns:
            ClaudeMessagesRequest in Anthropic format
        """
        ...

    @abstractmethod
    def request_from_anthropic(
        self,
        request: ClaudeMessagesRequest,
        *,
        model_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Convert Anthropic Messages request to this format for sending to provider.

        Args:
            request: ClaudeMessagesRequest in Anthropic format
            model_id: Target model identifier
            **kwargs: Additional format-specific options

        Returns:
            Request payload in this converter's format
        """
        ...

    @abstractmethod
    def response_to_anthropic(
        self,
        response: Any,
        original_request: ClaudeMessagesRequest,
    ) -> ClaudeMessagesResponse:
        """
        Convert provider response to Anthropic Messages format.

        Args:
            response: Response from the provider in this converter's format
            original_request: The original Anthropic request (for context)

        Returns:
            ClaudeMessagesResponse in Anthropic format
        """
        ...

    @abstractmethod
    def response_from_anthropic(
        self,
        response: ClaudeMessagesResponse,
    ) -> dict[str, Any]:
        """
        Convert Anthropic Messages response to this format for client.

        Args:
            response: ClaudeMessagesResponse in Anthropic format

        Returns:
            Response payload in this converter's format
        """
        ...


class BaseStreamingConverter(ABC):
    """
    Abstract base class for streaming format converters.

    Handles SSE chunk conversion between formats.
    """

    @abstractmethod
    async def stream_to_anthropic(
        self,
        stream: AsyncIterator[Any],
        original_request: ClaudeMessagesRequest,
    ) -> AsyncIterator[str]:
        """
        Convert streaming response to Anthropic SSE format.

        Args:
            stream: Async iterator of chunks from provider
            original_request: The original Anthropic request (for context)

        Yields:
            SSE-formatted strings in Anthropic format
        """
        ...

    @abstractmethod
    async def stream_from_anthropic(
        self,
        stream: AsyncIterator[str],
    ) -> AsyncIterator[str]:
        """
        Convert Anthropic SSE stream to this format for client.

        Args:
            stream: Async iterator of Anthropic SSE strings

        Yields:
            SSE-formatted strings in this converter's format
        """
        ...
