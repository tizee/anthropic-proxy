"""
Tests for mid-stream abort behavior.

Ensures that when upstream API errors occur during streaming, the proxy
behaves like the real Anthropic API by dropping the connection abruptly.
This triggers client-side retry logic as expected.
"""

import pytest
from fastapi import HTTPException

from anthropic_proxy.converters.anthropic import (
    convert_anthropic_streaming_with_usage_tracking,
)
from anthropic_proxy.midstream_abort import MidStreamAbort
from anthropic_proxy.types import ClaudeMessagesRequest


class TestMidStreamAbortBehavior:
    """Tests that mid-stream errors abort the connection (like real API)."""

    @pytest.fixture
    def basic_request(self):
        return ClaudeMessagesRequest(
            model="claude-3-opus",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

    @pytest.mark.asyncio
    async def test_mid_stream_http_exception_propagates(self, basic_request):
        """HTTPException mid-stream should propagate (not yield error event)."""
        sse_events = [
            'event: message_start\n',
            'data: {"type": "message_start", "message": {"id": "msg_123"}}\n',
            '\n',
        ]

        async def failing_stream():
            for event in sse_events:
                yield event
            raise HTTPException(status_code=502, detail="Connection failed")

        # Exception should propagate (not be caught)
        with pytest.raises(HTTPException):
            async for _ in convert_anthropic_streaming_with_usage_tracking(
                failing_stream(),
                basic_request,
                "test-model",
            ):
                pass

    @pytest.mark.asyncio
    async def test_mid_stream_connection_error_propagates(self, basic_request):
        """ConnectionError mid-stream should propagate."""
        sse_events = [
            'event: message_start\n',
            'data: {"type": "message_start"}\n',
            '\n',
        ]

        async def failing_stream():
            for event in sse_events:
                yield event
            raise ConnectionError("Connection reset by peer")

        # Exception should propagate
        with pytest.raises(ConnectionError):
            async for _ in convert_anthropic_streaming_with_usage_tracking(
                failing_stream(),
                basic_request,
                "test-model",
            ):
                pass

    @pytest.mark.asyncio
    async def test_normal_streaming_works(self, basic_request):
        """Normal streaming without errors should work normally."""
        sse_events = [
            'event: message_start\n',
            'data: {"type": "message_start", "message": {"id": "msg_123"}}\n',
            '\n',
            'event: message_delta\n',
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}\n',
            '\n',
        ]

        async def normal_stream():
            for event in sse_events:
                yield event

        chunks = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            normal_stream(),
            basic_request,
            "test-model",
        ):
            chunks.append(chunk)

        assert chunks == sse_events


class TestMidStreamAbortException:
    """Tests for the MidStreamAbort exception class."""

    def test_midstream_abort_is_exception(self):
        """MidStreamAbort should be an Exception subclass."""
        assert issubclass(MidStreamAbort, Exception)

    def test_midstream_abort_can_carry_message(self):
        """MidStreamAbort should carry an error message."""
        exc = MidStreamAbort("upstream 502")
        assert str(exc) == "upstream 502"

    def test_midstream_abort_can_be_raised(self):
        """MidStreamAbort should be raisable."""
        with pytest.raises(MidStreamAbort):
            raise MidStreamAbort("test error")
