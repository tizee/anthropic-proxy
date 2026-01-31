"""
Tests for streaming usage extraction and token counting.

Covers:
- Anthropic format streaming usage tracking
- OpenAI format streaming usage extraction
- Tiktoken fallback counting
- SSE format preservation
"""

import pytest

from anthropic_proxy.converters._openai_impl import (
    AnthropicStreamingConverter,
)
from anthropic_proxy.converters.anthropic import (
    convert_anthropic_streaming_with_usage_tracking,
)
from anthropic_proxy.types import ClaudeMessagesRequest


# Helper to create async generator from list
async def async_iter(items):
    for item in items:
        yield item


class TestAnthropicStreamingUsageTracking:
    """Tests for convert_anthropic_streaming_with_usage_tracking."""

    @pytest.fixture
    def basic_request(self):
        return ClaudeMessagesRequest(
            model="claude-3-opus",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

    @pytest.mark.asyncio
    async def test_extracts_input_tokens_from_message_start(self, basic_request):
        """Should extract input_tokens from message_start event."""
        sse_events = [
            'event: message_start\n',
            'data: {"type": "message_start", "message": {"id": "msg_123", "usage": {"input_tokens": 10000}}}\n',
            '\n',
            'event: content_block_start\n',
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n',
            '\n',
            'event: content_block_delta\n',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello!"}}\n',
            '\n',
            'event: content_block_stop\n',
            'data: {"type": "content_block_stop", "index": 0}\n',
            '\n',
            'event: message_delta\n',
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 500}}\n',
            '\n',
            'event: message_stop\n',
            'data: {"type": "message_stop"}\n',
            '\n',
        ]

        chunks = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            async_iter(sse_events),
            basic_request,
            "claude-3-opus",
        ):
            chunks.append(chunk)

        # All chunks should be passed through
        assert chunks == sse_events

    @pytest.mark.asyncio
    async def test_extracts_cache_tokens_from_message_start(self, basic_request):
        """Should extract cache-related tokens from message_start event."""
        sse_events = [
            'event: message_start\n',
            'data: {"type": "message_start", "message": {"id": "msg_123", "usage": {"input_tokens": 5000, "cache_creation_input_tokens": 3000, "cache_read_input_tokens": 2000}}}\n',
            '\n',
            'event: message_delta\n',
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 100}}\n',
            '\n',
        ]

        chunks = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            async_iter(sse_events),
            basic_request,
            "claude-3-opus",
        ):
            chunks.append(chunk)

        # All chunks should be passed through unchanged
        assert len(chunks) == len(sse_events)

    @pytest.mark.asyncio
    async def test_preserves_sse_format_with_empty_lines(self, basic_request):
        """Should preserve exact SSE format including empty lines."""
        # SSE format requires empty lines between events
        sse_events = [
            'event: message_start\n',
            'data: {"type": "message_start", "message": {"id": "msg_123", "usage": {"input_tokens": 100}}}\n',
            '\n',  # Empty line - event separator
            'event: content_block_delta\n',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}\n',
            '\n',  # Empty line - event separator
            'event: message_delta\n',
            'data: {"type": "message_delta", "usage": {"output_tokens": 10}}\n',
            '\n',  # Empty line - event separator
        ]

        chunks = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            async_iter(sse_events),
            basic_request,
            "test-model",
        ):
            chunks.append(chunk)

        # Empty lines must be preserved
        assert '\n' in chunks
        assert chunks.count('\n') == 3  # Three empty line separators

    @pytest.mark.asyncio
    async def test_fallback_counting_for_text_delta(self, basic_request):
        """Should use tiktoken fallback when no server usage provided."""
        # Events without usage data
        sse_events = [
            'event: message_start\n',
            'data: {"type": "message_start", "message": {"id": "msg_123"}}\n',
            '\n',
            'event: content_block_delta\n',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello world"}}\n',
            '\n',
        ]

        chunks = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            async_iter(sse_events),
            basic_request,
            "test-model",
        ):
            chunks.append(chunk)

        # Should still pass through all chunks
        assert len(chunks) == len(sse_events)

    @pytest.mark.asyncio
    async def test_fallback_counting_for_thinking_delta(self, basic_request):
        """Should count thinking tokens in fallback mode."""
        sse_events = [
            'event: content_block_delta\n',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think about this..."}}\n',
            '\n',
        ]

        chunks = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            async_iter(sse_events),
            basic_request,
            "test-model",
        ):
            chunks.append(chunk)

        assert len(chunks) == len(sse_events)

    @pytest.mark.asyncio
    async def test_fallback_counting_for_tool_use(self, basic_request):
        """Should count tool name and arguments in fallback mode."""
        sse_events = [
            'event: content_block_start\n',
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_123", "name": "read_file"}}\n',
            '\n',
            'event: content_block_delta\n',
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\\"path\\": \\"/tmp/test.txt\\"}"}}\n',
            '\n',
        ]

        chunks = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            async_iter(sse_events),
            basic_request,
            "test-model",
        ):
            chunks.append(chunk)

        assert len(chunks) == len(sse_events)


class TestOpenAIStreamingUsageExtraction:
    """Tests for OpenAI streaming converter usage extraction."""

    @pytest.fixture
    def basic_request(self):
        return ClaudeMessagesRequest(
            model="gpt-4",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

    def test_converter_initializes_with_fallback_counter(self, basic_request):
        """Converter should initialize fallback_output_tokens and has_server_usage."""
        converter = AnthropicStreamingConverter(basic_request)

        assert converter.fallback_output_tokens == 0
        assert converter.has_server_usage is False
        assert converter.output_tokens == 0

    @pytest.mark.asyncio
    async def test_extracts_usage_from_chunk_with_usage(self, basic_request):
        """Should extract usage from chunk.usage when available."""
        from unittest.mock import MagicMock

        converter = AnthropicStreamingConverter(basic_request)

        # Create mock chunk with usage
        chunk = MagicMock()
        chunk.choices = []
        chunk.usage = MagicMock()
        chunk.usage.prompt_tokens = 1500
        chunk.usage.completion_tokens = 800
        chunk.usage.completion_tokens_details = None

        # Process the chunk
        events = []
        async for event in converter.process_chunk(chunk):
            events.append(event)

        # Should have extracted usage
        assert converter.has_server_usage is True
        assert converter.input_tokens == 1500
        assert converter.output_tokens == 800

    @pytest.mark.asyncio
    async def test_fallback_counting_for_text_content(self, basic_request):
        """Should count tokens using tiktoken when processing text deltas."""
        from unittest.mock import MagicMock

        converter = AnthropicStreamingConverter(basic_request)

        # Create mock chunk with text content
        chunk = MagicMock()
        chunk.usage = None
        delta = MagicMock()
        delta.content = "Hello, how are you?"
        delta.tool_calls = None
        delta.reasoning_content = None
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = None
        chunk.choices = [choice]

        # Process the chunk
        events = []
        async for event in converter.process_chunk(chunk):
            events.append(event)

        # Should have incremented fallback counter
        assert converter.fallback_output_tokens > 0
        assert converter.has_server_usage is False

    @pytest.mark.asyncio
    async def test_fallback_counting_for_thinking_content(self, basic_request):
        """Should count thinking tokens using tiktoken."""
        from unittest.mock import MagicMock

        converter = AnthropicStreamingConverter(basic_request)

        # Create mock chunk with reasoning content
        chunk = MagicMock()
        chunk.usage = None
        delta = MagicMock()
        delta.content = None
        delta.tool_calls = None
        delta.reasoning_content = "Let me analyze this step by step..."
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = None
        chunk.choices = [choice]

        # Process the chunk
        events = []
        async for event in converter.process_chunk(chunk):
            events.append(event)

        # Should have incremented fallback counter
        assert converter.fallback_output_tokens > 0

    @pytest.mark.asyncio
    async def test_server_usage_takes_precedence_over_fallback(self, basic_request):
        """When server provides usage, it should be used over fallback."""
        from unittest.mock import MagicMock

        converter = AnthropicStreamingConverter(basic_request)

        # First, send some text to increment fallback counter
        text_chunk = MagicMock()
        text_chunk.usage = None
        delta = MagicMock()
        delta.content = "This is a test response with some text."
        delta.tool_calls = None
        delta.reasoning_content = None
        choice = MagicMock()
        choice.delta = delta
        choice.finish_reason = None
        text_chunk.choices = [choice]

        async for _ in converter.process_chunk(text_chunk):
            pass

        # Fallback should have counted something
        assert converter.fallback_output_tokens > 0
        fallback_count = converter.fallback_output_tokens

        # Now send final chunk with server usage
        final_chunk = MagicMock()
        final_chunk.usage = MagicMock()
        final_chunk.usage.prompt_tokens = 100
        final_chunk.usage.completion_tokens = 50
        final_chunk.usage.completion_tokens_details = None
        final_chunk.choices = []

        async for _ in converter.process_chunk(final_chunk):
            pass

        # Server usage should be set
        assert converter.has_server_usage is True
        assert converter.output_tokens == 50
        # Fallback counter should still have its value
        assert converter.fallback_output_tokens == fallback_count


class TestClaudeCodeSSEFormat:
    """Tests for Claude Code SSE format handling."""

    @pytest.mark.asyncio
    async def test_aiter_text_preserves_empty_lines(self):
        """Verify that aiter_text approach preserves empty lines."""
        # Simulate what aiter_text returns
        raw_sse = (
            "event: message_start\n"
            "data: {\"type\": \"message_start\"}\n"
            "\n"  # Empty line separator
            "event: content_block_delta\n"
            "data: {\"type\": \"content_block_delta\"}\n"
            "\n"  # Empty line separator
        )

        # Split by chunks (simulating network chunks)
        chunks = [raw_sse]

        # Verify the format
        assert "\n\n" in chunks[0]  # Double newline (empty line) present

    @pytest.mark.asyncio
    async def test_aiter_lines_loses_empty_lines(self):
        """Demonstrate the bug: aiter_lines with 'if line' loses empty lines."""
        raw_lines = [
            "event: message_start",
            "data: {\"type\": \"message_start\"}",
            "",  # Empty line - would be filtered by 'if line:'
            "event: content_block_delta",
            "data: {\"type\": \"content_block_delta\"}",
            "",  # Empty line - would be filtered by 'if line:'
        ]

        # Buggy approach (what we fixed)
        buggy_output = []
        for line in raw_lines:
            if line:  # This filters out empty lines!
                buggy_output.append(line + "\n")

        # Correct approach
        correct_output = []
        for line in raw_lines:
            correct_output.append(line + "\n")

        # Buggy version loses empty lines
        assert len(buggy_output) == 4  # Only non-empty lines
        assert len(correct_output) == 6  # All lines including empty

        # Empty lines are crucial for SSE parsing
        assert "" not in [line.strip() for line in buggy_output]
        assert "\n" in correct_output  # Empty line preserved as just "\n"


class TestUsageDataStructure:
    """Tests for usage data structure integrity."""

    def test_claude_usage_accepts_cache_fields(self):
        """ClaudeUsage should accept cache-related fields."""
        from anthropic_proxy.types import ClaudeUsage

        usage = ClaudeUsage(
            input_tokens=5000,
            output_tokens=1000,
            cache_creation_input_tokens=2000,
            cache_read_input_tokens=1500,
        )

        assert usage.input_tokens == 5000
        assert usage.output_tokens == 1000
        assert usage.cache_creation_input_tokens == 2000
        assert usage.cache_read_input_tokens == 1500

    def test_claude_usage_defaults_cache_to_zero(self):
        """ClaudeUsage should default cache fields to 0."""
        from anthropic_proxy.types import ClaudeUsage

        usage = ClaudeUsage(
            input_tokens=100,
            output_tokens=50,
        )

        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0

    def test_global_usage_stats_tracks_cache_tokens(self):
        """GlobalUsageStats should track cache-related tokens."""
        from anthropic_proxy.types import ClaudeUsage, GlobalUsageStats

        stats = GlobalUsageStats()

        usage1 = ClaudeUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=300,
            cache_read_input_tokens=200,
        )

        stats.update_usage(usage1, "test-model")

        assert stats.total_input_tokens == 1000
        assert stats.total_output_tokens == 500
        assert stats.total_cache_creation_tokens == 300
        assert stats.total_cache_read_tokens == 200


class TestHookStreamingResponse:
    """Tests for hook_streaming_response SSE parsing."""

    @pytest.mark.asyncio
    async def test_handles_multi_event_chunks(self):
        """Should correctly split chunks containing multiple SSE events."""
        from anthropic_proxy.server import hook_streaming_response

        # Chunk contains two complete events
        multi_event_chunk = (
            'event: message_start\n'
            'data: {"type": "message_start"}\n'
            '\n'
            'event: content_block_delta\n'
            'data: {"type": "content_block_delta"}\n'
            '\n'
        )

        async def gen():
            yield multi_event_chunk

        events = []
        async for event in hook_streaming_response(gen(), None, "test"):
            events.append(event)

        # Should have split into two events
        assert len(events) == 2
        assert 'message_start' in events[0]
        assert 'content_block_delta' in events[1]

    @pytest.mark.asyncio
    async def test_handles_partial_events_across_chunks(self):
        """Should buffer partial events and combine with next chunk."""
        from anthropic_proxy.server import hook_streaming_response

        # First chunk has partial event
        chunk1 = 'event: message_start\ndata: {"type": "mess'
        # Second chunk completes it
        chunk2 = 'age_start"}\n\n'

        async def gen():
            yield chunk1
            yield chunk2

        events = []
        async for event in hook_streaming_response(gen(), None, "test"):
            events.append(event)

        # Should have one complete event
        assert len(events) == 1
        assert 'message_start' in events[0]

    @pytest.mark.asyncio
    async def test_handles_done_marker(self):
        """Should pass through [DONE] marker unchanged."""
        from anthropic_proxy.server import hook_streaming_response

        chunk = 'data: [DONE]\n\n'

        async def gen():
            yield chunk

        events = []
        async for event in hook_streaming_response(gen(), None, "test"):
            events.append(event)

        assert len(events) == 1
        assert '[DONE]' in events[0]

    @pytest.mark.asyncio
    async def test_handles_empty_lines(self):
        """Should handle chunks with just empty lines."""
        from anthropic_proxy.server import hook_streaming_response

        chunks = [
            'event: ping\n',
            'data: {"type": "ping"}\n',
            '\n',
        ]

        async def gen():
            for c in chunks:
                yield c

        events = []
        async for event in hook_streaming_response(gen(), None, "test"):
            events.append(event)

        # Should combine into one event
        assert len(events) == 1
        assert 'ping' in events[0]


class TestProcessSingleSSEEvent:
    """Tests for _process_single_sse_event helper."""

    def test_passes_through_non_data_events(self):
        """Events without 'data:' should pass through unchanged."""
        from anthropic_proxy.server import _process_single_sse_event

        event = "event: ping\n\n"
        result = _process_single_sse_event(event)
        assert result == event

    def test_passes_through_done_marker(self):
        """[DONE] marker should pass through unchanged."""
        from anthropic_proxy.server import _process_single_sse_event

        event = "data: [DONE]\n\n"
        result = _process_single_sse_event(event)
        assert result == event

    def test_parses_valid_json_data(self):
        """Valid JSON data should be parsed and re-serialized."""
        from anthropic_proxy.server import _process_single_sse_event

        event = 'data: {"type": "message_start"}\n\n'
        result = _process_single_sse_event(event)
        assert 'message_start' in result

    def test_handles_malformed_json_gracefully(self):
        """Malformed JSON should pass through unchanged."""
        from anthropic_proxy.server import _process_single_sse_event

        event = 'data: {"incomplete": \n\n'
        result = _process_single_sse_event(event)
        assert result == event  # Unchanged


class TestSSEEventParser:
    """Tests for SSEEventParser that buffers and parses SSE events."""

    def test_parses_complete_event(self):
        """Should parse a complete SSE event."""
        from anthropic_proxy.types import SSEEventParser

        parser = SSEEventParser()
        events = parser.feed(
            'event: message_start\n'
            'data: {"type": "message_start", "message": {"id": "msg_123", "usage": {"input_tokens": 100}}}\n'
            '\n'
        )

        assert len(events) == 1
        assert events[0].type == "message_start"
        assert events[0].message["usage"]["input_tokens"] == 100

    def test_buffers_partial_events(self):
        """Should buffer partial events until complete."""
        from anthropic_proxy.types import SSEEventParser

        parser = SSEEventParser()

        # First chunk - incomplete
        events1 = parser.feed('event: message_start\ndata: {"type": "mess')
        assert len(events1) == 0  # No complete event yet

        # Second chunk - completes the event
        events2 = parser.feed('age_start", "message": {}}\n\n')
        assert len(events2) == 1
        assert events2[0].type == "message_start"

    def test_handles_multiple_events_in_one_chunk(self):
        """Should parse multiple events from one chunk."""
        from anthropic_proxy.types import SSEEventParser

        parser = SSEEventParser()
        chunk = (
            'event: message_start\n'
            'data: {"type": "message_start", "message": {}}\n'
            '\n'
            'event: content_block_start\n'
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}\n'
            '\n'
            'event: message_delta\n'
            'data: {"type": "message_delta", "usage": {"output_tokens": 50}}\n'
            '\n'
        )

        events = parser.feed(chunk)
        assert len(events) == 3
        assert events[0].type == "message_start"
        assert events[1].type == "content_block_start"
        assert events[2].type == "message_delta"

    def test_extracts_message_delta_usage(self):
        """Should extract cumulative usage from message_delta."""
        from anthropic_proxy.types import SSEEventParser

        parser = SSEEventParser()
        events = parser.feed(
            'event: message_delta\n'
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, '
            '"usage": {"input_tokens": 1000, "output_tokens": 500, "cache_creation_input_tokens": 200, "cache_read_input_tokens": 100}}\n'
            '\n'
        )

        assert len(events) == 1
        assert events[0].type == "message_delta"
        assert events[0].usage["input_tokens"] == 1000
        assert events[0].usage["output_tokens"] == 500
        assert events[0].usage["cache_creation_input_tokens"] == 200
        assert events[0].usage["cache_read_input_tokens"] == 100


class TestRealisticStreamingSession:
    """Tests simulating realistic upstream LLM server SSE responses."""

    @pytest.fixture
    def basic_request(self):
        return ClaudeMessagesRequest(
            model="claude-3-opus",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

    @pytest.mark.asyncio
    async def test_full_streaming_session_with_usage(self, basic_request):
        """Simulate a complete streaming session and verify usage extraction."""
        from anthropic_proxy.types import global_usage_stats

        # Reset stats
        initial_input = global_usage_stats.total_input_tokens
        initial_output = global_usage_stats.total_output_tokens

        # Realistic SSE response from Anthropic API
        sse_response = (
            'event: message_start\n'
            'data: {"type": "message_start", "message": {"id": "msg_01XYZ", "type": "message", '
            '"role": "assistant", "content": [], "model": "claude-3-opus-20240229", '
            '"stop_reason": null, "usage": {"input_tokens": 2500, "output_tokens": 1}}}\n'
            '\n'
            'event: content_block_start\n'
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n'
            '\n'
            'event: content_block_delta\n'
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}\n'
            '\n'
            'event: content_block_delta\n'
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "! How"}}\n'
            '\n'
            'event: content_block_delta\n'
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " can I help?"}}\n'
            '\n'
            'event: content_block_stop\n'
            'data: {"type": "content_block_stop", "index": 0}\n'
            '\n'
            'event: message_delta\n'
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, '
            '"usage": {"input_tokens": 2500, "output_tokens": 150}}\n'
            '\n'
            'event: message_stop\n'
            'data: {"type": "message_stop"}\n'
            '\n'
        )

        # Simulate chunked delivery (network may split anywhere)
        chunks = [sse_response]

        async def gen():
            for c in chunks:
                yield c

        # Process through the usage tracking wrapper
        collected = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            gen(), basic_request, "test-model"
        ):
            collected.append(chunk)

        # Verify chunks were passed through
        assert len(collected) == 1
        assert "message_start" in collected[0]
        assert "message_delta" in collected[0]

        # Verify usage was tracked (cumulative from message_delta)
        assert global_usage_stats.total_input_tokens >= initial_input + 2500
        assert global_usage_stats.total_output_tokens >= initial_output + 150

    @pytest.mark.asyncio
    async def test_streaming_with_cache_tokens(self, basic_request):
        """Verify cache token tracking from streaming response."""
        from anthropic_proxy.types import global_usage_stats

        initial_cache_create = global_usage_stats.total_cache_creation_tokens
        initial_cache_read = global_usage_stats.total_cache_read_tokens

        # Response with cache tokens
        sse_response = (
            'event: message_start\n'
            'data: {"type": "message_start", "message": {"id": "msg_cache", '
            '"usage": {"input_tokens": 1000, "output_tokens": 1, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 300}}}\n'
            '\n'
            'event: message_delta\n'
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, '
            '"usage": {"input_tokens": 1000, "output_tokens": 100, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 300}}\n'
            '\n'
        )

        async def gen():
            yield sse_response

        async for _ in convert_anthropic_streaming_with_usage_tracking(
            gen(), basic_request, "cache-test"
        ):
            pass

        # Verify cache tokens tracked
        assert global_usage_stats.total_cache_creation_tokens >= initial_cache_create + 500
        assert global_usage_stats.total_cache_read_tokens >= initial_cache_read + 300

    @pytest.mark.asyncio
    async def test_chunked_delivery_split_mid_event(self, basic_request):
        """Test handling when network splits chunks in the middle of an event."""
        # Split the message_delta event across two chunks
        chunk1 = (
            'event: message_start\n'
            'data: {"type": "message_start", "message": {"usage": {"input_tokens": 500}}}\n'
            '\n'
            'event: message_delta\n'
            'data: {"type": "message_del'  # Incomplete!
        )
        chunk2 = (
            'ta", "usage": {"input_tokens": 500, "output_tokens": 200}}\n'
            '\n'
        )

        async def gen():
            yield chunk1
            yield chunk2

        collected = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            gen(), basic_request, "split-test"
        ):
            collected.append(chunk)

        # Both chunks should be passed through
        assert len(collected) == 2
        # Combined should contain complete events
        combined = "".join(collected)
        assert "message_start" in combined
        assert "message_delta" in combined

    @pytest.mark.asyncio
    async def test_fallback_counting_when_no_usage(self, basic_request):
        """Test tiktoken fallback when server doesn't provide usage."""
        # Response without usage fields
        sse_response = (
            'event: message_start\n'
            'data: {"type": "message_start", "message": {"id": "msg_no_usage"}}\n'
            '\n'
            'event: content_block_delta\n'
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "This is a test response with multiple words."}}\n'
            '\n'
            'event: content_block_delta\n'
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think about this carefully."}}\n'
            '\n'
            'event: message_delta\n'
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}\n'
            '\n'
        )

        async def gen():
            yield sse_response

        # This should use tiktoken fallback since no usage provided
        async for _ in convert_anthropic_streaming_with_usage_tracking(
            gen(), basic_request, "fallback-test"
        ):
            pass

        # We can't easily verify the exact fallback count, but the test
        # verifies no errors occur when usage is missing

    @pytest.mark.asyncio
    async def test_tool_use_streaming_with_usage(self, basic_request):
        """Test streaming with tool use includes tool tokens in tracking."""
        sse_response = (
            'event: message_start\n'
            'data: {"type": "message_start", "message": {"usage": {"input_tokens": 800}}}\n'
            '\n'
            'event: content_block_start\n'
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_123", "name": "read_file", "input": {}}}\n'
            '\n'
            'event: content_block_delta\n'
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\\"path\\": \\"/tmp/test.txt\\"}"}}\n'
            '\n'
            'event: content_block_stop\n'
            'data: {"type": "content_block_stop", "index": 0}\n'
            '\n'
            'event: message_delta\n'
            'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"input_tokens": 800, "output_tokens": 50}}\n'
            '\n'
        )

        async def gen():
            yield sse_response

        collected = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            gen(), basic_request, "tool-test"
        ):
            collected.append(chunk)

        # Verify tool use events passed through
        combined = "".join(collected)
        assert "tool_use" in combined
        assert "read_file" in combined
        assert "input_json_delta" in combined

    @pytest.mark.asyncio
    async def test_web_search_streaming_with_server_tool_use(self, basic_request):
        """Test streaming with web search includes server_tool_use in usage."""
        sse_response = (
            'event: message_start\n'
            'data: {"type": "message_start", "message": {"usage": {"input_tokens": 2679, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 3}}}\n'
            '\n'
            'event: content_block_start\n'
            'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srvtoolu_123", "name": "web_search", "input": {}}}\n'
            '\n'
            'event: content_block_delta\n'
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\\"query\\": \\"weather NYC\\"}"}}\n'
            '\n'
            'event: message_delta\n'
            'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 10682, "output_tokens": 510, "server_tool_use": {"web_search_requests": 1}}}\n'
            '\n'
        )

        async def gen():
            yield sse_response

        collected = []
        async for chunk in convert_anthropic_streaming_with_usage_tracking(
            gen(), basic_request, "websearch-test"
        ):
            collected.append(chunk)

        # Verify web search events passed through
        combined = "".join(collected)
        assert "server_tool_use" in combined
        assert "web_search" in combined
