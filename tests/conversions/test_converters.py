"""
Tests for the converters package.

Tests the BaseConverter protocol implementations and format conversion logic.
"""

import json
import pytest
from anthropic_proxy.converters import (
    get_converter,
    get_streaming_converter,
    FORMAT_ANTHROPIC,
    FORMAT_OPENAI,
    FORMAT_GEMINI,
    AnthropicConverter,
    OpenAIConverter,
    OpenAIToAnthropicStreamingConverter,
    GeminiConverter,
    GeminiStreamingConverter,
)
from anthropic_proxy.types import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeMessage,
    ClaudeContentBlockText,
    ClaudeContentBlockToolUse,
    ClaudeContentBlockThinking,
    ClaudeUsage,
)


class TestConverterFactory:
    """Tests for converter factory functions."""

    def test_get_anthropic_converter(self):
        converter = get_converter(FORMAT_ANTHROPIC)
        assert isinstance(converter, AnthropicConverter)

    def test_get_openai_converter(self):
        converter = get_converter(FORMAT_OPENAI)
        assert isinstance(converter, OpenAIConverter)

    def test_get_gemini_converter(self):
        converter = get_converter(FORMAT_GEMINI)
        assert isinstance(converter, GeminiConverter)

    def test_get_unknown_converter_raises(self):
        with pytest.raises(ValueError, match="Unknown format type"):
            get_converter("unknown")

    def test_get_streaming_converter_anthropic(self):
        converter = get_streaming_converter(FORMAT_ANTHROPIC)
        assert converter is not None

    def test_get_streaming_converter_openai(self):
        converter = get_streaming_converter(FORMAT_OPENAI)
        assert converter is not None

    def test_get_streaming_converter_gemini(self):
        converter = get_streaming_converter(FORMAT_GEMINI)
        assert converter is not None


class TestAnthropicConverter:
    """Tests for AnthropicConverter (identity conversion)."""

    def test_request_to_anthropic(self):
        converter = AnthropicConverter()
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = converter.request_to_anthropic(payload)
        assert isinstance(request, ClaudeMessagesRequest)
        assert request.model == "claude-sonnet-4-20250514"
        assert request.max_tokens == 1024

    def test_request_from_anthropic(self):
        converter = AnthropicConverter()
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[ClaudeMessage(role="user", content="Hello")],
        )
        payload = converter.request_from_anthropic(request)
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["max_tokens"] == 1024

    def test_request_from_anthropic_with_model_override(self):
        converter = AnthropicConverter()
        request = ClaudeMessagesRequest(
            model="original-model",
            max_tokens=1024,
            messages=[ClaudeMessage(role="user", content="Hello")],
        )
        payload = converter.request_from_anthropic(request, model_id="override-model")
        assert payload["model"] == "override-model"

    def test_response_from_anthropic(self):
        converter = AnthropicConverter()
        response = ClaudeMessagesResponse(
            id="msg_123",
            model="claude-sonnet-4-20250514",
            content=[ClaudeContentBlockText(type="text", text="Hello!")],
            stop_reason="end_turn",
            usage=ClaudeUsage(input_tokens=10, output_tokens=5),
        )
        payload = converter.response_from_anthropic(response)
        assert payload["id"] == "msg_123"
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["content"][0]["text"] == "Hello!"


class TestOpenAIConverter:
    """Tests for OpenAIConverter."""

    def test_request_to_anthropic_simple(self):
        converter = OpenAIConverter()
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 1024,
        }
        request = converter.request_to_anthropic(payload)
        assert isinstance(request, ClaudeMessagesRequest)
        assert request.model == "gpt-4"
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"

    def test_request_to_anthropic_with_system(self):
        converter = OpenAIConverter()
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 1024,
        }
        request = converter.request_to_anthropic(payload)
        assert request.system == "You are helpful."
        assert len(request.messages) == 1

    def test_request_to_anthropic_with_tools(self):
        converter = OpenAIConverter()
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Get weather"}],
            "max_tokens": 1024,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
        request = converter.request_to_anthropic(payload)
        assert request.tools is not None
        assert len(request.tools) == 1
        assert request.tools[0].name == "get_weather"

    def test_request_to_anthropic_with_tool_choice_auto(self):
        converter = OpenAIConverter()
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
            "tool_choice": "auto",
        }
        request = converter.request_to_anthropic(payload)
        assert request.tool_choice is not None
        assert request.tool_choice.type == "auto"

    def test_request_to_anthropic_with_tool_choice_required(self):
        converter = OpenAIConverter()
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1024,
            "tool_choice": "required",
        }
        request = converter.request_to_anthropic(payload)
        assert request.tool_choice is not None
        assert request.tool_choice.type == "any"

    def test_response_from_anthropic_simple(self):
        converter = OpenAIConverter()
        response = ClaudeMessagesResponse(
            id="msg_123",
            model="claude-sonnet-4-20250514",
            content=[ClaudeContentBlockText(type="text", text="Hello!")],
            stop_reason="end_turn",
            usage=ClaudeUsage(input_tokens=10, output_tokens=5),
        )
        payload = converter.response_from_anthropic(response)
        assert payload["object"] == "chat.completion"
        assert payload["choices"][0]["message"]["content"] == "Hello!"
        assert payload["choices"][0]["finish_reason"] == "stop"

    def test_response_from_anthropic_with_tool_calls(self):
        converter = OpenAIConverter()
        response = ClaudeMessagesResponse(
            id="msg_123",
            model="claude-sonnet-4-20250514",
            content=[
                ClaudeContentBlockText(type="text", text="Let me check."),
                ClaudeContentBlockToolUse(
                    type="tool_use",
                    id="tool_123",
                    name="get_weather",
                    input={"city": "NYC"},
                ),
            ],
            stop_reason="tool_use",
            usage=ClaudeUsage(input_tokens=10, output_tokens=20),
        )
        payload = converter.response_from_anthropic(response)
        assert payload["choices"][0]["finish_reason"] == "tool_calls"
        assert len(payload["choices"][0]["message"]["tool_calls"]) == 1
        assert payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_response_from_anthropic_with_thinking(self):
        converter = OpenAIConverter()
        response = ClaudeMessagesResponse(
            id="msg_123",
            model="claude-sonnet-4-20250514",
            content=[
                ClaudeContentBlockThinking(type="thinking", thinking="Let me think..."),
                ClaudeContentBlockText(type="text", text="The answer is 42."),
            ],
            stop_reason="end_turn",
            usage=ClaudeUsage(input_tokens=10, output_tokens=15),
        )
        payload = converter.response_from_anthropic(response)
        assert payload["choices"][0]["message"]["reasoning_content"] == "Let me think..."
        assert payload["choices"][0]["message"]["content"] == "The answer is 42."


class TestGeminiConverter:
    """Tests for GeminiConverter."""

    def test_request_to_anthropic_simple(self):
        converter = GeminiConverter()
        payload = {
            "model": "gemini-pro",
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
            ],
            "generationConfig": {"maxOutputTokens": 1024},
        }
        request = converter.request_to_anthropic(payload)
        assert isinstance(request, ClaudeMessagesRequest)
        assert request.model == "gemini-pro"
        assert len(request.messages) == 1

    def test_request_to_anthropic_with_system(self):
        converter = GeminiConverter()
        payload = {
            "model": "gemini-pro",
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
            ],
            "systemInstruction": {"parts": [{"text": "You are helpful."}]},
            "generationConfig": {"maxOutputTokens": 1024},
        }
        request = converter.request_to_anthropic(payload)
        assert request.system == "You are helpful."

    def test_request_to_anthropic_with_function_call(self):
        converter = GeminiConverter()
        payload = {
            "model": "gemini-pro",
            "contents": [
                {"role": "user", "parts": [{"text": "Get weather"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "NYC"},
                                "id": "call_123",
                            }
                        }
                    ],
                },
            ],
            "generationConfig": {"maxOutputTokens": 1024},
        }
        request = converter.request_to_anthropic(payload)
        assert len(request.messages) == 2
        assistant_msg = request.messages[1]
        assert assistant_msg.role == "assistant"
        assert isinstance(assistant_msg.content, list)
        assert isinstance(assistant_msg.content[0], ClaudeContentBlockToolUse)
        assert assistant_msg.content[0].name == "get_weather"

    def test_response_from_anthropic_simple(self):
        converter = GeminiConverter()
        response = ClaudeMessagesResponse(
            id="msg_123",
            model="gemini-pro",
            content=[ClaudeContentBlockText(type="text", text="Hello!")],
            stop_reason="end_turn",
            usage=ClaudeUsage(input_tokens=10, output_tokens=5),
        )
        payload = converter.response_from_anthropic(response)
        assert "candidates" in payload
        assert payload["candidates"][0]["content"]["parts"][0]["text"] == "Hello!"
        assert payload["candidates"][0]["finishReason"] == "STOP"

    def test_response_from_anthropic_with_tool_use(self):
        converter = GeminiConverter()
        response = ClaudeMessagesResponse(
            id="msg_123",
            model="gemini-pro",
            content=[
                ClaudeContentBlockToolUse(
                    type="tool_use",
                    id="tool_123",
                    name="get_weather",
                    input={"city": "NYC"},
                ),
            ],
            stop_reason="tool_use",
            usage=ClaudeUsage(input_tokens=10, output_tokens=20),
        )
        payload = converter.response_from_anthropic(response)
        assert payload["candidates"][0]["finishReason"] == "TOOL_USE"
        fc = payload["candidates"][0]["content"]["parts"][0]["functionCall"]
        assert fc["name"] == "get_weather"
        assert fc["id"] == "tool_123"

    def test_response_from_anthropic_with_thinking(self):
        converter = GeminiConverter()
        response = ClaudeMessagesResponse(
            id="msg_123",
            model="gemini-pro",
            content=[
                ClaudeContentBlockThinking(
                    type="thinking", thinking="Let me think...", signature="sig_123"
                ),
                ClaudeContentBlockText(type="text", text="The answer."),
            ],
            stop_reason="end_turn",
            usage=ClaudeUsage(input_tokens=10, output_tokens=15),
        )
        payload = converter.response_from_anthropic(response)
        parts = payload["candidates"][0]["content"]["parts"]
        assert parts[0]["thought"] is True
        assert parts[0]["thoughtSignature"] == "sig_123"
        assert parts[1]["text"] == "The answer."


class TestOpenAIStreamingConverter:
    """Tests for OpenAI streaming conversion (Anthropic SSE -> OpenAI SSE)."""

    @pytest.mark.asyncio
    async def test_stream_from_anthropic_text(self):
        """Test converting Anthropic text streaming to OpenAI format."""
        converter = OpenAIToAnthropicStreamingConverter()

        # Simulate Anthropic SSE events
        async def anthropic_stream():
            yield 'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_123", "model": "claude-3"}}\n\n'
            yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n\n'
            yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}\n\n'
            yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world!"}}\n\n'
            yield 'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 10, "output_tokens": 5}}\n\n'
            yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

        chunks = []
        async for chunk in converter.stream_from_anthropic(anthropic_stream(), model="test-model"):
            if chunk.strip() and chunk != "data: [DONE]\n\n":
                chunks.append(chunk)

        # Verify we got proper OpenAI format chunks
        assert len(chunks) >= 3
        # First chunk should have role
        first_data = json.loads(chunks[0].replace("data: ", ""))
        assert first_data["object"] == "chat.completion.chunk"
        assert first_data["choices"][0]["delta"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_stream_from_anthropic_tool_use(self):
        """Test converting Anthropic tool use streaming to OpenAI format."""
        converter = OpenAIToAnthropicStreamingConverter()

        async def anthropic_stream():
            yield 'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_123", "model": "claude-3"}}\n\n'
            yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call_123", "name": "get_weather"}}\n\n'
            yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\\"city\\""}}\n\n'
            yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": ": \\"NYC\\"}"}}\n\n'
            yield 'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"input_tokens": 10, "output_tokens": 20}}\n\n'
            yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

        chunks = []
        async for chunk in converter.stream_from_anthropic(anthropic_stream(), model="test-model"):
            if chunk.strip() and chunk != "data: [DONE]\n\n":
                chunks.append(chunk)

        # Find the tool call chunk
        tool_call_found = False
        for chunk in chunks:
            data = json.loads(chunk.replace("data: ", ""))
            if "tool_calls" in data.get("choices", [{}])[0].get("delta", {}):
                tool_call_found = True
                tc = data["choices"][0]["delta"]["tool_calls"][0]
                assert tc["id"] == "call_123"
                assert tc["function"]["name"] == "get_weather"
                break

        assert tool_call_found, "Tool call chunk not found in output"

    @pytest.mark.asyncio
    async def test_stream_from_anthropic_finish_reasons(self):
        """Test that finish reasons are correctly mapped."""
        converter = OpenAIToAnthropicStreamingConverter()

        test_cases = [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
            ("stop_sequence", "stop"),
        ]

        for anthropic_reason, expected_openai_reason in test_cases:
            async def anthropic_stream():
                yield 'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_123", "model": "claude-3"}}\n\n'
                yield f'event: message_delta\ndata: {{"type": "message_delta", "delta": {{"stop_reason": "{anthropic_reason}"}}, "usage": {{"input_tokens": 10, "output_tokens": 5}}}}\n\n'
                yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

            chunks = []
            async for chunk in converter.stream_from_anthropic(anthropic_stream(), model="test"):
                if chunk.strip() and chunk != "data: [DONE]\n\n":
                    chunks.append(chunk)

            # Find the chunk with finish_reason
            finish_reason_found = False
            for chunk in chunks:
                data = json.loads(chunk.replace("data: ", ""))
                if data.get("choices", [{}])[0].get("finish_reason"):
                    assert data["choices"][0]["finish_reason"] == expected_openai_reason
                    finish_reason_found = True

            assert finish_reason_found, f"Finish reason not found for {anthropic_reason}"


class TestGeminiStreamingConverter:
    """Tests for Gemini streaming conversion (Anthropic SSE -> Gemini NDJSON)."""

    @pytest.mark.asyncio
    async def test_stream_from_anthropic_text(self):
        """Test converting Anthropic text streaming to Gemini format."""
        converter = GeminiStreamingConverter()

        async def anthropic_stream():
            yield 'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_123", "model": "claude-3"}}\n\n'
            yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n\n'
            yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}\n\n'
            yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world!"}}\n\n'
            yield 'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 10, "output_tokens": 5}}\n\n'
            yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

        chunks = []
        async for chunk in converter.stream_from_anthropic(anthropic_stream(), model="test-model"):
            if chunk.strip():
                chunks.append(chunk)

        # Verify we got proper Gemini format chunks
        assert len(chunks) >= 2
        # First text chunk
        first_data = json.loads(chunks[0])
        assert "candidates" in first_data
        assert first_data["candidates"][0]["content"]["parts"][0]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_stream_from_anthropic_tool_use(self):
        """Test converting Anthropic tool use streaming to Gemini format."""
        converter = GeminiStreamingConverter()

        async def anthropic_stream():
            yield 'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_123", "model": "claude-3"}}\n\n'
            yield 'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call_123", "name": "get_weather"}}\n\n'
            yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\\"city\\""}}\n\n'
            yield 'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": ": \\"NYC\\"}"}}\n\n'
            yield 'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n'
            yield 'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"input_tokens": 10, "output_tokens": 20}}\n\n'
            yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

        chunks = []
        async for chunk in converter.stream_from_anthropic(anthropic_stream(), model="test-model"):
            if chunk.strip():
                chunks.append(chunk)

        # Find the function call chunk
        fc_found = False
        for chunk in chunks:
            data = json.loads(chunk)
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            for part in parts:
                if "functionCall" in part:
                    fc_found = True
                    assert part["functionCall"]["name"] == "get_weather"
                    assert part["functionCall"]["id"] == "call_123"
                    break

        assert fc_found, "Function call chunk not found in output"

    @pytest.mark.asyncio
    async def test_stream_from_anthropic_finish_reasons(self):
        """Test that finish reasons are correctly mapped to Gemini format."""
        converter = GeminiStreamingConverter()

        test_cases = [
            ("end_turn", "STOP"),
            ("max_tokens", "MAX_TOKENS"),
            ("tool_use", "TOOL_USE"),
            ("refusal", "SAFETY"),
        ]

        for anthropic_reason, expected_gemini_reason in test_cases:
            async def anthropic_stream():
                yield 'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_123", "model": "claude-3"}}\n\n'
                yield f'event: message_delta\ndata: {{"type": "message_delta", "delta": {{"stop_reason": "{anthropic_reason}"}}, "usage": {{"input_tokens": 10, "output_tokens": 5}}}}\n\n'
                yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

            chunks = []
            async for chunk in converter.stream_from_anthropic(anthropic_stream(), model="test"):
                if chunk.strip():
                    chunks.append(chunk)

            # Find the chunk with finishReason
            finish_found = False
            for chunk in chunks:
                data = json.loads(chunk)
                finish_reason = data.get("candidates", [{}])[0].get("finishReason")
                if finish_reason:
                    assert finish_reason == expected_gemini_reason
                    finish_found = True

            assert finish_found, f"Finish reason not found for {anthropic_reason}"
