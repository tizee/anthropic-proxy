"""
Unit tests for OpenAI request and message sanitizers.
"""

import pytest

from anthropic_proxy.utils import sanitize_openai_messages, sanitize_openai_request


class TestSanitizeOpenAIRequest:
    """Tests for sanitize_openai_request function."""

    def test_removes_thinking_field(self):
        """Test that 'thinking' field is removed from payload."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "enabled", "budget_tokens": 1000},
        }
        result = sanitize_openai_request(payload)
        assert "thinking" not in result
        assert result["model"] == "gpt-4"
        assert result["messages"] == [{"role": "user", "content": "hello"}]

    def test_removes_thinking_config_field(self):
        """Test that 'thinkingConfig' field is removed from payload."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "thinkingConfig": {"thinkingBudget": -1},
        }
        result = sanitize_openai_request(payload)
        assert "thinkingConfig" not in result
        assert result["model"] == "gpt-4"

    def test_removes_both_thinking_fields(self):
        """Test that both thinking fields are removed when present."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "enabled"},
            "thinkingConfig": {"thinkingBudget": 1000},
        }
        result = sanitize_openai_request(payload)
        assert "thinking" not in result
        assert "thinkingConfig" not in result
        assert result["model"] == "gpt-4"

    def test_preserves_other_fields(self):
        """Test that valid OpenAI fields are preserved."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True,
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "tool_choice": "auto",
        }
        result = sanitize_openai_request(payload)
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100
        assert result["stream"] is True
        assert result["tools"] == [{"type": "function", "function": {"name": "test"}}]
        assert result["tool_choice"] == "auto"

    def test_empty_payload(self):
        """Test that empty payload is handled correctly."""
        payload = {}
        result = sanitize_openai_request(payload)
        assert result == {}

    def test_no_unsupported_fields(self):
        """Test that payload without unsupported fields is unchanged."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = sanitize_openai_request(payload)
        assert result == payload

    def test_modifies_payload_in_place(self):
        """Test that the payload is modified in place (not copied)."""
        payload = {
            "model": "gpt-4",
            "thinking": {"type": "enabled"},
        }
        result = sanitize_openai_request(payload)
        assert result is payload
        assert "thinking" not in payload


class TestSanitizeOpenAIMessages:
    """Tests for sanitize_openai_messages function."""

    def test_valid_messages_unchanged(self):
        """Test that valid message sequences are returned unchanged."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = sanitize_openai_messages(messages)
        assert result == messages
        assert result is messages  # Same object returned

    def test_tool_calls_followed_by_tool_result(self):
        """Test valid tool call -> tool result sequence."""
        messages = [
            {"role": "user", "content": "call a tool"},
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
        ]
        result = sanitize_openai_messages(messages)
        assert result == messages
        assert result is messages  # Same object returned

    def test_multiple_tool_calls(self):
        """Test multiple tool calls followed by their results."""
        messages = [
            {
                "role": "assistant",
                "content": "calling tools",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "tool1", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "tool2", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result2"},
        ]
        result = sanitize_openai_messages(messages)
        assert result == messages
        assert result is messages

    def test_user_message_between_tool_calls_and_results(self):
        """Test that user messages between tool_calls and tool results are deferred."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
            },
            {"role": "user", "content": "interruption"},  # Should be deferred
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
        ]
        result = sanitize_openai_messages(messages)
        # The user message should come after the tool result
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[2]["role"] == "user"
        assert result[2]["content"] == "interruption"

    def test_orphaned_tool_result_no_tool_call(self):
        """Test that orphaned tool results (no matching tool_call) are removed."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "orphan_123", "content": "orphan result"},
        ]
        result = sanitize_openai_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_unknown_tool_call_id(self):
        """Test that tool results for unknown tool_call_ids are skipped."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "unknown_456", "content": "unknown result"},
            {"role": "tool", "tool_call_id": "call_123", "content": "correct result"},
        ]
        result = sanitize_openai_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_123"

    def test_missing_tool_results_added(self):
        """Test that missing tool results are added with placeholder content."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
            },
            # Missing tool result
        ]
        result = sanitize_openai_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_123"
        assert "interrupted" in result[1]["content"].lower() or "No result" in result[1]["content"]

    def test_multiple_missing_tool_results(self):
        """Test that multiple missing tool results are all added."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "tool1", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "tool2", "arguments": "{}"}},
                ],
            },
        ]
        result = sanitize_openai_messages(messages)
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        tool_call_ids = {msg["tool_call_id"] for msg in result[1:]}
        assert tool_call_ids == {"call_1", "call_2"}

    def test_complex_conversation_with_tools(self):
        """Test a complex conversation with multiple tool uses."""
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "first question"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "search result"},
            {"role": "assistant", "content": "answer based on search"},
            {"role": "user", "content": "second question"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "calc", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "calc result"},
        ]
        result = sanitize_openai_messages(messages)
        assert len(result) == 8
        # All messages should be in valid order
        for i, msg in enumerate(result):
            if msg["role"] == "tool":
                # Tool results should follow assistant messages with matching tool_calls
                assert i > 0
                assert result[i - 1]["role"] in ["assistant", "tool"]

    def test_empty_messages(self):
        """Test that empty messages list is handled."""
        messages = []
        result = sanitize_openai_messages(messages)
        assert result == []
        assert result is messages

    def test_system_message_deferred(self):
        """Test that system messages between tool_calls and tool results are deferred."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
            },
            {"role": "system", "content": "new instruction"},  # Should be deferred
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
        ]
        result = sanitize_openai_messages(messages)
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[2]["role"] == "system"

    def test_partial_tool_results_then_new_assistant(self):
        """Test handling when new assistant message comes before all tool results."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "tool1", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "tool2", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
            # Missing result for call_2
            {
                "role": "assistant",
                "content": "partial response",
            },
        ]
        result = sanitize_openai_messages(messages)
        # Should have placeholder for call_2, then the new assistant message
        assert len(result) == 4
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_1"
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_2"
        assert result[3]["role"] == "assistant"


class TestSanitizeOpenAIMessagesEdgeCases:
    """Edge case tests for sanitize_openai_messages."""

    def test_assistant_without_tool_calls_during_pending(self):
        """Test assistant message without tool_calls during pending state."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
            },
            {"role": "assistant", "content": "another message"},  # No tool_calls
        ]
        result = sanitize_openai_messages(messages)
        # Should add placeholder for call_1 before the second assistant
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_1"
        assert result[2]["role"] == "assistant"

    def test_only_tool_result_no_context(self):
        """Test conversation with only a tool result."""
        messages = [
            {"role": "tool", "tool_call_id": "orphan", "content": "orphan"},
        ]
        result = sanitize_openai_messages(messages)
        # Should be removed since no matching tool_call
        assert result == []

    def test_multiple_user_messages_between_tools(self):
        """Test multiple user messages between tool_calls and results."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
            },
            {"role": "user", "content": "msg1"},
            {"role": "user", "content": "msg2"},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        result = sanitize_openai_messages(messages)
        # Both user messages should be deferred after tool result
        assert len(result) == 4
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[2]["role"] == "user"
        assert result[3]["role"] == "user"
