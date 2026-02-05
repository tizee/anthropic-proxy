"""
Tests for Codex Responses API input sanitization.

The Codex backend requires `summary` on `function_call_output` items.
These tests verify that the sanitizer adds missing summaries to prevent
400 errors from the upstream API.
"""

import pytest

from anthropic_proxy.codex import _convert_chat_to_responses


class TestCodexFunctionCallOutputSummary:
    """function_call_output items missing summary get a default added."""

    def test_missing_summary_gets_added(self):
        """Codex rejects function_call_output without summary; sanitizer must add one."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "message", "role": "user", "content": "call the tool"},
                {
                    "type": "function_call",
                    "id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/test.txt"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents here",
                    # summary is missing
                },
            ],
            "instructions": "Be helpful.",
            "stream": True,
        }
        result = _convert_chat_to_responses(request)
        fco = [item for item in result["input"] if item.get("type") == "function_call_output"]
        assert len(fco) == 1
        assert "summary" in fco[0], "function_call_output must have summary after sanitization"

    def test_existing_summary_preserved(self):
        """Items that already have summary should not be modified."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "result data",
                    "summary": "Custom summary provided by caller",
                },
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        fco = result["input"][0]
        assert fco["summary"] == "Custom summary provided by caller"

    def test_multiple_function_call_outputs_all_sanitized(self):
        """All function_call_output items missing summary get one added."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "message", "role": "user", "content": "do two things"},
                {"type": "function_call", "id": "call_1", "name": "tool_a", "arguments": "{}"},
                {"type": "function_call", "id": "call_2", "name": "tool_b", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "output A"},
                {"type": "function_call_output", "call_id": "call_2", "output": "output B"},
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        fcos = [item for item in result["input"] if item.get("type") == "function_call_output"]
        assert len(fcos) == 2
        for fco in fcos:
            assert "summary" in fco

    def test_mixed_with_and_without_summary(self):
        """Only items missing summary are patched; others stay unchanged."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "has summary",
                    "summary": "existing",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "no summary",
                },
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        fcos = {item["call_id"]: item for item in result["input"]}
        assert fcos["call_1"]["summary"] == "existing"
        assert "summary" in fcos["call_2"]

    def test_no_function_call_outputs_unchanged(self):
        """Requests without function_call_output items pass through unchanged."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        assert result is request  # zero-copy passthrough

    def test_long_output_summary_truncated(self):
        """Summary derived from output should be reasonably truncated."""
        long_output = "x" * 1000
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": long_output,
                },
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        fco = result["input"][0]
        assert "summary" in fco
        assert len(fco["summary"]) <= 200

    def test_chat_completions_format_not_affected(self):
        """Chat Completions format conversion is not affected by input sanitization."""
        request = {
            "model": "gpt-5.2-codex",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = _convert_chat_to_responses(request)
        assert "input" in result
        assert "instructions" in result
        assert result["instructions"] == "Be helpful."
