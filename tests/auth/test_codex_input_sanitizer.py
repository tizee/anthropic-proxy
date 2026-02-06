"""
Tests for Codex Responses API input sanitization.

The Codex backend requires ``summary`` on ``reasoning`` items passed back
as input for multi-turn conversations. When the agent sends reasoning items
(from a previous response output) without ``summary``, the backend returns
400 "Missing required parameter: 'input[N].summary'".

The sanitizer must add an empty ``summary`` array to reasoning items that
lack one, while leaving all other item types untouched.
"""

import pytest

from anthropic_proxy.codex import _convert_chat_to_responses


class TestCodexReasoningItemSummary:
    """reasoning items missing summary get an empty array added."""

    def test_reasoning_missing_summary_gets_empty_array(self):
        """Codex rejects reasoning items without summary; sanitizer must add one."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {
                    "type": "reasoning",
                    "id": "rs_001",
                    "content": [{"type": "reasoning_encrypted", "data": "abc"}],
                    # summary is missing -- Codex backend will reject this
                },
            ],
            "instructions": "Be helpful.",
            "stream": True,
        }
        result = _convert_chat_to_responses(request)
        reasoning = [i for i in result["input"] if i.get("type") == "reasoning"]
        assert len(reasoning) == 1
        assert "summary" in reasoning[0], "reasoning item must have summary after sanitization"
        assert reasoning[0]["summary"] == []

    def test_reasoning_existing_summary_preserved(self):
        """Reasoning items that already have summary should not be modified."""
        summaries = [{"type": "summary_text", "text": "The model reasoned about X."}]
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_001",
                    "content": [],
                    "summary": summaries,
                },
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        reasoning = result["input"][0]
        assert reasoning["summary"] is summaries

    def test_multiple_reasoning_items_all_sanitized(self):
        """All reasoning items missing summary get one added."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "message", "role": "user", "content": "step 1"},
                {"type": "reasoning", "id": "rs_001", "content": []},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
                {"type": "message", "role": "user", "content": "step 2"},
                {"type": "reasoning", "id": "rs_002", "content": []},
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        reasoning_items = [i for i in result["input"] if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 2
        for item in reasoning_items:
            assert "summary" in item
            assert item["summary"] == []

    def test_mixed_reasoning_with_and_without_summary(self):
        """Only reasoning items missing summary are patched; others unchanged."""
        existing_summary = [{"type": "summary_text", "text": "Existing."}]
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rs_001",
                    "content": [],
                    "summary": existing_summary,
                },
                {
                    "type": "reasoning",
                    "id": "rs_002",
                    "content": [{"type": "reasoning_encrypted", "data": "xyz"}],
                    # missing summary
                },
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        items = {i["id"]: i for i in result["input"]}
        assert items["rs_001"]["summary"] is existing_summary
        assert items["rs_002"]["summary"] == []

    def test_no_reasoning_items_unchanged(self):
        """Requests without reasoning items pass through unchanged."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "message", "role": "user", "content": "hello"},
                {"type": "function_call", "id": "call_1", "name": "test", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "result"},
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        assert result is request  # zero-copy passthrough

    def test_function_call_output_not_modified(self):
        """function_call_output items must NOT get summary added (they don't accept it)."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "function_call_output", "call_id": "call_1", "output": "result"},
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        fco = result["input"][0]
        assert "summary" not in fco, "function_call_output must NOT have summary"

    def test_spurious_summary_stripped_from_function_call_output(self):
        """Agent may attach summary to function_call_output; it must be stripped."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "file contents",
                    "summary": "Read a file",  # invalid for this type
                },
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        fco = result["input"][0]
        assert "summary" not in fco

    def test_spurious_summary_stripped_from_message(self):
        """summary on message items must be stripped."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                    "summary": "greeted user",  # invalid for this type
                },
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        msg = result["input"][0]
        assert "summary" not in msg

    def test_spurious_summary_stripped_from_function_call(self):
        """summary on function_call items must be stripped."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/a.txt"}',
                    "summary": "called read_file",  # invalid for this type
                },
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        fc = result["input"][0]
        assert "summary" not in fc

    def test_strip_and_add_summary_in_same_pass(self):
        """Both fixes in one request: strip from non-reasoning, add to reasoning."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "result",
                    "summary": "bogus",  # must be stripped
                },
                {
                    "type": "reasoning",
                    "id": "rs_001",
                    "content": [],
                    # summary missing -- must be added
                },
            ],
            "instructions": "Be helpful.",
        }
        result = _convert_chat_to_responses(request)
        fco = result["input"][0]
        reasoning = result["input"][1]
        assert "summary" not in fco
        assert reasoning["summary"] == []

    def test_chat_completions_format_not_affected(self):
        """Chat Completions format conversion path is unaffected by sanitization."""
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

    def test_realistic_multi_turn_with_reasoning_and_tools(self):
        """Realistic multi-turn: reasoning + function_call + function_call_output."""
        request = {
            "model": "gpt-5.2-codex",
            "input": [
                {"type": "message", "role": "user", "content": "read the file"},
                # Previous model output items fed back as input:
                {
                    "type": "reasoning",
                    "id": "rs_001",
                    "content": [{"type": "reasoning_encrypted", "data": "enc_data"}],
                    # summary missing -- this caused the 400 error
                },
                {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_abc",
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/test.txt"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": "file contents here",
                },
                {"type": "message", "role": "user", "content": "now edit it"},
            ],
            "instructions": "You are a coding assistant.",
        }
        result = _convert_chat_to_responses(request)

        # Reasoning item gets summary added
        reasoning = [i for i in result["input"] if i.get("type") == "reasoning"]
        assert len(reasoning) == 1
        assert reasoning[0]["summary"] == []

        # function_call_output is NOT modified
        fco = [i for i in result["input"] if i.get("type") == "function_call_output"]
        assert len(fco) == 1
        assert "summary" not in fco[0]

        # Other items preserved
        assert len(result["input"]) == 5
