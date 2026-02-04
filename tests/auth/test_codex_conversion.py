"""
Tests for the Codex Chat Completions -> Responses API conversion.

Covers the full chain: Anthropic request -> OpenAI Chat Completions -> Codex Responses API format.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic_proxy.codex import _convert_chat_to_responses
from anthropic_proxy.types import (
    ClaudeContentBlockText,
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeTool,
    ClaudeToolChoice,
)


def _make_anthropic_request(**overrides) -> ClaudeMessagesRequest:
    """Build a minimal ClaudeMessagesRequest with overrides."""
    defaults = dict(
        model="codex/gpt-5.2-codex",
        max_tokens=4096,
        messages=[
            ClaudeMessage(
                role="user",
                content=[ClaudeContentBlockText(type="text", text="Hello")],
            )
        ],
    )
    defaults.update(overrides)
    return ClaudeMessagesRequest(**defaults)


class TestConvertChatToResponses(unittest.TestCase):
    """Unit tests for _convert_chat_to_responses."""

    # ------------------------------------------------------------------
    # Basic conversion
    # ------------------------------------------------------------------

    def test_system_message_becomes_instructions(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            "stream": True,
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertEqual(result["instructions"], "You are helpful.")
        self.assertEqual(len(result["input"]), 1)
        self.assertEqual(result["input"][0]["role"], "user")
        self.assertNotIn("messages", result)

    def test_no_system_message_gets_default_instructions(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [
                {"role": "user", "content": "Hi"},
            ],
            "stream": True,
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertEqual(result["instructions"], "You are a helpful assistant.")
        self.assertEqual(len(result["input"]), 1)

    def test_multiple_system_messages_concatenated(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [
                {"role": "system", "content": "Rule one."},
                {"role": "system", "content": "Rule two."},
                {"role": "user", "content": "Hi"},
            ],
            "stream": True,
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertEqual(result["instructions"], "Rule one.\n\nRule two.")

    def test_model_preserved(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertEqual(result["model"], "gpt-5.2-codex")

    # ------------------------------------------------------------------
    # Passthrough for already-converted requests
    # ------------------------------------------------------------------

    def test_passthrough_when_input_present(self):
        responses_request = {
            "model": "gpt-5.2-codex",
            "instructions": "Be brief.",
            "input": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        result = _convert_chat_to_responses(responses_request)

        self.assertIs(result, responses_request)

    def test_passthrough_when_instructions_present(self):
        responses_request = {
            "model": "gpt-5.2-codex",
            "instructions": "Be brief.",
        }
        result = _convert_chat_to_responses(responses_request)

        self.assertIs(result, responses_request)

    # ------------------------------------------------------------------
    # Optional field forwarding
    # ------------------------------------------------------------------

    def test_tools_forwarded(self):
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": tools,
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertEqual(result["tools"], tools)

    def test_tool_choice_forwarded(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": "auto",
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertEqual(result["tool_choice"], "auto")

    def test_temperature_forwarded(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertEqual(result["temperature"], 0.7)

    def test_max_tokens_becomes_max_output_tokens(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 2048,
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertEqual(result["max_output_tokens"], 2048)
        self.assertNotIn("max_tokens", result)

    def test_absent_optional_fields_not_included(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertNotIn("tools", result)
        self.assertNotIn("tool_choice", result)
        self.assertNotIn("temperature", result)
        self.assertNotIn("max_output_tokens", result)

    # ------------------------------------------------------------------
    # Multi-turn conversation
    # ------------------------------------------------------------------

    def test_multi_turn_conversation(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ],
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertEqual(result["instructions"], "You are helpful.")
        self.assertEqual(len(result["input"]), 3)
        self.assertEqual(result["input"][0]["role"], "user")
        self.assertEqual(result["input"][1]["role"], "assistant")
        self.assertEqual(result["input"][2]["role"], "user")

    # ------------------------------------------------------------------
    # stream field
    # ------------------------------------------------------------------

    def test_stream_defaults_to_true(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertTrue(result["stream"])

    def test_stream_preserves_false(self):
        chat_request = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        }
        result = _convert_chat_to_responses(chat_request)

        self.assertFalse(result["stream"])


class TestAnthropicToCodexEndToEnd(unittest.TestCase):
    """End-to-end tests: Anthropic request -> to_openai_request() -> _convert_chat_to_responses()."""

    def test_simple_message(self):
        request = _make_anthropic_request()
        openai_req = request.to_openai_request()
        result = _convert_chat_to_responses(openai_req)

        self.assertIn("instructions", result)
        self.assertIn("input", result)
        self.assertNotIn("messages", result)
        # No system prompt in the original request -> default instructions
        self.assertEqual(result["instructions"], "You are a helpful assistant.")
        self.assertEqual(len(result["input"]), 1)
        self.assertEqual(result["input"][0]["role"], "user")

    def test_with_system_prompt(self):
        request = _make_anthropic_request(
            system=[{"type": "text", "text": "You are a coding assistant."}],
        )
        openai_req = request.to_openai_request()
        result = _convert_chat_to_responses(openai_req)

        self.assertEqual(result["instructions"], "You are a coding assistant.")

    def test_with_tools(self):
        tools = [
            ClaudeTool(
                name="read_file",
                description="Read a file",
                input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]
        request = _make_anthropic_request(tools=tools)
        openai_req = request.to_openai_request()
        result = _convert_chat_to_responses(openai_req)

        self.assertIn("tools", result)
        self.assertEqual(len(result["tools"]), 1)
        self.assertEqual(result["tools"][0]["function"]["name"], "read_file")

    def test_max_tokens_mapped(self):
        request = _make_anthropic_request(max_tokens=1024)
        openai_req = request.to_openai_request()
        result = _convert_chat_to_responses(openai_req)

        self.assertEqual(result["max_output_tokens"], 1024)

    def test_multi_turn_with_system(self):
        request = _make_anthropic_request(
            system=[{"type": "text", "text": "Be concise."}],
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[ClaudeContentBlockText(type="text", text="Hi")],
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[ClaudeContentBlockText(type="text", text="Hello!")],
                ),
                ClaudeMessage(
                    role="user",
                    content=[ClaudeContentBlockText(type="text", text="How are you?")],
                ),
            ],
        )
        openai_req = request.to_openai_request()
        result = _convert_chat_to_responses(openai_req)

        self.assertEqual(result["instructions"], "Be concise.")
        # 3 conversation messages (no system in input)
        self.assertEqual(len(result["input"]), 3)
        roles = [m["role"] for m in result["input"]]
        self.assertEqual(roles, ["user", "assistant", "user"])


if __name__ == "__main__":
    unittest.main()
