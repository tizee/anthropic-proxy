"""
Test the full conversation cycle: Antigravity Response -> Claude Response -> Client -> Claude Request -> Antigravity Request

This tests the scenario where:
1. Antigravity returns a response (simulated with Gemini parts)
2. We convert it to Claude format (as in gemini_streaming.py)
3. Client receives it and sends it back in the next request
4. We convert it to Antigravity format again
5. Check if any nested structures are introduced
"""

import unittest

from anthropic_proxy.converters import anthropic_to_gemini_request
from anthropic_proxy.types import (
    ClaudeContentBlockText,
    ClaudeContentBlockToolUse,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockThinking,
    ClaudeMessage,
    ClaudeMessagesRequest,
)


class TestFullConversationCycle(unittest.TestCase):
    """Test the full cycle to catch nested structure bugs."""

    def _simulate_antigravity_text_response(self) -> dict:
        """Simulate an Antigravity response with text."""
        return {
            "role": "model",
            "parts": [{"text": "Hello, how can I help you today?"}]
        }

    def _simulate_antigravity_thinking_response(self) -> dict:
        """Simulate an Antigravity response with thinking."""
        return {
            "role": "model",
            "parts": [
                {"text": "Let me think about this...", "thoughtSignature": "sig-abc123"}
            ]
        }

    def _simulate_antigravity_tool_call_response(self) -> dict:
        """Simulate an Antigravity response with tool call."""
        return {
            "role": "model",
            "parts": [
                {"text": "I'll check the weather for you."},
                {"functionCall": {"name": "get_weather", "args": {"city": "SF"}, "id": "tool-1"}}
            ]
        }

    def _antigravity_part_to_claude_content_block(self, part: dict) -> dict:
        """
        Simulate how we convert Antigravity parts to Claude content blocks
        (simplified version of what happens in gemini_streaming.py)
        """
        if "text" in part and "thoughtSignature" in part:
            # Thinking part with signature
            return {
                "type": "thinking",
                "thinking": part["text"],
                "signature": part.get("thoughtSignature")
            }
        elif "text" in part and part.get("thought") is True:
            # Thinking part without signature (unsigned)
            return {
                "type": "thinking",
                "thinking": part["text"]
            }
        elif "text" in part:
            # Regular text part
            return {
                "type": "text",
                "text": part["text"]
            }
        elif "functionCall" in part:
            # Tool call part
            fc = part["functionCall"]
            return {
                "type": "tool_use",
                "id": fc.get("id", ""),
                "name": fc.get("name", ""),
                "input": fc.get("args", {})
            }
        else:
            # Unknown part type - return as-is for investigation
            return part

    def _build_claude_message_from_antigravity_response(self, ag_response: dict) -> ClaudeMessage:
        """Build a ClaudeMessage from an Antigravity response (simulating response conversion)."""
        parts = ag_response.get("parts", [])
        content_blocks = [
            self._antigravity_part_to_claude_content_block(part) for part in parts
        ]
        return ClaudeMessage(
            role="assistant",
            content=content_blocks
        )

    def test_text_response_cycle(self):
        """Test full cycle with text response."""
        # Step 1: Antigravity returns a text response
        ag_response = self._simulate_antigravity_text_response()

        # Step 2: Convert to Claude format (what client receives)
        claude_message = self._build_claude_message_from_antigravity_response(ag_response)

        # Step 3: Client sends it back in next request
        next_request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                claude_message,  # Previous assistant response
                ClaudeMessage(role="user", content="Continue"),
            ],
        )

        # Step 4: Convert back to Antigravity format
        result = anthropic_to_gemini_request(
            next_request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        # Step 5: Verify no nested structures
        contents = result["contents"]
        assistant_msg_parts = contents[0]["parts"]

        # Check text part is not nested
        self.assertEqual(assistant_msg_parts[0], {"text": "Hello, how can I help you today?"})
        self.assertIsInstance(assistant_msg_parts[0]["text"], str)

    def test_thinking_response_cycle(self):
        """Test full cycle with thinking response."""
        # Step 1: Antigravity returns a thinking response
        ag_response = self._simulate_antigravity_thinking_response()

        # Step 2: Convert to Claude format
        claude_message = self._build_claude_message_from_antigravity_response(ag_response)

        # Step 3: Client sends it back
        next_request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5-thinking",
            max_tokens=100,
            messages=[
                claude_message,
                ClaudeMessage(role="user", content="Continue"),
            ],
        )

        # Step 4: Convert back to Antigravity
        result = anthropic_to_gemini_request(
            next_request,
            "claude-sonnet-4-5-thinking",
            is_antigravity=True,
            session_id="test-session",
        )

        # Step 5: Verify no nested structures
        contents = result["contents"]
        parts = contents[0]["parts"]

        # Text should be a string, not a dict
        self.assertIsInstance(parts[0]["text"], str)
        self.assertEqual(parts[0]["text"], "Let me think about this...")

    def test_tool_call_response_cycle(self):
        """Test full cycle with tool call response."""
        # Step 1: Antigravity returns tool call
        ag_response = self._simulate_antigravity_tool_call_response()

        # Step 2: Convert to Claude format
        claude_message = self._build_claude_message_from_antigravity_response(ag_response)

        # Step 3: Client sends tool result back
        next_request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                claude_message,
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="tool-1",
                            content="72F and sunny"
                        )
                    ]
                ),
            ],
        )

        # Step 4: Convert back to Antigravity
        result = anthropic_to_gemini_request(
            next_request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        # Step 5: Verify no nested structures
        contents = result["contents"]

        # Check assistant message parts
        assistant_parts = contents[0]["parts"]
        self.assertEqual(assistant_parts[0], {"text": "I'll check the weather for you."})
        self.assertIsInstance(assistant_parts[0]["text"], str)

        # Check user message parts (tool result)
        user_parts = contents[1]["parts"]
        func_resp = user_parts[0].get("functionResponse", {})
        self.assertIn("name", func_resp)
        self.assertIn("id", func_resp)

    def test_multi_turn_conversation_cycle(self):
        """Test multiple conversation turns to catch accumulation issues."""
        messages = []

        # Simulate 5 rounds of conversation
        for i in range(5):
            # Assistant response
            ag_response = {
                "role": "model",
                "parts": [{"text": f"Response {i}"}]
            }
            claude_assistant = self._build_claude_message_from_antigravity_response(ag_response)
            messages.append(claude_assistant)

            # User message
            messages.append(ClaudeMessage(role="user", content=f"Question {i}"))

        # Convert full conversation to Antigravity
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=messages,
        )

        result = anthropic_to_gemini_request(
            request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        # Verify NO nested text structures anywhere
        contents = result["contents"]
        for msg_idx, content in enumerate(contents):
            for part_idx, part in enumerate(content.get("parts", [])):
                if "text" in part:
                    text_val = part["text"]
                    if isinstance(text_val, dict):
                        self.fail(
                            f"Found nested text at contents[{msg_idx}].parts[{part_idx}]: "
                            f"Expected str but got dict with keys {list(text_val.keys())}"
                        )

    def test_raw_dict_content_blocks_issue(self):
        """
        Test what happens if content blocks are raw dicts instead of Pydantic models.
        This simulates a potential bug where content blocks aren't properly typed.
        """
        # Simulate client sending message with raw dict content blocks
        # (bypassing Pydantic validation - this shouldn't happen but let's be safe)
        request_dict = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll help you."}
                    ]
                },
                {
                    "role": "user",
                    "content": "Thanks"
                }
            ]
        }

        # Parse through Pydantic (this should work)
        request = ClaudeMessagesRequest(**request_dict)

        # Convert to Antigravity
        result = anthropic_to_gemini_request(
            request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        # Verify no nesting
        contents = result["contents"]
        parts = contents[0]["parts"]
        self.assertEqual(parts[0], {"text": "I'll help you."})
        self.assertIsInstance(parts[0]["text"], str)


if __name__ == "__main__":
    unittest.main()
