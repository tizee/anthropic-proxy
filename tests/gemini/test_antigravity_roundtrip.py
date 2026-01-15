"""
Test the round-trip conversion: Antigravity format -> Claude format -> Antigravity format

This tests the scenario where:
1. Antigravity returns a response
2. We convert it to Claude format for the client
3. Client sends it back in the next request
4. We convert it back to Antigravity format

The bug is that somewhere in this cycle, nested structures like {"text": {"text": "..."}} are created.
"""

import unittest

from anthropic_proxy.converters import anthropic_to_gemini_request
from anthropic_proxy.types import (
    ClaudeContentBlockText,
    ClaudeContentBlockToolUse,
    ClaudeContentBlockToolResult,
    ClaudeMessage,
    ClaudeMessagesRequest,
)


class TestAntigravityRoundTrip(unittest.TestCase):
    """Test round-trip conversion to catch nested structure bugs."""

    def test_round_trip_simple_text(self):
        """Test that simple text blocks survive round-trip without nesting."""
        # First response from Antigravity (simulated as Claude format)
        first_response = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[ClaudeContentBlockText(type="text", text="Hello, how can I help?")]
                )
            ],
        )

        # Client sends this back in the next request (with their message)
        next_request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                # Previous assistant response
                ClaudeMessage(
                    role="assistant",
                    content=[ClaudeContentBlockText(type="text", text="Hello, how can I help?")]
                ),
                # New user message
                ClaudeMessage(role="user", content="What's the weather?"),
            ],
        )

        # Convert to Antigravity format
        result = anthropic_to_gemini_request(
            next_request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        contents = result["contents"]
        # Check that text parts are not nested
        for content in contents:
            for part in content.get("parts", []):
                if "text" in part:
                    # part["text"] should be a string, not a dict
                    self.assertIsInstance(part["text"], str,
                        f"Found nested text structure: {part}")
                    # Double-check no dict
                    self.assertNotIsInstance(part.get("text"), dict,
                        f"Text field should not be a dict: {part}")

    def test_round_trip_with_tool_call(self):
        """Test that tool calls survive round-trip without nesting."""
        # Previous conversation with tool call
        next_request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(type="text", text="I'll check the weather."),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu-123",
                            name="get_weather",
                            input={"city": "SF"}
                        )
                    ]
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="toolu-123",
                            content="72F and sunny"
                        )
                    ]
                ),
            ],
        )

        # Convert to Antigravity format
        result = anthropic_to_gemini_request(
            next_request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        contents = result["contents"]

        # Verify no nested text structures
        for content in contents:
            for part in content.get("parts", []):
                if "text" in part:
                    self.assertIsInstance(part["text"], str,
                        f"Found nested text in tool result: {part}")

    def test_round_trip_with_content_list_tool_result(self):
        """Test tool result with content list (not just string)."""
        # Tool result can be a list of content blocks
        next_request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu-456",
                            name="search",
                            input={"query": "test"}
                        )
                    ]
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="toolu-456",
                            content=[
                                {"type": "text", "text": "Result 1"},
                                {"type": "text", "text": "Result 2"}
                            ]
                        )
                    ]
                ),
            ],
        )

        # Convert to Antigravity format
        result = anthropic_to_gemini_request(
            next_request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        contents = result["contents"]
        parts = contents[1]["parts"]  # User message parts

        # The functionResponse should have the correct structure
        function_response = parts[0].get("functionResponse", {})
        response_content = function_response.get("response", {}).get("content")

        # Response content should be a list, not causing nested text.text
        self.assertIsNotNone(response_content)
        self.assertIsInstance(response_content, list)

    def test_long_conversation_no_nesting(self):
        """Test a long conversation to catch any nesting issues."""
        messages = []
        for i in range(20):  # Create 20 message pairs (40 total)
            messages.append(ClaudeMessage(
                role="assistant",
                content=[ClaudeContentBlockText(type="text", text=f"Response {i}")]
            ))
            messages.append(ClaudeMessage(
                role="user",
                content=f"Question {i}"
            ))

        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=messages,
        )

        # Convert to Antigravity format
        result = anthropic_to_gemini_request(
            request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        contents = result["contents"]

        # Verify NO part has {"text": {"text": "..."}} structure
        for idx, content in enumerate(contents):
            for part_idx, part in enumerate(content.get("parts", [])):
                if "text" in part:
                    text_value = part["text"]
                    if isinstance(text_value, dict):
                        self.fail(
                            f"Found nested text structure at contents[{idx}].parts[{part_idx}]: "
                            f"{part}"
                        )


if __name__ == "__main__":
    unittest.main()
