"""
Test the Claude Messages to Antigravity conversion pipeline.

Tests the full flow:
Claude MessagesRequest -> anthropic_to_gemini_request (is_antigravity=True) -> Gemini contents
"""

import unittest

from anthropic_proxy.converters import anthropic_to_gemini_request, ensure_tool_ids, _clean_malformed_parts
from anthropic_proxy.types import (
    ClaudeContentBlockText,
    ClaudeContentBlockThinking,
    ClaudeContentBlockToolUse,
    ClaudeMessage,
    ClaudeMessagesRequest,
)


class TestAntigravityConversionPipeline(unittest.TestCase):
    """Test the full conversion pipeline from Claude Messages to Antigravity format."""

    def test_text_block_conversion(self):
        """Test that text blocks are converted correctly."""
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Hello")
            ],
        )

        result = anthropic_to_gemini_request(
            request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        contents = result["contents"]
        self.assertEqual(len(contents), 1)

        parts = contents[0]["parts"]
        self.assertEqual(len(parts), 1)
        # Should be {"text": "Hello"}, not {"text": {"text": "Hello"}}
        self.assertEqual(parts[0], {"text": "Hello"})
        # Verify no nested structure
        self.assertNotIn("text", parts[0].get("text", {}))

    def test_thinking_block_conversion(self):
        """Test that thinking blocks are converted correctly."""
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5-thinking",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockThinking(
                            type="thinking",
                            thinking="Let me think about this...",
                            signature="sig-123"
                        )
                    ]
                )
            ],
        )

        result = anthropic_to_gemini_request(
            request,
            "claude-sonnet-4-5-thinking",
            is_antigravity=True,
            session_id="test-session",
        )

        contents = result["contents"]
        parts = contents[0]["parts"]

        # Should be {"text": "...", "thoughtSignature": "..."}
        self.assertEqual(parts[0]["text"], "Let me think about this...")
        self.assertEqual(parts[0]["thoughtSignature"], "sig-123")
        # Verify no nested structure
        self.assertNotIn("text", parts[0].get("text", {}))
        self.assertNotIn("thinking", parts[0])

    def test_tool_use_conversion(self):
        """Test that tool_use blocks are converted correctly."""
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu-123",
                            name="get_weather",
                            input={"city": "SF"}
                        )
                    ]
                )
            ],
        )

        result = anthropic_to_gemini_request(
            request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        contents = result["contents"]
        parts = contents[0]["parts"]

        # Should be {"functionCall": {"name": ..., "args": ..., "id": ...}}
        self.assertIn("functionCall", parts[0])
        function_call = parts[0]["functionCall"]
        self.assertEqual(function_call["name"], "get_weather")
        self.assertEqual(function_call["args"], {"city": "SF"})
        self.assertEqual(function_call["id"], "toolu-123")

    def test_mixed_content_blocks(self):
        """Test that mixed content blocks are converted correctly."""
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(type="text", text="I'll help you."),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu-456",
                            name="search",
                            input={"query": "test"}
                        )
                    ]
                )
            ],
        )

        result = anthropic_to_gemini_request(
            request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        contents = result["contents"]
        parts = contents[0]["parts"]

        self.assertEqual(len(parts), 2)
        # First part should be text
        self.assertEqual(parts[0], {"text": "I'll help you."})
        # Second part should be functionCall
        self.assertIn("functionCall", parts[1])

    def test_ensure_tool_ids_preserves_structure(self):
        """Test that ensure_tool_ids doesn't create nested structures."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {"text": "Hello"},
                    {"functionCall": {"name": "test", "args": {}, "id": "call-1"}}
                ]
            }
        ]

        result = ensure_tool_ids(contents)

        # Verify structure is preserved
        parts = result[0]["parts"]
        self.assertEqual(parts[0], {"text": "Hello"})
        self.assertIn("functionCall", parts[1])
        # Check no nested text.text
        self.assertNotIn("text", parts[0].get("text", {}))

    def test_clean_malformed_parts_preserves_valid_structure(self):
        """Test that _clean_malformed_parts preserves valid parts."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {"text": "Valid text"},
                    {"functionCall": {"name": "test", "args": {}}}
                ]
            }
        ]

        result = _clean_malformed_parts(contents)

        # Verify valid parts are preserved
        parts = result[0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], {"text": "Valid text"})
        self.assertIn("functionCall", parts[1])

    def test_clean_malformed_parts_removes_nested_thinking(self):
        """Test that _clean_malformed_parts removes nested thinking parts."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {"text": "Valid text"},
                    {"thinking": {"thinking": "Bad nested thinking", "thoughtSignature": "sig"}}
                ]
            }
        ]

        result = _clean_malformed_parts(contents)

        # Verify nested thinking is removed
        parts = result[0]["parts"]
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0], {"text": "Valid text"})

    def test_no_double_wrapping_in_conversion(self):
        """Test that parts are not double-wrapped during conversion."""
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Test message")
            ],
        )

        result = anthropic_to_gemini_request(
            request,
            "claude-sonnet-4-5",
            is_antigravity=True,
        )

        contents = result["contents"]
        parts = contents[0]["parts"]

        # Direct check: part should be dict with "text" key pointing to string
        part = parts[0]
        self.assertIsInstance(part, dict)
        self.assertIn("text", part)
        self.assertIsInstance(part["text"], str)
        self.assertEqual(part["text"], "Test message")

        # Ensure no nested {"text": {"text": "..."}} structure
        # If part["text"] were a dict, this would fail
        self.assertNotIsInstance(part.get("text"), dict)

    def test_clean_malformed_parts_fixes_nested_text(self):
        """Test that _clean_malformed_parts fixes nested text structures."""
        from anthropic_proxy.converters import _clean_malformed_parts

        contents = [
            {
                "role": "model",
                "parts": [
                    {"text": "Valid text"},
                    {"text": {"text": "Nested text"}},  # Malformed
                ]
            }
        ]

        result = _clean_malformed_parts(contents)

        # Verify nested text is fixed
        parts = result[0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], {"text": "Valid text"})
        # The nested structure should be flattened
        self.assertEqual(parts[1], {"text": "Nested text"})
        # Ensure no dict in text field
        self.assertIsInstance(parts[1]["text"], str)


if __name__ == "__main__":
    unittest.main()
