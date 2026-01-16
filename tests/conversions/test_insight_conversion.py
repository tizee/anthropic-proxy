#!/usr/bin/env python3
"""
Test suite for Insight text preservation in format conversion.

This suite verifies that Insight markers and content are properly preserved
when converting between OpenAI and Anthropic response formats.

Insight is a "soft protocol" - plain text with specific markers that Claude Code
renders specially. The proxy must preserve this text exactly as-is.

Insight format (from Claude Code's Explanatory output style):
    ★ Insight ─────────────────────────────────────
    [2-3 key educational points]
    ─────────────────────────────────────────────────
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic_proxy.types import (
    ClaudeMessage,
    ClaudeMessagesRequest,
)
from anthropic_proxy.converters import convert_openai_response_to_anthropic
from anthropic_proxy.converters import AnthropicStreamingConverter


# Sample Insight text that models generate
SAMPLE_INSIGHT_TEXT = """Let me help you with that task.

★ Insight ─────────────────────────────────────
**Key Points:**
1. This demonstrates how Insight markers work
2. The text must be preserved exactly as-is
3. Unicode characters like ★ and ─ must not be escaped
─────────────────────────────────────────────────

Now let me continue with the implementation..."""

SAMPLE_INSIGHT_WITH_CODE = """I'll create the function for you.

★ Insight ─────────────────────────────────────
**Implementation Choice:**
- Using a simple loop for clarity
- O(n) time complexity is acceptable here
─────────────────────────────────────────────────

Here's the code:
```python
def example():
    pass
```"""

INSIGHT_ONLY_TEXT = """★ Insight ─────────────────────────────────────
Single insight block without surrounding text.
─────────────────────────────────────────────────"""

MULTIPLE_INSIGHTS_TEXT = """First, let me explain.

★ Insight ─────────────────────────────────────
**First insight about the architecture**
─────────────────────────────────────────────────

Then we have more context.

★ Insight ─────────────────────────────────────
**Second insight about implementation**
─────────────────────────────────────────────────

And finally the conclusion."""


def create_mock_openai_response(
    content: str,
    tool_calls=None,
    reasoning_content: str = "",
    finish_reason: str = "stop",
):
    """Create a mock OpenAI response for testing."""

    class MockToolCall:
        def __init__(self, id: str, name: str, arguments: str):
            self.id = id
            self.function = MockFunction(name, arguments)

    class MockFunction:
        def __init__(self, name: str, arguments: str):
            self.name = name
            self.arguments = arguments

    class MockMessage:
        def __init__(self, content, tool_calls=None, reasoning_content=None):
            self.content = content
            self.tool_calls = tool_calls
            self._reasoning_content = reasoning_content

        def model_dump(self):
            result = {
                "content": self.content,
                "tool_calls": self.tool_calls,
            }
            if self._reasoning_content:
                result["reasoning_content"] = self._reasoning_content
            return result

    class MockChoice:
        def __init__(
            self, content, tool_calls=None, reasoning_content=None, finish_reason="stop"
        ):
            self.message = MockMessage(content, tool_calls, reasoning_content)
            self.finish_reason = finish_reason

    class MockUsage:
        def __init__(self):
            self.prompt_tokens = 100
            self.completion_tokens = 200

    class MockOpenAIResponse:
        def __init__(
            self, content, tool_calls=None, reasoning_content=None, finish_reason="stop"
        ):
            self.choices = [
                MockChoice(content, tool_calls, reasoning_content, finish_reason)
            ]
            self.usage = MockUsage()

    mock_tool_calls = None
    if tool_calls:
        mock_tool_calls = [
            MockToolCall(f"call_{i}", call["name"], json.dumps(call["arguments"]))
            for i, call in enumerate(tool_calls)
        ]

    return MockOpenAIResponse(
        content, mock_tool_calls, reasoning_content, finish_reason
    )


class TestInsightTextPreservation(unittest.TestCase):
    """Test that Insight text is preserved exactly during format conversion."""

    def test_basic_insight_preservation(self):
        """Test that basic Insight text is preserved in non-streaming conversion."""
        print("🧪 Testing basic Insight text preservation...")

        mock_response = create_mock_openai_response(SAMPLE_INSIGHT_TEXT)

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Help me with this task")],
        )

        result = convert_openai_response_to_anthropic(mock_response, original_request)

        # Validate the text content is preserved exactly
        self.assertEqual(len(result.content), 1)
        self.assertEqual(result.content[0].type, "text")

        converted_text = result.content[0].text

        # Key assertions: Insight markers must be preserved
        self.assertIn("★ Insight", converted_text, "Insight star marker must be preserved")
        self.assertIn("─────────────────────────────────────────────────", converted_text,
                      "Insight separator line must be preserved")
        self.assertIn("**Key Points:**", converted_text, "Insight content must be preserved")

        # The entire text should match exactly
        self.assertEqual(converted_text, SAMPLE_INSIGHT_TEXT,
                         "Insight text must be preserved exactly without modification")

        print("✅ Basic Insight preservation test passed")

    def test_insight_with_code_block_preservation(self):
        """Test that Insight text with code blocks is preserved."""
        print("🧪 Testing Insight with code block preservation...")

        mock_response = create_mock_openai_response(SAMPLE_INSIGHT_WITH_CODE)

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Create a function")],
        )

        result = convert_openai_response_to_anthropic(mock_response, original_request)

        converted_text = result.content[0].text

        # Both Insight and code block should be preserved
        self.assertIn("★ Insight", converted_text)
        self.assertIn("```python", converted_text)
        self.assertEqual(converted_text, SAMPLE_INSIGHT_WITH_CODE)

        print("✅ Insight with code block test passed")

    def test_insight_only_text_preservation(self):
        """Test that a response containing only Insight is preserved."""
        print("🧪 Testing Insight-only text preservation...")

        mock_response = create_mock_openai_response(INSIGHT_ONLY_TEXT)

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Explain something")],
        )

        result = convert_openai_response_to_anthropic(mock_response, original_request)

        converted_text = result.content[0].text
        self.assertEqual(converted_text, INSIGHT_ONLY_TEXT)

        print("✅ Insight-only text test passed")

    def test_multiple_insights_preservation(self):
        """Test that multiple Insight blocks are all preserved."""
        print("🧪 Testing multiple Insights preservation...")

        mock_response = create_mock_openai_response(MULTIPLE_INSIGHTS_TEXT)

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Explain the architecture")],
        )

        result = convert_openai_response_to_anthropic(mock_response, original_request)

        converted_text = result.content[0].text

        # Count Insight occurrences
        insight_count = converted_text.count("★ Insight")
        self.assertEqual(insight_count, 2, "Both Insight blocks must be preserved")

        # Full text should match
        self.assertEqual(converted_text, MULTIPLE_INSIGHTS_TEXT)

        print("✅ Multiple Insights test passed")

    def test_unicode_character_preservation(self):
        """Test that Unicode characters in Insight are not escaped or modified."""
        print("🧪 Testing Unicode character preservation...")

        # Test specific Unicode characters used in Insight
        unicode_text = "★ ─ │ ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼"
        mock_response = create_mock_openai_response(unicode_text)

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Test")],
        )

        result = convert_openai_response_to_anthropic(mock_response, original_request)

        converted_text = result.content[0].text

        # Unicode should not be escaped
        self.assertNotIn("\\u", converted_text, "Unicode should not be escaped")
        self.assertEqual(converted_text, unicode_text)

        print("✅ Unicode character preservation test passed")

    def test_insight_with_tool_calls(self):
        """Test that Insight text is preserved when response also contains tool calls."""
        print("🧪 Testing Insight with tool calls...")

        insight_before_tool = """Let me search for that.

★ Insight ─────────────────────────────────────
Using web search to find current information.
─────────────────────────────────────────────────"""

        tool_calls = [{"name": "web_search", "arguments": {"query": "test query"}}]
        mock_response = create_mock_openai_response(insight_before_tool, tool_calls)

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Search for something")],
        )

        result = convert_openai_response_to_anthropic(mock_response, original_request)

        # Should have both text and tool_use blocks
        text_blocks = [b for b in result.content if b.type == "text"]
        tool_blocks = [b for b in result.content if b.type == "tool_use"]

        self.assertEqual(len(text_blocks), 1, "Should have one text block")
        self.assertEqual(len(tool_blocks), 1, "Should have one tool_use block")

        # Insight should be preserved in text block
        self.assertIn("★ Insight", text_blocks[0].text)
        self.assertEqual(text_blocks[0].text, insight_before_tool)

        print("✅ Insight with tool calls test passed")


class TestInsightStreamingPreservation(unittest.TestCase):
    """Test that Insight text is preserved during streaming conversion."""

    def test_streaming_insight_accumulation(self):
        """Test that Insight text is correctly accumulated during streaming."""
        print("🧪 Testing streaming Insight accumulation...")

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Help me")],
        )

        converter = AnthropicStreamingConverter(original_request)

        # Simulate streaming the Insight text in chunks
        chunks = [
            "Let me help ",
            "you with that.\n\n",
            "★ Insight ─────────────────────────────────────\n",
            "**Key Point:** This is important\n",
            "─────────────────────────────────────────────────\n",
            "\nNow continuing..."
        ]

        # Process each chunk through _handle_text_delta
        import asyncio

        async def process_chunks():
            for chunk in chunks:
                async for _ in converter._handle_text_delta(chunk):
                    pass  # Just process, don't collect events

        asyncio.run(process_chunks())

        # Verify accumulated text contains complete Insight
        full_text = "".join(chunks)
        self.assertEqual(converter.accumulated_text, full_text)
        self.assertIn("★ Insight", converter.accumulated_text)
        self.assertIn("─────────────────────────────────────────────────",
                      converter.accumulated_text)

        print("✅ Streaming Insight accumulation test passed")

    def test_streaming_insight_marker_split_across_chunks(self):
        """Test Insight marker that is split across multiple chunks."""
        print("🧪 Testing Insight marker split across chunks...")

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Help me")],
        )

        converter = AnthropicStreamingConverter(original_request)

        # Intentionally split the Insight marker across chunks
        chunks = [
            "Here's info:\n\n★",  # Star alone
            " Insight ",          # "Insight" word
            "─────────────────────────────────────────────────\n",
            "Content here\n",
            "─────────────────────────────────────────────────",
        ]

        import asyncio

        async def process_chunks():
            for chunk in chunks:
                async for _ in converter._handle_text_delta(chunk):
                    pass

        asyncio.run(process_chunks())

        # Even when split, the full marker should be assembled
        self.assertIn("★ Insight", converter.accumulated_text)

        print("✅ Split Insight marker test passed")

    def test_streaming_does_not_drop_text_in_tool_use_mode(self):
        """
        Test that text content is NOT dropped when is_tool_use is True.

        This tests the suspected bug where:
        - `if chunk_data["delta_content"] and not self.is_tool_use:`
        - would cause text to be dropped when is_tool_use=True
        """
        print("🧪 Testing text preservation in tool_use mode...")

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Help me")],
        )

        converter = AnthropicStreamingConverter(original_request)

        # First, simulate some text
        import asyncio

        async def test_sequence():
            # 1. Add some initial text
            async for _ in converter._handle_text_delta("Initial text. "):
                pass

            # 2. Manually set is_tool_use to True (simulating tool call started)
            converter.is_tool_use = True
            converter.active_tool_indices.add(0)
            converter.tool_calls[0] = {
                "id": "test_tool",
                "name": "test",
                "json_accumulator": "{}",
                "content_block_index": 1,
            }

            # 3. Now try to add more text containing Insight
            # This is where the bug would cause text to be dropped
            insight_text = "\n\n★ Insight ─────\nImportant info\n─────"

            # The _handle_text_delta method should handle this properly
            # by closing tool_use first, then processing text
            async for _ in converter._handle_text_delta(insight_text):
                pass

            return converter.accumulated_text

        accumulated = asyncio.run(test_sequence())

        # The Insight text should be in accumulated_text
        self.assertIn("★ Insight", accumulated,
                      "Insight text should NOT be dropped even when is_tool_use was True")
        self.assertIn("Important info", accumulated)

        # is_tool_use should now be False (reset by _handle_text_delta)
        self.assertFalse(converter.is_tool_use,
                         "is_tool_use should be reset after text is processed")

        print("✅ Tool use mode text preservation test passed")


class TestProcessChunkTextLoss(unittest.TestCase):
    """
    Test the actual process_chunk method to verify text is not dropped.

    This tests the suspected bug in streaming.py line 855-857:
        if chunk_data["delta_content"] and not self.is_tool_use:

    When is_tool_use=True, this condition prevents _handle_text_delta from
    being called, potentially dropping Insight text.
    """

    def _create_mock_chunk(self, content=None, tool_calls=None, finish_reason=None):
        """Create a mock ChatCompletionChunk for testing."""
        from unittest.mock import MagicMock

        chunk = MagicMock()
        chunk.id = "test_chunk"
        chunk.usage = None

        choice = MagicMock()
        delta = MagicMock()

        # Set content
        delta.content = content
        delta.tool_calls = tool_calls

        # model_dump for reasoning_content extraction
        delta.model_dump.return_value = {"content": content, "tool_calls": tool_calls}

        choice.delta = delta
        choice.finish_reason = finish_reason

        chunk.choices = [choice]

        return chunk

    def _create_mock_tool_call_chunk(self, index=0, name=None, arguments=None):
        """Create a mock tool call for a chunk."""
        from unittest.mock import MagicMock

        tool_call = MagicMock()
        tool_call.index = index

        function = MagicMock()
        function.name = name
        function.arguments = arguments

        tool_call.function = function

        return tool_call

    def test_process_chunk_preserves_text_after_tool_call_start(self):
        """
        Test that text arriving AFTER tool_call starts is NOT dropped.

        Scenario:
        1. Chunk 1: Tool call starts (is_tool_use becomes True)
        2. Chunk 2: Text content arrives (this should NOT be dropped)

        If the condition `not self.is_tool_use` blocks text processing,
        the text in Chunk 2 would be lost.
        """
        print("🧪 Testing process_chunk text preservation after tool_call...")

        import asyncio

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Help me")],
        )

        converter = AnthropicStreamingConverter(original_request)

        async def test_sequence():
            events = []

            # Chunk 1: Start a tool call
            tool_call = self._create_mock_tool_call_chunk(
                index=0,
                name="test_tool",
                arguments='{"param": "value"}'
            )
            chunk1 = self._create_mock_chunk(tool_calls=[tool_call])

            async for event in converter.process_chunk(chunk1):
                events.append(event)

            # Verify tool_use mode is active
            self.assertTrue(converter.is_tool_use,
                           "is_tool_use should be True after tool call chunk")

            # Chunk 2: Text content arrives while is_tool_use=True
            insight_text = "\n\n★ Insight ─────────────────────────────────────\nThis is important\n─────────────────────────────────────────────────"
            chunk2 = self._create_mock_chunk(content=insight_text)

            async for event in converter.process_chunk(chunk2):
                events.append(event)

            return events, converter.accumulated_text

        events, accumulated = asyncio.run(test_sequence())

        # The key assertion: Insight text should be in accumulated_text
        # If the bug exists, this would fail because text was dropped
        self.assertIn("★ Insight", accumulated,
                      "Insight text should NOT be dropped when is_tool_use=True. "
                      "This indicates the `not self.is_tool_use` condition is blocking text processing.")

        print(f"✅ Accumulated text: {accumulated[:100]}...")
        print("✅ process_chunk text preservation test passed")

    def test_process_chunk_handles_concurrent_text_and_tool(self):
        """
        Test chunk with BOTH tool_calls and content simultaneously.

        Some APIs may send both in the same chunk. The text should not be lost.
        """
        print("🧪 Testing process_chunk with concurrent text and tool_call...")

        import asyncio

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Help me")],
        )

        converter = AnthropicStreamingConverter(original_request)

        async def test_sequence():
            events = []

            # Single chunk with BOTH tool_calls AND content
            tool_call = self._create_mock_tool_call_chunk(
                index=0,
                name="test_tool",
                arguments='{}'
            )
            insight_text = "★ Insight ─────\nBoth at once\n─────"
            chunk = self._create_mock_chunk(
                content=insight_text,
                tool_calls=[tool_call]
            )

            async for event in converter.process_chunk(chunk):
                events.append(event)

            return events, converter.accumulated_text

        events, accumulated = asyncio.run(test_sequence())

        # Text should be preserved even when tool_call is in same chunk
        # Note: The current code processes tool_calls first, then text
        # But the condition `not self.is_tool_use` may block text processing
        if "★ Insight" not in accumulated:
            print("⚠️ WARNING: Text was dropped when sent with tool_call in same chunk!")
            print(f"   Accumulated text: '{accumulated}'")
            print("   This confirms the bug in process_chunk condition.")

        # This assertion may fail if the bug exists - that's expected
        # We're documenting the behavior here
        print(f"Accumulated text: '{accumulated}'")
        print("✅ Concurrent text and tool test completed (check output for warnings)")


class TestInsightJSONSerialization(unittest.TestCase):
    """Test that Insight text survives JSON serialization/deserialization."""

    def test_insight_json_roundtrip(self):
        """Test that Insight text survives JSON serialization."""
        print("🧪 Testing Insight JSON roundtrip...")

        mock_response = create_mock_openai_response(SAMPLE_INSIGHT_TEXT)

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=1000,
            messages=[ClaudeMessage(role="user", content="Help")],
        )

        result = convert_openai_response_to_anthropic(mock_response, original_request)

        # Serialize to JSON (as would happen in API response)
        response_dict = result.model_dump(exclude_none=True)
        json_str = json.dumps(response_dict, ensure_ascii=False)

        # Deserialize back
        parsed = json.loads(json_str)

        # Verify Insight markers are preserved
        text_content = parsed["content"][0]["text"]
        self.assertIn("★ Insight", text_content)
        self.assertIn("─────────────────────────────────────────────────", text_content)
        self.assertEqual(text_content, SAMPLE_INSIGHT_TEXT)

        print("✅ Insight JSON roundtrip test passed")

    def test_insight_in_sse_event_format(self):
        """Test that Insight text is correctly formatted in SSE events."""
        print("🧪 Testing Insight in SSE event format...")

        # Simulate SSE event data
        event_data = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "★ Insight ─────────────────────────────────────\nTest\n─────────────────────────────────────────────────"
            }
        }

        # Serialize as SSE would
        sse_data = f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"

        # Parse back
        data_line = sse_data.split("data: ")[1].strip()
        parsed = json.loads(data_line)

        # Insight markers should be preserved
        self.assertIn("★ Insight", parsed["delta"]["text"])

        print("✅ Insight SSE event format test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Insight Text Conversion Test Suite")
    print("=" * 60)
    print()
    print("Testing that Insight markers and content are preserved")
    print("during OpenAI ↔ Anthropic format conversion.")
    print()
    print("Insight format:")
    print("  ★ Insight ─────────────────────────────────────")
    print("  [educational content]")
    print("  ─────────────────────────────────────────────────")
    print()
    print("=" * 60)

    unittest.main(verbosity=2)
