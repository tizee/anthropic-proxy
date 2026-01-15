import unittest

from anthropic_proxy.thinking_recovery import (
    ConversationState,
    analyze_conversation_state,
    close_tool_loop_for_thinking,
    needs_thinking_recovery,
)


class TestThinkingRecovery(unittest.TestCase):
    """Tests for the thinking recovery module."""

    def test_analyze_empty_conversation(self):
        """Test analyzing an empty conversation."""
        state = analyze_conversation_state([])
        self.assertFalse(state.in_tool_loop)
        self.assertEqual(state.turn_start_idx, -1)
        self.assertFalse(state.turn_has_thinking)

    def test_analyze_simple_conversation(self):
        """Test analyzing a simple user-model conversation."""
        contents = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there"}]},
        ]
        state = analyze_conversation_state(contents)
        self.assertFalse(state.in_tool_loop)
        self.assertEqual(state.turn_start_idx, 1)
        self.assertFalse(state.turn_has_thinking)
        self.assertEqual(state.last_model_idx, 1)

    def test_analyze_conversation_with_thinking(self):
        """Test analyzing conversation with thinking block."""
        contents = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {
                "role": "model",
                "parts": [
                    {"thought": True, "text": "Let me think..."},
                    {"text": "Hi there"},
                ],
            },
        ]
        state = analyze_conversation_state(contents)
        self.assertTrue(state.turn_has_thinking)
        self.assertTrue(state.last_model_has_thinking)

    def test_analyze_in_tool_loop(self):
        """Test detecting a conversation in a tool loop."""
        contents = [
            {"role": "user", "parts": [{"text": "What's the weather?"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}, "id": "tool-1"}}
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "get_weather", "id": "tool-1", "response": {"content": "72F"}}}
                ],
            },
        ]
        state = analyze_conversation_state(contents)
        self.assertTrue(state.in_tool_loop)
        self.assertTrue(state.last_model_has_tool_calls)

    def test_analyze_conversation_with_tool_calls_and_thinking(self):
        """Test analyzing conversation with both tool calls and thinking."""
        contents = [
            {"role": "user", "parts": [{"text": "What's the weather?"}]},
            {
                "role": "model",
                "parts": [
                    {"thought": True, "text": "I need to check the weather."},
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}, "id": "tool-1"}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "get_weather", "id": "tool-1", "response": {"content": "72F"}}}
                ],
            },
        ]
        state = analyze_conversation_state(contents)
        self.assertTrue(state.in_tool_loop)
        self.assertTrue(state.turn_has_thinking)  # Turn started with thinking
        self.assertTrue(state.last_model_has_tool_calls)

    def test_needs_thinking_recovery_in_tool_loop_without_thinking(self):
        """Test that recovery is needed when in tool loop without thinking."""
        state = ConversationState(in_tool_loop=True, turn_has_thinking=False)
        self.assertTrue(needs_thinking_recovery(state))

    def test_needs_thinking_recovery_not_in_tool_loop(self):
        """Test that recovery is not needed when not in tool loop."""
        state = ConversationState(in_tool_loop=False, turn_has_thinking=False)
        self.assertFalse(needs_thinking_recovery(state))

    def test_needs_thinking_recovery_in_tool_loop_with_thinking(self):
        """Test that recovery is not needed when turn started with thinking."""
        state = ConversationState(in_tool_loop=True, turn_has_thinking=True)
        self.assertFalse(needs_thinking_recovery(state))

    def test_close_tool_loop_no_tool_results(self):
        """Test closing tool loop when there are no tool results."""
        contents = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi"}]},
        ]
        result = close_tool_loop_for_thinking(contents)
        self.assertEqual(len(result), 4)  # 2 original + 2 synthetic
        self.assertEqual(result[-2]["role"], "model")
        self.assertEqual(result[-2]["parts"][0]["text"], "[Processing previous context.]")
        self.assertEqual(result[-1]["role"], "user")
        self.assertEqual(result[-1]["parts"][0]["text"], "[Continue]")

    def test_close_tool_loop_single_tool_result(self):
        """Test closing tool loop with one tool result."""
        contents = [
            {"role": "user", "parts": [{"text": "What's the weather?"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}, "id": "tool-1"}}
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "get_weather", "id": "tool-1", "response": {"content": "72F"}}}
                ],
            },
        ]
        result = close_tool_loop_for_thinking(contents)
        self.assertEqual(len(result), 5)  # 3 original + 2 synthetic
        self.assertEqual(result[-2]["parts"][0]["text"], "[Tool execution completed.]")

    def test_close_tool_loop_multiple_tool_results(self):
        """Test closing tool loop with multiple tool results."""
        contents = [
            {"role": "user", "parts": [{"text": "Check weather and time"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}, "id": "tool-1"}},
                    {"functionCall": {"name": "get_time", "args": {}, "id": "tool-2"}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "get_weather", "id": "tool-1", "response": {"content": "72F"}}},
                    {"functionResponse": {"name": "get_time", "id": "tool-2", "response": {"content": "10:30 AM"}}},
                ],
            },
        ]
        result = close_tool_loop_for_thinking(contents)
        self.assertEqual(len(result), 5)  # 3 original + 2 synthetic
        self.assertEqual(result[-2]["parts"][0]["text"], "[2 tool executions completed.]")

    def test_close_tool_loop_strips_thinking_blocks(self):
        """Test that closing tool loop strips all thinking blocks."""
        contents = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {
                "role": "model",
                "parts": [
                    {"thought": True, "text": "Corrupted thinking..."},
                    {"text": "Some response"},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "test", "id": "tool-1", "response": {"content": "result"}}}
                ],
            },
        ]
        result = close_tool_loop_for_thinking(contents)

        # Check that thinking blocks were stripped from original messages
        for content in result[:-2]:  # Exclude synthetic messages
            for part in content.get("parts", []):
                self.assertNotIn("thought", part)

    def test_close_tool_loop_handles_empty_parts(self):
        """Test closing tool loop handles messages without parts."""
        contents = [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": []},  # Empty parts
        ]
        result = close_tool_loop_for_thinking(contents)
        # Should still add synthetic messages
        self.assertEqual(result[-2]["role"], "model")
        self.assertEqual(result[-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
