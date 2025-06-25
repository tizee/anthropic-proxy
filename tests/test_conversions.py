#!/usr/bin/env python3
import unittest

"""
Test script for Claude<->OpenAI message conversion functionality.
Tests both Claude request to OpenAI request conversion and OpenAI response to Claude response conversion.
"""

import json
import sys

# Add parent directory to path for imports
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our conversion functions and models
from models import (
    ClaudeContentBlockImage,
    ClaudeContentBlockText,
    ClaudeContentBlockThinking,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockToolUse,
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeThinkingConfigDisabled,
    ClaudeThinkingConfigEnabled,
    ClaudeTool,
    convert_openai_response_to_anthropic,
)


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
            # Add reasoning_content to the model_dump for extraction
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
            self.prompt_tokens = 10
            self.completion_tokens = 20

    class MockOpenAIResponse:
        def __init__(
            self, content, tool_calls=None, reasoning_content=None, finish_reason="stop"
        ):
            self.choices = [
                MockChoice(content, tool_calls, reasoning_content, finish_reason)
            ]
            self.usage = MockUsage()

    # Create tool calls if provided
    mock_tool_calls = None
    if tool_calls:
        mock_tool_calls = [
            MockToolCall(f"call_{i}", call["name"], json.dumps(call["arguments"]))
            for i, call in enumerate(tool_calls)
        ]

    return MockOpenAIResponse(
        content, mock_tool_calls, reasoning_content, finish_reason
    )


class TestClaudeToOpenAIConversion(unittest.TestCase):
    """Test conversion from Claude format to OpenAI format."""

    def test_basic_claude_to_openai(self):
        """Test basic Claude message conversion to OpenAI format."""
        # Test basic Claude message conversion to OpenAI format

        test_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Hello, world!")],
        )

        result = test_request.to_openai_request()

        # Validate structure
        self.assertIn("model", result)
        self.assertIn("messages", result)
        self.assertIn("max_tokens", result)
        self.assertEqual(result["model"], "test-model")
        self.assertEqual(len(result["messages"]), 1)
        self.assertEqual(result["messages"][0]["role"], "user")
        self.assertEqual(result["messages"][0]["content"], "Hello, world!")

        # Basic conversion test completed

    def test_system_message_conversion(self):
        """Test system message conversion."""
        # Test system message conversion

        test_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Hello!")],
            system="You are a helpful assistant.",
        )

        result = test_request.to_openai_request()

        # Should have system message first
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0]["role"], "system")
        self.assertEqual(
            result["messages"][0]["content"], "You are a helpful assistant."
        )
        self.assertEqual(result["messages"][1]["role"], "user")
        self.assertEqual(result["messages"][1]["content"], "Hello!")

        # System message conversion test completed

    def test_tool_conversion(self):
        """Test tool conversion from Claude to OpenAI format."""
        print("🧪 Testing tool conversion...")

        calculator_tool = ClaudeTool(
            name="calculator",
            description="Evaluate mathematical expressions",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        )

        test_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="What is 2+2?")],
            tools=[calculator_tool],
        )

        result = test_request.to_openai_request()

        # Validate tools conversion
        self.assertIn("tools", result)
        self.assertEqual(len(result["tools"]), 1)

        tool = result["tools"][0]
        self.assertEqual(tool["type"], "function")
        self.assertIn("function", tool)
        self.assertEqual(tool["function"]["name"], "calculator")
        self.assertEqual(
            tool["function"]["description"], "Evaluate mathematical expressions"
        )
        self.assertIn("parameters", tool["function"])

        print("✅ Tool conversion test passed")


class TestOpenAIToClaudeConversion(unittest.TestCase):
    """Test conversion from OpenAI format to Claude format."""

    def test_openai_to_claude_basic(self):
        """Test basic OpenAI to Claude response conversion."""
        print("🧪 Testing basic OpenAI to Claude conversion...")

        # Create mock OpenAI response
        mock_response = create_mock_openai_response("Hello! How can I help you?")

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Hello")],
        )

        result = convert_openai_response_to_anthropic(mock_response, original_request)

        # Validate structure
        self.assertEqual(result.role, "assistant")
        self.assertGreaterEqual(len(result.content), 1)
        self.assertEqual(result.content[0].type, "text")
        self.assertEqual(result.content[0].text, "Hello! How can I help you?")
        self.assertEqual(result.usage.input_tokens, 10)
        self.assertEqual(result.usage.output_tokens, 20)

        print("✅ Basic OpenAI to Claude conversion test passed")

    def test_openai_to_claude_with_tools(self):
        """Test OpenAI to Claude conversion with tool calls."""
        print("🧪 Testing OpenAI to Claude conversion with tools...")

        # Create mock OpenAI response with tool calls
        tool_calls = [{"name": "calculator", "arguments": {"expression": "2+2"}}]
        mock_response = create_mock_openai_response(
            "I'll calculate that for you.", tool_calls
        )

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="What is 2+2?")],
        )

        result = convert_openai_response_to_anthropic(mock_response, original_request)

        # Should have text content and tool use
        self.assertGreaterEqual(len(result.content), 2)

        # Find text and tool blocks
        text_block = None
        tool_block = None

        for block in result.content:
            if block.type == "text":
                text_block = block
            elif block.type == "tool_use":
                tool_block = block

        self.assertIsNotNone(text_block)
        self.assertIsNotNone(tool_block)
        self.assertEqual(text_block.text, "I'll calculate that for you.")
        self.assertEqual(tool_block.name, "calculator")
        self.assertEqual(tool_block.input, {"expression": "2+2"})

        print("✅ OpenAI to Claude tool conversion test passed")

    def test_reasoning_content_to_thinking(self):
        """Test conversion of OpenAI reasoning_content to Claude thinking content block."""
        print("🧪 Testing reasoning_content to thinking conversion...")

        reasoning_text = "Let me think about this step by step. First, I need to understand what the user is asking for. They want to know about 2+2, which is a simple arithmetic problem. I should calculate this: 2+2=4."

        # Create mock OpenAI response with reasoning_content
        mock_response = create_mock_openai_response(
            content="The answer is 4.", reasoning_content=reasoning_text
        )

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="What is 2+2?")],
        )

        result = convert_openai_response_to_anthropic(mock_response, original_request)

        # Should have both thinking and text content blocks
        self.assertGreaterEqual(len(result.content), 2)

        # Find thinking and text blocks
        thinking_block = None
        text_block = None

        for block in result.content:
            if block.type == "thinking":
                thinking_block = block
            elif block.type == "text":
                text_block = block

        # Validate thinking block
        self.assertIsNotNone(thinking_block, "Should have thinking content block")
        self.assertEqual(
            thinking_block.thinking,
            reasoning_text,
            "Thinking content should match reasoning_content",
        )
        self.assertTrue(
            hasattr(thinking_block, "signature"), "Should have thinking signature"
        )

        # Validate text block
        self.assertIsNotNone(text_block, "Should have text content block")
        self.assertEqual(
            text_block.text,
            "The answer is 4.",
            "Text content should match response content",
        )

        print("✅ Reasoning content to thinking conversion test passed")


class TestMessageProcessing(unittest.TestCase):
    """Test message processing and content block handling."""

    def test_mixed_content_message_conversion(self):
        """Test conversion of messages with mixed content types."""
        print("🧪 Testing mixed content message conversion...")

        # Test text + image message
        mixed_message = ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockText(type="text", text="Look at this image: "),
                ClaudeContentBlockImage(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "fake_image_data",
                    },
                ),
                ClaudeContentBlockText(type="text", text=" What do you see?"),
            ],
        )

        test_request = ClaudeMessagesRequest(
            model="test-model", max_tokens=100, messages=[mixed_message]
        )

        result = test_request.to_openai_request()
        messages = result["messages"]

        self.assertEqual(len(messages), 1)
        user_message = messages[0]
        self.assertEqual(user_message["role"], "user")
        self.assertIsInstance(user_message["content"], list)
        self.assertEqual(len(user_message["content"]), 3)

        # Check text content
        text_parts = [
            part for part in user_message["content"] if part["type"] == "text"
        ]
        self.assertEqual(len(text_parts), 2)
        self.assertEqual(text_parts[0]["text"], "Look at this image: ")
        self.assertEqual(text_parts[1]["text"], " What do you see?")

        # Check image content
        image_parts = [
            part for part in user_message["content"] if part["type"] == "image_url"
        ]
        self.assertEqual(len(image_parts), 1)
        self.assertIn(
            "data:image/jpeg;base64,fake_image_data", image_parts[0]["image_url"]["url"]
        )

        print("✅ Mixed content message conversion test passed")

    def test_tool_result_message_ordering(self):
        """Test that tool result messages maintain correct chronological order."""
        print("🧪 Testing tool result message ordering...")

        # Test case with mixed content that includes tool results
        test_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=4000,
            messages=[
                # User message with tool result + text content
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="call_test_123",
                            content="Tool operation completed successfully",
                        ),
                        ClaudeContentBlockText(
                            type="text", text="Thanks! Now let's try something else."
                        ),
                    ],
                )
            ],
        )

        # Convert to OpenAI format
        result = test_request.to_openai_request()
        messages = result["messages"]

        # Should have exactly 2 messages in correct order
        self.assertEqual(
            len(messages), 2, f"Expected exactly 2 messages, got {len(messages)}"
        )

        # First message should be the tool result
        tool_message = messages[0]
        self.assertEqual(
            tool_message["role"],
            "tool",
            f"First message should be tool role, got {tool_message['role']}",
        )
        self.assertEqual(
            tool_message["tool_call_id"], "call_test_123", "Tool call ID should match"
        )
        self.assertIn(
            "Tool operation completed successfully",
            tool_message["content"],
            "Tool result content should match",
        )

        # Second message should be the user content
        user_message = messages[1]
        self.assertEqual(
            user_message["role"],
            "user",
            f"Second message should be user role, got {user_message['role']}",
        )
        self.assertEqual(
            user_message["content"],
            "Thanks! Now let's try something else.",
            "User content should match",
        )

        print("✅ Tool result message ordering test passed")

    def test_thinking_content_conversion(self):
        """Test that thinking content is properly converted to text in assistant message conversion."""
        print("🧪 Testing thinking content conversion...")

        # Test assistant message with text + thinking (thinking should be converted to text)
        message_with_thinking = ClaudeMessage(
            role="assistant",
            content=[
                ClaudeContentBlockText(type="text", text="Regular assistant message"),
                ClaudeContentBlockThinking(
                    type="thinking", thinking="This is internal thinking"
                ),
            ],
        )

        test_request = ClaudeMessagesRequest(
            model="test-model", max_tokens=100, messages=[message_with_thinking]
        )

        result = test_request.to_openai_request()
        messages = result["messages"]

        self.assertEqual(len(messages), 1)
        assistant_message = messages[0]
        self.assertEqual(assistant_message["role"], "assistant")
        # Text + thinking content should now be merged into a single string
        self.assertIsInstance(assistant_message["content"], str)
        self.assertEqual(
            assistant_message["content"],
            "Regular assistant messageThis is internal thinking",
        )

        print("✅ Thinking content conversion test passed")


class TestContentBlockMethods(unittest.TestCase):
    """Test individual content block conversion methods."""

    def test_content_block_methods(self):
        """Test individual content block conversion methods."""
        print("🧪 Testing content block methods...")

        # Test text block
        text_block = ClaudeContentBlockText(type="text", text="Test text")
        text_result = text_block.to_openai()
        self.assertEqual(text_result, {"type": "text", "text": "Test text"})

        # Test image block
        image_block = ClaudeContentBlockImage(
            type="image",
            source={
                "type": "base64",
                "media_type": "image/png",
                "data": "test_image_data",
            },
        )
        image_result = image_block.to_openai()
        self.assertEqual(image_result["type"], "image_url")
        self.assertEqual(
            image_result["image_url"]["url"], "data:image/png;base64,test_image_data"
        )

        # Test thinking block (should return text block)
        thinking_block = ClaudeContentBlockThinking(
            type="thinking", thinking="Internal thoughts"
        )
        thinking_result = thinking_block.to_openai()
        self.assertEqual(thinking_result["type"], "text")
        self.assertEqual(thinking_result["text"], "Internal thoughts")

        # Test tool use block
        tool_use_block = ClaudeContentBlockToolUse(
            type="tool_use",
            id="call_456",
            name="calculator",
            input={"expression": "2+2"},
        )
        tool_use_result = tool_use_block.to_openai()
        self.assertEqual(tool_use_result["id"], "call_456")
        self.assertEqual(tool_use_result["function"]["name"], "calculator")

        # Test tool result block
        tool_result_block = ClaudeContentBlockToolResult(
            type="tool_result", tool_use_id="call_456", content="4"
        )
        tool_result_result = tool_result_block.to_openai_message()
        self.assertEqual(tool_result_result["role"], "tool")
        self.assertEqual(tool_result_result["tool_call_id"], "call_456")
        self.assertEqual(tool_result_result["content"], "4")

        print("✅ Content block methods test passed")


class TestAdvancedFeatures(unittest.TestCase):
    """Test advanced features like function call parsing and complex scenarios."""

    def test_function_call_parsing(self):
        """Test function call parsing from thinking content."""
        print("🧪 Testing function call parsing from thinking content...")

        from models import parse_function_calls_from_thinking

        # Test case 1: Thinking content with function call
        thinking_content = """I need to fix the test_conversions.py file to use Python's unittest framework properly. The previous MultiEdit attempt failed because the exact string pattern couldn't be found. Let me analyze the current state of the file and what needs to be done.

First, looking at the error message: "String to replace not found in file" for "result = test_request.to_openai_request()\\n\\n    assert 'model' in result". This suggests that the code structure might have changed after the initial refactoring, or there might be different whitespace than expected.

Let me recall the current state of the file after previous edits:
1. We've wrapped all test functions in a TestMessageConversion class
2. Converted functions to methods with self parameter
3. Added import unittest
4. Updated main to use unittest.main()

Now, the remaining tasks are:
- Convert assert statements to unittest methods
- Remove print statements
- Remove return statements
- Fix indentation issues

The problem with the previous edit is that I tried to match a specific code pattern with exact newlines and indentation, which might not exist exactly as I specified. Instead, I should target each assert statement individually with more precise patterns.

Let me try a different approach. I'll focus on one test method at a time, starting with test_basic_claude_to_openai, and ensure all asserts are converted and prints are removed.

For test_basic_claude_to_openai, the current code likely looks like:
def test_basic_claude_to_openai(self):
    print("🧪 Testing basic Claude to OpenAI conversion...")

    test_request = ClaudeMessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[
            ClaudeMessage(role='user', content='Hello, world!')
        ]
    )

    result = test_request.to_openai_request()

    assert 'model' in result
    assert 'messages' in result
    assert 'max_tokens' in result
    assert result['model'] == 'test-model'
    assert len(result['messages']) == 1
    assert result['messages'][0]['role'] == 'user'
    assert result['messages'][0]['content'] == 'Hello, world!'

    print("✅ Basic Claude to OpenAI conversion test passed")
    return True

I need to:
1. Remove the print statements
2. Replace each assert with self.assert* methods
3. Remove the return statement

I'll create a MultiEdit with precise patterns for each of these changes, making sure to account for possible whitespace differences. I'll avoid combining multiple lines in old_string to prevent matching issues.

Let me create a new MultiEdit operation with these precise changes for the first test method, then I can proceed with others once this is working.<|FunctionCallBegin|>[
{"name": "MultiEdit", "parameters": {"file_path": "/Users/tizee/projects/project-AI/tools/claude-code-proxy.tizee/test_conversions.py", "edits": [
{"old_string": "    print(\\"🧪 Testing basic Claude to OpenAI conversion...\\")", "new_string": ""},
{"old_string": "    assert 'model' in result", "new_string": "    self.assertIn('model', result)"},
{"old_string": "    assert 'messages' in result", "new_string": "    self.assertIn('messages', result)"},
{"old_string": "    assert 'max_tokens' in result", "new_string": "    self.assertIn('max_tokens', result)"},
{"old_string": "    assert result['model'] == 'test-model'", "new_string": "    self.assertEqual(result['model'], 'test-model')"},
{"old_string": "    assert len(result['messages']) == 1", "new_string": "    self.assertEqual(len(result['messages']), 1)"},
{"old_string": "    assert result['messages'][0]['role'] == 'user'", "new_string": "    self.assertEqual(result['messages'][0]['role'], 'user')"},
{"old_string": "    assert result['messages'][0]['content'] == 'Hello, world!'", "new_string": "    self.assertEqual(result['messages'][0]['content'], 'Hello, world!')"},
{"old_string": "    print(\\"✅ Basic Claude to OpenAI conversion test passed\\")", "new_string": ""},
{"old_string": "    return True", "new_string": ""}
]}}
]<|FunctionCallEnd|>"""

        cleaned_thinking, function_calls = parse_function_calls_from_thinking(
            thinking_content
        )

        # Verify function call was parsed correctly
        self.assertEqual(len(function_calls), 1)

        tool_call = function_calls[0]
        self.assertIn("id", tool_call)
        self.assertEqual(tool_call["type"], "function")
        self.assertEqual(tool_call["function"]["name"], "MultiEdit")

        # Verify arguments contain expected data
        import json

        arguments = json.loads(tool_call["function"]["arguments"])
        self.assertIn("file_path", arguments)
        self.assertIn("edits", arguments)
        self.assertIsInstance(arguments["edits"], list)
        self.assertGreater(len(arguments["edits"]), 0)

        # Verify thinking content was cleaned (function calls removed)
        self.assertLess(len(cleaned_thinking), len(thinking_content))
        self.assertNotIn("<|FunctionCallBegin|>", cleaned_thinking)
        self.assertNotIn("<|FunctionCallEnd|>", cleaned_thinking)

        # Test case 2: Thinking content without function calls
        simple_thinking = "This is just thinking content without any function calls."
        cleaned_simple, simple_calls = parse_function_calls_from_thinking(
            simple_thinking
        )

        self.assertEqual(len(simple_calls), 0)
        self.assertEqual(cleaned_simple, simple_thinking)

        print("✅ Function call parsing test passed")

    def test_complex_conversation_flow(self):
        """Test a complex multi-turn conversation with tools."""
        print("🧪 Testing complex conversation flow...")

        # Create a complex conversation similar to real Claude Code usage
        test_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=4000,
            messages=[
                # 1. Initial user question
                ClaudeMessage(role="user", content="What's the weather like?"),
                # 2. Assistant responds with tool use
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text", text="I'll check the weather for you."
                        ),
                        # -> tool_calls
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu_weather_123",
                            name="get_weather",
                            input={"location": "current"},
                        ),
                    ],
                ),
                # 3. User provides tool result and asks follow-up
                ClaudeMessage(
                    role="user",
                    content=[
                        # split here -> tool role message
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="toolu_weather_123",
                            content="Sunny, 75°F",
                        ),
                        # -> new user role message
                        ClaudeContentBlockText(
                            type="text", text="That's nice! What about tomorrow?"
                        ),
                    ],
                ),
                # 4. Assistant final response
                ClaudeMessage(
                    role="assistant",
                    content="Let me check tomorrow's forecast as well.",
                ),
            ],
            tools=[
                ClaudeTool(
                    name="get_weather",
                    description="Get weather information",
                    input_schema={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                )
            ],
        )

        # Convert to OpenAI format
        result = test_request.to_openai_request()
        messages = result["messages"]

        # Validate message structure - should be 5 messages due to message splitting
        # user -> assistant -> tool -> user -> assistant
        self.assertEqual(len(messages), 5, f"Expected 5 messages, got {len(messages)}")

        # Check message roles and order
        expected_roles = ["user", "assistant", "tool", "user", "assistant"]
        actual_roles = [msg["role"] for msg in messages]
        self.assertEqual(
            actual_roles,
            expected_roles,
            f"Expected roles {expected_roles}, got {actual_roles}",
        )

        # Validate assistant message with tool calls
        assistant_msg = messages[1]
        self.assertEqual(assistant_msg["role"], "assistant")
        self.assertEqual(assistant_msg["content"], "I'll check the weather for you.")
        self.assertIn("tool_calls", assistant_msg)
        self.assertEqual(len(assistant_msg["tool_calls"]), 1)
        self.assertEqual(
            assistant_msg["tool_calls"][0]["function"]["name"], "get_weather"
        )

        # Validate tool result
        tool_msg = messages[2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "toolu_weather_123")
        self.assertEqual(tool_msg["content"], "Sunny, 75°F")

        # Validate follow-up user message
        followup_user_msg = messages[3]
        self.assertEqual(followup_user_msg["role"], "user")
        self.assertEqual(
            followup_user_msg["content"], "That's nice! What about tomorrow?"
        )

        print("✅ Complex conversation flow test passed")

    def test_thinking_configuration(self):
        """Test thinking configuration handling."""
        print("🧪 Testing thinking configuration...")

        # Test with thinking enabled
        test_request_enabled = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[
                ClaudeMessage(role="user", content="Think about this problem...")
            ],
            thinking=ClaudeThinkingConfigEnabled(type="enabled", budget_tokens=500),
        )

        result_enabled = test_request_enabled.to_openai_request()
        # Should still convert normally (thinking is handled in routing)
        self.assertIn("messages", result_enabled)
        self.assertEqual(len(result_enabled["messages"]), 1)

        # Test with thinking disabled
        test_request_disabled = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Regular request...")],
            thinking=ClaudeThinkingConfigDisabled(type="disabled"),
        )

        result_disabled = test_request_disabled.to_openai_request()
        self.assertIn("messages", result_disabled)
        self.assertEqual(len(result_disabled["messages"]), 1)

        print("✅ Thinking configuration test passed")

    def test_message_to_openai_conversion(self):
        """Test message to_openai conversion methods."""
        print("🧪 Testing message to_openai conversion...")

        # Test simple text message conversion
        text_message = ClaudeMessage(role="user", content="Simple text")
        openai_messages = text_message.to_openai_messages()
        self.assertEqual(len(openai_messages), 1)
        self.assertEqual(openai_messages[0]["role"], "user")
        self.assertEqual(openai_messages[0]["content"], "Simple text")

        # Test mixed content message conversion
        mixed_message = ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockText(type="text", text="Hello "),
                ClaudeContentBlockText(type="text", text="world!"),
            ],
        )
        openai_mixed = mixed_message.to_openai_messages()
        self.assertEqual(len(openai_mixed), 1)
        self.assertEqual(openai_mixed[0]["role"], "user")
        # Multiple text blocks should now be merged into a single string
        self.assertIsInstance(openai_mixed[0]["content"], str)
        self.assertEqual(openai_mixed[0]["content"], "Hello world!")

        # Test assistant message with tool use conversion
        assistant_message = ClaudeMessage(
            role="assistant",
            content=[
                ClaudeContentBlockText(type="text", text="Let me help you."),
                ClaudeContentBlockToolUse(
                    type="tool_use",
                    id="tool_123",
                    name="helper",
                    input={"param": "value"},
                ),
            ],
        )
        openai_assistant = assistant_message.to_openai_messages()
        self.assertEqual(len(openai_assistant), 1)
        self.assertEqual(openai_assistant[0]["role"], "assistant")
        self.assertEqual(openai_assistant[0]["content"], "Let me help you.")
        self.assertIn("tool_calls", openai_assistant[0])
        self.assertEqual(len(openai_assistant[0]["tool_calls"]), 1)
        self.assertEqual(openai_assistant[0]["tool_calls"][0]["id"], "tool_123")
        self.assertEqual(
            openai_assistant[0]["tool_calls"][0]["function"]["name"], "helper"
        )

        # Test user message with tool result conversion (should split into multiple messages)
        user_message = ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockToolResult(
                    type="tool_result", tool_use_id="tool_123", content="Success"
                ),
                ClaudeContentBlockText(type="text", text="Thanks!"),
            ],
        )
        openai_user = user_message.to_openai_messages()
        self.assertEqual(
            len(openai_user), 2
        )  # Should split into tool message + user message

        # First should be tool result message
        self.assertEqual(openai_user[0]["role"], "tool")
        self.assertEqual(openai_user[0]["tool_call_id"], "tool_123")
        self.assertEqual(openai_user[0]["content"], "Success")

        # Second should be user text message
        self.assertEqual(openai_user[1]["role"], "user")
        self.assertEqual(openai_user[1]["content"], "Thanks!")

        print("✅ Message to_openai conversion test passed")

    def test_tool_sequence_interruption_conversion(self):
        """Test the specific tool message sequence conversion that's failing in the interruption test."""
        print("🔧 Testing tool message sequence conversion for interruption case...")

        # Mock tool definition
        exit_plan_mode_tool = ClaudeTool(
            name="exit_plan_mode",
            description="Exit plan mode tool",
            input_schema={
                "type": "object",
                "properties": {
                    "plan": {"type": "string", "description": "The plan to exit"}
                },
                "required": ["plan"],
            },
        )

        # Create the Claude request that mimics the failing test
        claude_request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                # Assistant message with tool use
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="call_jkl345mno678",
                            name="exit_plan_mode",
                            input={
                                "plan": "I will create an example configuration file with placeholder values for each field, maintaining the same structure and adding helpful comments."
                            },
                        )
                    ],
                ),
                # User message with tool_result first, then text (critical test case)
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="call_jkl345mno678",
                            content="The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). STOP what you are doing and wait for the user to tell you how to proceed.",
                        ),
                        ClaudeContentBlockText(
                            type="text",
                            text="[Request interrupted by user for tool use]",
                        ),
                        ClaudeContentBlockText(
                            type="text",
                            text="Actually, the example file already exists. Please check before creating new files.",
                        ),
                    ],
                ),
            ],
            tools=[exit_plan_mode_tool],
        )

        print(f"📝 Claude request has {len(claude_request.messages)} messages")

        # Debug: Print each Claude message
        for i, msg in enumerate(claude_request.messages):
            print(
                f"  Claude Message {i}: role={msg.role}, content_blocks={len(msg.content) if isinstance(msg.content, list) else 1}"
            )
            if isinstance(msg.content, list):
                for j, block in enumerate(msg.content):
                    print(f"    Block {j}: type={block.type}")

        # Convert to OpenAI format
        openai_request = claude_request.to_openai_request()
        openai_messages = openai_request["messages"]

        print(f"📤 Converted to {len(openai_messages)} OpenAI messages:")
        for i, msg in enumerate(openai_messages):
            role = msg.get("role", "unknown")
            has_tool_calls = (
                msg.get("tool_calls") is not None and len(msg.get("tool_calls", [])) > 0
            )
            has_content = bool(msg.get("content"))
            print(
                f"  Message {i}: role={role}, has_tool_calls={has_tool_calls}, has_content={has_content}"
            )

            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "unknown")
                print(f"    Tool message: tool_call_id={tool_call_id}")

        # Check if the sequence follows OpenAI rules
        valid_sequence = True
        last_had_tool_calls = False

        for i, msg in enumerate(openai_messages):
            if msg.get("role") == "tool":
                if not last_had_tool_calls:
                    print(
                        f"❌ Invalid sequence: Tool message at index {i} doesn't follow assistant message with tool_calls"
                    )
                    valid_sequence = False
                    break

            last_had_tool_calls = (
                msg.get("role") == "assistant"
                and msg.get("tool_calls") is not None
                and len(msg.get("tool_calls", [])) > 0
            )

        if valid_sequence:
            print("✅ Message sequence is valid for OpenAI API")
        else:
            print("❌ Message sequence violates OpenAI API rules")

        # Assert that the sequence is valid - this will make the test fail if it's not
        self.assertTrue(
            valid_sequence, "Tool message sequence should be valid for OpenAI API"
        )

        # Additional assertions to verify the specific structure
        self.assertGreaterEqual(
            len(openai_messages),
            3,
            "Should have at least 3 messages: assistant + tool + user",
        )

        # First message should be assistant with tool_calls
        self.assertEqual(openai_messages[0]["role"], "assistant")
        self.assertIn("tool_calls", openai_messages[0])
        self.assertEqual(len(openai_messages[0]["tool_calls"]), 1)
        self.assertEqual(openai_messages[0]["tool_calls"][0]["id"], "call_jkl345mno678")

        # Second message should be tool result
        self.assertEqual(openai_messages[1]["role"], "tool")
        self.assertEqual(openai_messages[1]["tool_call_id"], "call_jkl345mno678")

        # Third message should be user
        self.assertEqual(openai_messages[2]["role"], "user")

        print("✅ Tool sequence interruption conversion test passed")

    def test_streaming_tool_id_consistency_bug(self):
        """Test the specific tool use ID consistency bug in streaming responses."""
        print("🐛 Testing streaming tool use ID consistency bug...")

        # This test reproduces the bug where assistant message content gets lost
        # when converting streaming responses that contain both text and tool_calls

        # Create a Claude message that has both text content and tool_use (like the bug scenario)
        mixed_assistant_message = ClaudeMessage(
            role="assistant",
            content=[
                ClaudeContentBlockText(
                    type="text",
                    text="Of course. I will add commands to the `Makefile` to generate test coverage reports using `pytest`, and I will update the `README.md` accordingly.",
                ),
                ClaudeContentBlockToolUse(
                    type="tool_use",
                    id="tool_0_exit_plan_mode",  # This is the Claude Code frontend format
                    name="exit_plan_mode",
                    input={
                        "plan": '1. **Update Makefile**:\n    - Add a `test-cov` target to generate a terminal-based coverage report.\n    - Add a `test-cov-html` target to generate a more detailed HTML coverage report.\n    - Update the `help` command to include these new testing options.\n2.  **Update README.md**:\n    - Add a new "Test Coverage" section explaining how to run the new `make test-cov` and `make test-cov-html` commands.'
                    },
                ),
            ],
        )

        # Convert to OpenAI format
        openai_messages = mixed_assistant_message.to_openai_messages()

        # Should have exactly 1 message (no splitting needed since no tool_result)
        self.assertEqual(
            len(openai_messages), 1, f"Expected 1 message, got {len(openai_messages)}"
        )

        message = openai_messages[0]

        # Check basic structure
        self.assertEqual(message["role"], "assistant")
        self.assertIn("content", message)
        self.assertIn("tool_calls", message)

        # CRITICAL: Content should NOT be empty or None - it should contain the text
        self.assertIsNotNone(
            message["content"], "Assistant message content should not be None"
        )
        self.assertNotEqual(
            message["content"], "", "Assistant message content should not be empty"
        )
        self.assertIn(
            "I will add commands",
            message["content"],
            "Content should contain the original text",
        )

        # Tool calls should be properly formatted
        self.assertEqual(len(message["tool_calls"]), 1)
        tool_call = message["tool_calls"][0]
        self.assertEqual(
            tool_call["id"], "tool_0_exit_plan_mode"
        )  # ID should be preserved
        self.assertEqual(tool_call["function"]["name"], "exit_plan_mode")

        print("✅ Streaming tool use ID consistency bug test passed")

    def test_complete_tool_use_flow_with_mixed_content(self):
        """Test the complete flow: Claude request → OpenAI request → OpenAI response → Claude response."""
        print("🔄 Testing complete tool use flow with mixed content...")

        # 1. Create Claude request with mixed content (text + tool_use)
        claude_request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text="I'll help you implement those features. Let me create a plan first.",
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="tool_0_exit_plan_mode",
                            name="exit_plan_mode",
                            input={
                                "plan": "Implementation plan for the requested features"
                            },
                        ),
                    ],
                )
            ],
        )

        # 2. Convert Claude → OpenAI
        openai_request = claude_request.to_openai_request()
        openai_messages = openai_request["messages"]

        # Verify OpenAI format
        self.assertEqual(len(openai_messages), 1)
        assistant_msg = openai_messages[0]
        self.assertEqual(assistant_msg["role"], "assistant")

        # CRITICAL: Content should be preserved
        self.assertIn("content", assistant_msg)
        self.assertIn("I'll help you implement", assistant_msg["content"])

        # Tool calls should be present
        self.assertIn("tool_calls", assistant_msg)
        self.assertEqual(len(assistant_msg["tool_calls"]), 1)
        self.assertEqual(assistant_msg["tool_calls"][0]["id"], "tool_0_exit_plan_mode")

        # 3. Simulate OpenAI response (what would come back from OpenAI API)
        mock_openai_response = create_mock_openai_response(
            content="I'll help you implement those features. Let me create a plan first.",
            tool_calls=[
                {
                    "name": "exit_plan_mode",
                    "arguments": {
                        "plan": "Implementation plan for the requested features"
                    },
                }
            ],
        )

        # 4. Convert OpenAI response → Claude response
        claude_response = convert_openai_response_to_anthropic(
            mock_openai_response, claude_request
        )

        # Verify Claude response format
        self.assertEqual(claude_response.role, "assistant")
        self.assertGreaterEqual(
            len(claude_response.content), 2
        )  # Should have text + tool_use

        # Find content blocks
        text_blocks = [
            block for block in claude_response.content if block.type == "text"
        ]
        tool_blocks = [
            block for block in claude_response.content if block.type == "tool_use"
        ]

        # Verify content preservation
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(len(tool_blocks), 1)
        self.assertIn("I'll help you implement", text_blocks[0].text)
        self.assertEqual(tool_blocks[0].name, "exit_plan_mode")

        print("✅ Complete tool use flow with mixed content test passed")

    def test_exit_plan_mode_scenario_from_logs(self):
        """Test the exact exit_plan_mode scenario from user's logs to isolate the 'no content' issue."""
        print("🔍 Testing exit_plan_mode scenario from logs...")

        # Recreate the exact scenario from the user's logs
        # This tests whether the issue is in model behavior or proxy conversion

        # 1. Create the Claude request that would be sent to OpenAI
        # This represents the conversation state when exit_plan_mode is called
        claude_request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                # Previous assistant message with exit_plan_mode tool call
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text='Of course. I will add commands to the `Makefile` to generate test coverage reports using `pytest`, and I will update the `README.md` accordingly.\n\nHere is my plan:\n\n1.  **Update Makefile**:\n    *   Add a `test-cov` target to generate a terminal-based coverage report.\n    *   Add a `test-cov-html` target to generate a more detailed HTML coverage report.\n    *   Update the `help` command to include these new testing options.\n2.  **Update README.md**:\n    *   Add a new "Test Coverage" section explaining how to run the new `make test-cov` and `make test-cov-html` commands.',
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="tool_0_exit_plan_mode",
                            name="exit_plan_mode",
                            input={
                                "plan": '1. **Update Makefile**:\\n    - Add a `test-cov` target to generate a terminal-based coverage report.\\n    - Add a `test-cov-html` target to generate a more detailed HTML coverage report.\\n    - Update the `help` command to include these new testing options.\\n2.  **Update README.md**:\\n    - Add a new "Test Coverage" section explaining how to run the new `make test-cov` and `make test-cov-html` commands.'
                            },
                        ),
                    ],
                ),
                # Tool result message (exit_plan_mode approved)
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="tool_0_exit_plan_mode",
                            content="User has approved your plan. You can now start coding.",
                        )
                    ],
                ),
                # User's follow-up message
                ClaudeMessage(role="user", content="重试一下"),
            ],
        )

        print(f"📝 Created Claude request with {len(claude_request.messages)} messages")

        # 2. Convert to OpenAI format (this is what gets sent to the actual model)
        openai_request = claude_request.to_openai_request()
        openai_messages = openai_request["messages"]

        print(f"📤 Converted to {len(openai_messages)} OpenAI messages:")
        for i, msg in enumerate(openai_messages):
            role = msg.get("role", "unknown")
            has_tool_calls = "tool_calls" in msg and msg["tool_calls"]
            content_preview = ""
            if "content" in msg and msg["content"]:
                content_str = str(msg["content"])
                content_preview = (
                    content_str[:50] + "..." if len(content_str) > 50 else content_str
                )

            print(
                f"  Message {i}: role={role}, has_tool_calls={has_tool_calls}, content='{content_preview}'"
            )

        # 3. Verify the OpenAI request structure matches expectations
        # This should match the problematic sequence from the logs

        # The first message should be assistant with both content and tool_calls
        self.assertGreaterEqual(
            len(openai_messages),
            3,
            "Should have at least assistant + tool + user messages",
        )

        # Check first message (assistant with tool call)
        first_msg = openai_messages[0]
        self.assertEqual(first_msg["role"], "assistant")
        self.assertIn("content", first_msg, "Assistant message should have content")
        self.assertIn(
            "tool_calls", first_msg, "Assistant message should have tool_calls"
        )

        # CRITICAL: Check if content is preserved
        if "content" in first_msg:
            content = first_msg["content"]
            if content is None or content == "" or content == "(no content)":
                print(
                    f"❌ FOUND THE BUG: Assistant message content is '{content}' - should contain the plan text!"
                )
                self.fail(
                    f"Assistant message content should not be empty/None, got: '{content}'"
                )
            else:
                print(f"✅ Assistant message content preserved: '{content[:100]}...'")
                self.assertIn(
                    "I will add commands",
                    content,
                    "Content should contain the original text",
                )

        # Check tool call structure
        tool_calls = first_msg["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["id"], "tool_0_exit_plan_mode")
        self.assertEqual(tool_calls[0]["function"]["name"], "exit_plan_mode")

        # 4. Simulate what happens when this gets sent to an actual model
        # Create a mock response that simulates the model's behavior after exit_plan_mode
        mock_model_response = create_mock_openai_response(
            content="",  # This simulates the potential model behavior of returning empty content
            tool_calls=[
                {
                    "name": "Read",
                    "arguments": {
                        "file_path": "/Users/tizee/projects/project-AI/tools/claude-code-proxy.tizee/Makefile"
                    },
                }
            ],
            finish_reason="tool_calls",
        )

        # 5. Convert the mock response back to Claude format
        claude_response = convert_openai_response_to_anthropic(
            mock_model_response, claude_request
        )

        print("🔄 Mock model response converted back to Claude format:")
        print(f"   Role: {claude_response.role}")
        print(f"   Content blocks: {len(claude_response.content)}")

        for i, block in enumerate(claude_response.content):
            if hasattr(block, "type"):
                if block.type == "text":
                    text_preview = (
                        block.text[:50] + "..." if len(block.text) > 50 else block.text
                    )
                    print(f"   Block {i}: text = '{text_preview}'")
                elif block.type == "tool_use":
                    print(f"   Block {i}: tool_use = {block.name}")

        # 6. The key test: Check if the conversion preserves the expected behavior
        # If the original model returns empty content with tool_calls, that might be normal
        # But if our conversion is losing non-empty content, that's our bug

        print(
            "🎯 Test completed - check the output above to see if content is properly preserved"
        )
        print("✅ Exit plan mode scenario test completed")

    def test_tool_use_id_uniqueness(self):
        """Test that tool use IDs are unique across multiple conversions."""
        print("🔑 Testing tool use ID uniqueness...")

        # Create multiple tool calls with the same original ID (simulating Gemini API behavior)
        mock_tool_calls = [
            {
                "name": "Edit",
                "arguments": {
                    "file_path": "/test1.py",
                    "old_string": "old",
                    "new_string": "new",
                },
            },
            {
                "name": "Edit",
                "arguments": {
                    "file_path": "/test2.py",
                    "old_string": "old",
                    "new_string": "new",
                },
            },
            {"name": "Read", "arguments": {"file_path": "/test3.py"}},
        ]

        # Create multiple responses that would have the same tool IDs (like Gemini)
        response1 = create_mock_openai_response(
            "I'll make these changes.", mock_tool_calls[:2]
        )
        response2 = create_mock_openai_response(
            "Let me also read this file.", mock_tool_calls[2:]
        )

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Please make these changes")],
        )

        # Convert both responses
        claude_response1 = convert_openai_response_to_anthropic(
            response1, original_request
        )
        claude_response2 = convert_openai_response_to_anthropic(
            response2, original_request
        )

        # Collect all tool use IDs from both responses
        tool_ids = []

        for response in [claude_response1, claude_response2]:
            for block in response.content:
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_ids.append(block.id)

        # Test 1: All IDs should be unique
        self.assertEqual(
            len(tool_ids),
            len(set(tool_ids)),
            f"Tool use IDs should be unique. Found duplicates: {tool_ids}",
        )

        # Test 2: IDs should follow our custom format (timestamp-based)
        for tool_id in tool_ids:
            self.assertRegex(
                tool_id,
                r"^toolu_\d+_[a-f0-9]{8}$",
                f"Tool ID should match format 'toolu_<timestamp>_<hex>': {tool_id}",
            )

        # Test 3: Verify that multiple calls to generate_unique_tool_id() produce different IDs
        from models import generate_unique_tool_id

        generated_ids = [generate_unique_tool_id() for _ in range(10)]
        self.assertEqual(
            len(generated_ids),
            len(set(generated_ids)),
            f"Generated IDs should be unique: {generated_ids}",
        )

        print("✅ Tool use ID uniqueness test passed")

    def test_tool_use_id_consistency_in_streaming(self):
        """Test that tool use IDs remain consistent when converting from streaming responses."""
        print("🔄 Testing tool use ID consistency in streaming...")

        # Simulate a streaming scenario where the same tool is called multiple times
        # This tests the fix for the Gemini API returning duplicate IDs like "tool_0_Edit"

        mock_tool_calls_with_duplicate_ids = [
            # Simulate what Gemini API might return (duplicate IDs)
            type(
                "MockToolCall",
                (),
                {
                    "id": "tool_0_Edit",  # This is the problematic duplicate ID
                    "function": type(
                        "MockFunction",
                        (),
                        {
                            "name": "Edit",
                            "arguments": '{"file_path": "/test1.py", "old_string": "old1", "new_string": "new1"}',
                        },
                    )(),
                },
            )(),
            type(
                "MockToolCall",
                (),
                {
                    "id": "tool_0_Edit",  # Same ID again - this should be made unique
                    "function": type(
                        "MockFunction",
                        (),
                        {
                            "name": "Edit",
                            "arguments": '{"file_path": "/test2.py", "old_string": "old2", "new_string": "new2"}',
                        },
                    )(),
                },
            )(),
        ]

        mock_response = create_mock_openai_response(
            "I'll make these edits.",
            [
                {
                    "name": call.function.name,
                    "arguments": json.loads(call.function.arguments),
                }
                for call in mock_tool_calls_with_duplicate_ids
            ],
        )

        # Force the tool call IDs to be the problematic duplicates
        mock_response.choices[0].message.tool_calls = mock_tool_calls_with_duplicate_ids

        original_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Please edit these files")],
        )

        # Convert to Claude format
        claude_response = convert_openai_response_to_anthropic(
            mock_response, original_request
        )

        # Collect tool use blocks
        tool_blocks = [
            block
            for block in claude_response.content
            if hasattr(block, "type") and block.type == "tool_use"
        ]

        # Should have 2 tool use blocks
        self.assertEqual(len(tool_blocks), 2, "Should have 2 tool use blocks")

        # IDs should be unique despite the original duplicates
        tool_ids = [block.id for block in tool_blocks]
        self.assertEqual(
            len(tool_ids),
            len(set(tool_ids)),
            f"Tool use IDs should be unique even when source had duplicates: {tool_ids}",
        )

        # IDs should NOT be the original problematic format
        for tool_id in tool_ids:
            self.assertNotEqual(
                tool_id,
                "tool_0_Edit",
                f"Tool ID should not be the original duplicate ID: {tool_id}",
            )

        # IDs should follow our unique format
        for tool_id in tool_ids:
            self.assertRegex(
                tool_id,
                r"^toolu_\d+_[a-f0-9]{8}$",
                f"Tool ID should follow unique format: {tool_id}",
            )

        print("✅ Tool use ID consistency in streaming test passed")

    @staticmethod
    def create_mock_openai_response(
        content: str,
        tool_calls=None,
        reasoning_content: str = "",
        finish_reason: str = "stop",
    ):
        """Run all conversion tests."""
        print("🚀 Running Claude<->OpenAI conversion tests...\n")

        tests = [
            test_basic_claude_to_openai,
            test_system_message_conversion,
            test_tool_conversion,
            test_openai_to_claude_basic,
            test_openai_to_claude_with_tools,
            test_reasoning_content_to_thinking,
            test_mixed_content_message_conversion,
            test_tool_result_message_ordering,
            test_thinking_content_filtering,
            test_content_block_methods,
            test_complex_conversation_flow,
            test_thinking_configuration,
            test_message_extraction_methods,
        ]

        passed = 0
        total = len(tests)

        for test in tests:
            try:
                if test():
                    passed += 1
            except Exception as e:
                print(f"❌ Test {test.__name__} failed: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n📊 Test Results: {passed}/{total} tests passed")

        if passed == total:
            print("🎉 All conversion tests passed!")
            return True
        else:
            print(f"⚠️ {total - passed} conversion tests failed")
            return False


if __name__ == "__main__":
    unittest.main()
    success = run_all_conversion_tests()
    sys.exit(0 if success else 1)
