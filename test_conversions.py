#!/usr/bin/env python3
import unittest

"""
Test script for Claude<->OpenAI message conversion functionality.
Tests both Claude request to OpenAI request conversion and OpenAI response to Claude response conversion.
"""

import sys
import json
import uuid
from typing import Any, Dict, List

# Import our conversion functions and models
from models import (
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeMessage,
    ClaudeContentBlockText,
    ClaudeContentBlockImage,
    ClaudeContentBlockToolUse,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockThinking,
    ClaudeTool,
    ClaudeThinkingConfigEnabled,
    ClaudeThinkingConfigDisabled,
    ClaudeUsage,
    convert_openai_response_to_anthropic,
    Constants,
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


class TestMessageConversion(unittest.TestCase):
    def test_basic_claude_to_openai(self):
        """Test basic Claude message conversion to OpenAI format."""
        print("🧪 Testing basic Claude to OpenAI conversion...")

        test_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Hello, world!")],
        )

        result = test_request.to_openai_request()

        # Validate structure
        assert "model" in result
        assert "messages" in result
        assert "max_tokens" in result
        assert result["model"] == "test-model"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello, world!"

        print("✅ Basic Claude to OpenAI conversion test passed")

    def test_system_message_conversion(self):
        """Test system message conversion."""
        print("🧪 Testing system message conversion...")

        test_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Hello!")],
            system="You are a helpful assistant.",
        )

        result = test_request.to_openai_request()

        # Should have system message first
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "Hello!"

        print("✅ System message conversion test passed")

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
        assert "tools" in result
        assert len(result["tools"]) == 1

        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert "function" in tool
        assert tool["function"]["name"] == "calculator"
        assert tool["function"]["description"] == "Evaluate mathematical expressions"
        assert "parameters" in tool["function"]

        print("✅ Tool conversion test passed")

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
        assert result.role == "assistant"
        assert len(result.content) >= 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello! How can I help you?"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20

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
        assert len(result.content) >= 2

        # Find text and tool blocks
        text_block = None
        tool_block = None

        for block in result.content:
            if block.type == "text":
                text_block = block
            elif block.type == "tool_use":
                tool_block = block

        assert text_block is not None
        assert tool_block is not None
        assert text_block.text == "I'll calculate that for you."
        assert tool_block.name == "calculator"
        assert tool_block.input == {"expression": "2+2"}

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
        assert len(result.content) >= 2

        # Find thinking and text blocks
        thinking_block = None
        text_block = None

        for block in result.content:
            if block.type == "thinking":
                thinking_block = block
            elif block.type == "text":
                text_block = block

        # Validate thinking block
        assert thinking_block is not None, "Should have thinking content block"
        assert thinking_block.thinking == reasoning_text, (
            "Thinking content should match reasoning_content"
        )
        assert hasattr(thinking_block, "signature"), "Should have thinking signature"

        # Validate text block
        assert text_block is not None, "Should have text content block"
        assert text_block.text == "The answer is 4.", (
            "Text content should match response content"
        )

        print("✅ Reasoning content to thinking conversion test passed")

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

        assert len(messages) == 1
        user_message = messages[0]
        assert user_message["role"] == "user"
        assert isinstance(user_message["content"], list)
        assert len(user_message["content"]) == 3

        # Check text content
        text_parts = [
            part for part in user_message["content"] if part["type"] == "text"
        ]
        assert len(text_parts) == 2
        assert text_parts[0]["text"] == "Look at this image: "
        assert text_parts[1]["text"] == " What do you see?"

        # Check image content
        image_parts = [
            part for part in user_message["content"] if part["type"] == "image_url"
        ]
        assert len(image_parts) == 1
        assert (
            "data:image/jpeg;base64,fake_image_data"
            in image_parts[0]["image_url"]["url"]
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
        assert len(messages) == 2, f"Expected exactly 2 messages, got {len(messages)}"

        # First message should be the tool result
        tool_message = messages[0]
        assert tool_message["role"] == "tool", (
            f"First message should be tool role, got {tool_message['role']}"
        )
        assert tool_message["tool_call_id"] == "call_test_123", (
            "Tool call ID should match"
        )
        assert "Tool operation completed successfully" in tool_message["content"], (
            "Tool result content should match"
        )

        # Second message should be the user content
        user_message = messages[1]
        assert user_message["role"] == "user", (
            f"Second message should be user role, got {user_message['role']}"
        )
        assert user_message["content"] == "Thanks! Now let's try something else.", (
            f"User content should match"
        )

        print("✅ Tool result message ordering test passed")

    def test_thinking_content_conversion(self):
        """Test that thinking content is properly converted to text in message conversion."""
        print("🧪 Testing thinking content conversion...")

        # Test message with text + thinking (thinking should be converted to text)
        message_with_thinking = ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockText(type="text", text="Regular user message"),
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

        assert len(messages) == 1
        user_message = messages[0]
        assert user_message["role"] == "user"
        # Content should be a list with both text and converted thinking content
        assert isinstance(user_message["content"], list)
        assert len(user_message["content"]) == 2

        # First item should be the regular text
        assert user_message["content"][0]["type"] == "text"
        assert user_message["content"][0]["text"] == "Regular user message"

        # Second item should be the converted thinking content
        assert user_message["content"][1]["type"] == "text"
        assert user_message["content"][1]["text"] == "This is internal thinking"

        print("✅ Thinking content conversion test passed")

    def test_content_block_methods(self):
        """Test individual content block conversion methods."""
        print("🧪 Testing content block methods...")

        # Test text block
        text_block = ClaudeContentBlockText(type="text", text="Test text")
        text_result = text_block.to_openai()
        assert text_result == {"type": "text", "text": "Test text"}

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
        assert image_result["type"] == "image_url"
        assert (
            image_result["image_url"]["url"] == "data:image/png;base64,test_image_data"
        )

        # Test thinking block (should return text block)
        thinking_block = ClaudeContentBlockThinking(
            type="thinking", thinking="Internal thoughts"
        )
        thinking_result = thinking_block.to_openai()
        assert thinking_result["type"] == "text"
        assert thinking_result["text"] == "Internal thoughts"

        # Test tool use block
        tool_use_block = ClaudeContentBlockToolUse(
            type="tool_use",
            id="call_456",
            name="calculator",
            input={"expression": "2+2"},
        )
        tool_use_result = tool_use_block.to_openai()
        assert tool_use_result["id"] == "call_456"
        assert tool_use_result["function"]["name"] == "calculator"

        # Test tool result block
        tool_result_block = ClaudeContentBlockToolResult(
            type="tool_result", tool_use_id="call_456", content="4"
        )
        tool_result_result = tool_result_block.to_openai_message()
        assert tool_result_result["role"] == "tool"
        assert tool_result_result["tool_call_id"] == "call_456"
        assert tool_result_result["content"] == "4"

        print("✅ Content block methods test passed")

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
        assert len(messages) == 5, f"Expected 5 messages, got {len(messages)}"

        # Check message roles and order
        expected_roles = ["user", "assistant", "tool", "user", "assistant"]
        actual_roles = [msg["role"] for msg in messages]
        assert actual_roles == expected_roles, (
            f"Expected roles {expected_roles}, got {actual_roles}"
        )

        # Validate assistant message with tool calls
        assistant_msg = messages[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "I'll check the weather for you."
        assert "tool_calls" in assistant_msg
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"

        # Validate tool result
        tool_msg = messages[2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "toolu_weather_123"
        assert tool_msg["content"] == "Sunny, 75°F"

        # Validate follow-up user message
        followup_user_msg = messages[3]
        assert followup_user_msg["role"] == "user"
        assert followup_user_msg["content"] == "That's nice! What about tomorrow?"

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
        assert "messages" in result_enabled
        assert len(result_enabled["messages"]) == 1

        # Test with thinking disabled
        test_request_disabled = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Regular request...")],
            thinking=ClaudeThinkingConfigDisabled(type="disabled"),
        )

        result_disabled = test_request_disabled.to_openai_request()
        assert "messages" in result_disabled
        assert len(result_disabled["messages"]) == 1

        print("✅ Thinking configuration test passed")

    def test_message_to_openai_conversion(self):
        """Test message to_openai conversion methods."""
        print("🧪 Testing message to_openai conversion...")

        # Test simple text message conversion
        text_message = ClaudeMessage(role="user", content="Simple text")
        openai_messages = text_message.to_openai_messages()
        assert len(openai_messages) == 1
        assert openai_messages[0]["role"] == "user"
        assert openai_messages[0]["content"] == "Simple text"

        # Test mixed content message conversion
        mixed_message = ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockText(type="text", text="Hello "),
                ClaudeContentBlockText(type="text", text="world!"),
            ],
        )
        openai_mixed = mixed_message.to_openai_messages()
        assert len(openai_mixed) == 1
        assert openai_mixed[0]["role"] == "user"
        assert isinstance(openai_mixed[0]["content"], list)
        assert len(openai_mixed[0]["content"]) == 2
        assert openai_mixed[0]["content"][0]["type"] == "text"
        assert openai_mixed[0]["content"][0]["text"] == "Hello "
        assert openai_mixed[0]["content"][1]["text"] == "world!"

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
        assert len(openai_assistant) == 1
        assert openai_assistant[0]["role"] == "assistant"
        assert openai_assistant[0]["content"] == "Let me help you."
        assert "tool_calls" in openai_assistant[0]
        assert len(openai_assistant[0]["tool_calls"]) == 1
        assert openai_assistant[0]["tool_calls"][0]["id"] == "tool_123"
        assert openai_assistant[0]["tool_calls"][0]["function"]["name"] == "helper"

        # Test user message with tool result conversion (should split into multiple messages)
        user_message = ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockToolResult(
                    type="tool_result", tool_use_id="tool_123", content="Success"
                ),
                ClaudeContentBlockText(type="text", text="Thanks!")
            ],
        )
        openai_user = user_message.to_openai_messages()
        assert len(openai_user) == 2  # Should split into tool message + user message

        # First should be tool result message
        assert openai_user[0]["role"] == "tool"
        assert openai_user[0]["tool_call_id"] == "tool_123"
        assert openai_user[0]["content"] == "Success"

        # Second should be user text message
        assert openai_user[1]["role"] == "user"
        assert openai_user[1]["content"] == "Thanks!"

        print("✅ Message to_openai conversion test passed")

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
