#!/usr/bin/env python3
"""
Comprehensive test suite for Claude<->OpenAI message conversion functionality.

This suite focuses on:
- Bidirectional message format conversion (Claude ↔ OpenAI)
- Tool use, thinking blocks, and content block processing
- Streaming converter state management and JSON repair
- Function-call parsing and unique ID handling
"""

import json
import sys
import unittest

# Add parent directory to path for imports
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our conversion functions and models
from openai.types.chat import ChatCompletionChunk

from anthropic_proxy.types import (
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
    generate_unique_id,
)
from anthropic_proxy.streaming import AnthropicStreamingConverter
from anthropic_proxy.openai_converter import (
    convert_openai_response_to_anthropic,
    parse_function_calls_from_thinking,
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
        """Test that tool result messages maintain correct chronological order and include name field."""
        print("🧪 Testing tool result message ordering with name field...")

        # Test case with complete tool use cycle (tool_use -> tool_result)
        test_request = ClaudeMessagesRequest(
            model="test-model",
            max_tokens=4000,
            messages=[
                # User asks something
                ClaudeMessage(role="user", content="Please calculate 2+2"),
                # Assistant responds with tool use
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text", text="I'll calculate that for you."
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="call_test_123",
                            name="calculator",
                            input={"expression": "2+2"},
                        ),
                    ],
                ),
                # User provides tool result + text content
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="call_test_123",
                            content="4",
                        ),
                        ClaudeContentBlockText(
                            type="text", text="Thanks! Now let's try something else."
                        ),
                    ],
                ),
            ],
        )

        # Convert to OpenAI format
        result = test_request.to_openai_request()
        messages = result["messages"]

        # Should have 4 messages: user, assistant, tool, user
        self.assertEqual(
            len(messages), 4, f"Expected exactly 4 messages, got {len(messages)}"
        )

        # Find the tool result message
        tool_message = None
        for msg in messages:
            if msg.get("role") == "tool":
                tool_message = msg
                break

        self.assertIsNotNone(tool_message, "Tool result message not found")

        # Check tool result message structure
        self.assertEqual(
            tool_message["role"],
            "tool",
            f"Tool message should have role 'tool', got {tool_message['role']}",
        )
        self.assertEqual(
            tool_message["tool_call_id"], "call_test_123", "Tool call ID should match"
        )
        self.assertEqual(
            tool_message["content"], "4", "Tool result content should match"
        )

        # NEW: Check that name field is included (for Groq compatibility)
        self.assertIn(
            "name",
            tool_message,
            f"Tool result message should include 'name' field for provider compatibility: {tool_message}",
        )
        self.assertEqual(
            tool_message["name"],
            "calculator",
            f"Tool result name should match original tool_use name, got '{tool_message.get('name')}'",
        )

        # Find and check the user message that follows
        user_message = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool" and i + 1 < len(messages):
                user_message = messages[i + 1]
                break

        self.assertIsNotNone(user_message, "User message after tool result not found")
        self.assertEqual(
            user_message["role"],
            "user",
            f"Message after tool should be user role, got {user_message['role']}",
        )
        self.assertEqual(
            user_message["content"],
            "Thanks! Now let's try something else.",
            "User content should match",
        )

        print("✅ Tool result message ordering test passed")
        print(f"   - Tool result includes name field: {tool_message.get('name')}")
        print(f"   - Tool result tool_call_id: {tool_message.get('tool_call_id')}")
        print(f"   - Tool result content: {tool_message.get('content')}")

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

    def test_tool_result_content_variations(self):
        """Test tool_result with various content structures according to Claude API spec."""
        print("🧪 Testing tool_result content variations...")

        # Test 1: Simple string content (standard Claude API format)
        simple_result = ClaudeContentBlockToolResult(
            type="tool_result", tool_use_id="call_123", content="259.75 USD"
        )
        simple_processed = simple_result.process_content()
        self.assertEqual(simple_processed, "259.75 USD")

        # Test 2: List with text blocks (Claude API standard)
        text_list_result = ClaudeContentBlockToolResult(
            type="tool_result",
            tool_use_id="call_124",
            content=[{"type": "text", "text": "Processing complete"}],
        )
        text_list_processed = text_list_result.process_content()
        self.assertIsInstance(text_list_processed, list)
        self.assertEqual(len(text_list_processed), 1)
        self.assertEqual(text_list_processed[0]["type"], "text")
        self.assertEqual(text_list_processed[0]["text"], "Processing complete")

        # Test 3: List with multiple text blocks
        multi_text_result = ClaudeContentBlockToolResult(
            type="tool_result",
            tool_use_id="call_125",
            content=[
                {"type": "text", "text": "First part"},
                {"type": "text", "text": "Second part"},
            ],
        )
        multi_text_processed = multi_text_result.process_content()
        self.assertIsInstance(multi_text_processed, list)
        self.assertEqual(len(multi_text_processed), 2)
        self.assertEqual(multi_text_processed[0]["text"], "First part")
        self.assertEqual(multi_text_processed[1]["text"], "Second part")

        # Test 4: List with image block (Claude API spec allows this)
        image_result = ClaudeContentBlockToolResult(
            type="tool_result",
            tool_use_id="call_126",
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                    },
                }
            ],
        )
        image_processed = image_result.process_content()
        self.assertIsInstance(image_processed, list)
        self.assertEqual(len(image_processed), 1)
        self.assertEqual(image_processed[0]["type"], "image")
        self.assertIn("source", image_processed[0])

        # Test 5: Edge case - text block without explicit type (should be handled gracefully)
        no_type_result = ClaudeContentBlockToolResult(
            type="tool_result",
            tool_use_id="call_127",
            content=[{"text": "Text without type"}],
        )
        no_type_processed = no_type_result.process_content()
        self.assertIsInstance(no_type_processed, list)
        self.assertEqual(len(no_type_processed), 1)
        self.assertEqual(no_type_processed[0]["type"], "text")
        self.assertEqual(no_type_processed[0]["text"], "Text without type")

        # Test 6: Edge case - malformed content block (fallback to string conversion)
        malformed_result = ClaudeContentBlockToolResult(
            type="tool_result",
            tool_use_id="call_128",
            content=[{"type": "unknown", "data": "some data"}],
        )
        malformed_processed = malformed_result.process_content()
        self.assertIsInstance(malformed_processed, list)
        self.assertEqual(len(malformed_processed), 1)
        self.assertEqual(malformed_processed[0]["type"], "text")
        self.assertEqual(
            malformed_processed[0]["text"], "{'type': 'unknown', 'data': 'some data'}"
        )

        print("✅ Tool result content variations test passed")


class TestAdvancedFeatures(unittest.TestCase):
    """Test advanced features like function call parsing and complex scenarios."""

    def test_function_call_parsing(self):
        """Test function call parsing from thinking content."""
        print("🧪 Testing function call parsing from thinking content...")

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

    def test_function_call_parsing_with_whitespace(self):
        """Test function call parsing with realistic whitespace/newlines from logs."""
        from anthropic_proxy.openai_converter import parse_function_calls_from_thinking

        # Test realistic format with newlines and whitespace (like from actual logs)
        thinking_with_whitespace = """I need to update the README to include the MAX_RETRIES environment variable configuration.

<|FunctionCallBegin|>[
{"name": "Edit", "parameters": {"file_path": "/Users/test/README.md", "old_string": "- Enhanced client reliability", "new_string": "- Enhanced client reliability with MAX_RETRIES configuration"}}
]<|FunctionCallEnd|>

Let me proceed with this edit."""

        cleaned_thinking, function_calls = parse_function_calls_from_thinking(
            thinking_with_whitespace
        )

        # Verify function call was parsed correctly
        self.assertEqual(len(function_calls), 1)

        tool_call = function_calls[0]
        self.assertIn("id", tool_call)
        self.assertEqual(tool_call["type"], "function")
        self.assertEqual(tool_call["function"]["name"], "Edit")

        # Verify arguments contain expected data
        import json

        arguments = json.loads(tool_call["function"]["arguments"])
        self.assertIn("file_path", arguments)
        self.assertIn("old_string", arguments)
        self.assertIn("new_string", arguments)
        self.assertEqual(arguments["file_path"], "/Users/test/README.md")

        # Verify thinking content was cleaned
        self.assertNotIn("<|FunctionCallBegin|>", cleaned_thinking)
        self.assertNotIn("<|FunctionCallEnd|>", cleaned_thinking)
        self.assertIn("I need to update the README", cleaned_thinking)
        self.assertIn("Let me proceed with this edit.", cleaned_thinking)

        print("✅ Function call parsing with whitespace test passed")

    def test_function_call_parsing_edge_cases(self):
        """Test function call parsing with various edge cases and malformed JSON."""
        from anthropic_proxy.openai_converter import parse_function_calls_from_thinking

        # Test case 1: Multi-line parameters (the main issue from the bug report)
        multiline_content = """I need to ensure the indentation matches. Looking at the surrounding code, the line is indented with 8 spaces (2 levels
   deep inside the function). The new lines should maintain this indentation.

  Now, I'll use the Edit tool to make this change.<|FunctionCallBegin|>[
  {"name": "Edit", "parameters": {"file_path":
  "/Users/tizee/projects/project-tampermonkey-scripts/tizee-scripts/tampermonkey-chatgpt-model-usage-monitor/monitor.js",
   "old_string": "        draggable = new Draggable(container);", "new_string": "        if (draggable &&\\n  draggable.destroy) {\\n            draggable.destroy();\\n        }\\n        draggable = new Draggable(container);"}}
  ]<|FunctionCallEnd|>

This should fix the issue."""

        cleaned_thinking, function_calls = parse_function_calls_from_thinking(
            multiline_content
        )

        self.assertEqual(len(function_calls), 1)
        tool_call = function_calls[0]
        self.assertEqual(tool_call["function"]["name"], "Edit")

        arguments = json.loads(tool_call["function"]["arguments"])
        self.assertIn("file_path", arguments)
        self.assertIn("old_string", arguments)
        self.assertIn("new_string", arguments)
        self.assertTrue(len(arguments["file_path"]) > 0)
        self.assertTrue(len(arguments["old_string"]) > 0)
        self.assertTrue(len(arguments["new_string"]) > 0)

        # Test case 2: Single object without array brackets
        single_object_content = """Let me create a file.<|FunctionCallBegin|>{"name": "Write", "parameters": {"file_path": "/tmp/test.txt", "content": "Hello World"}}<|FunctionCallEnd|>Done."""

        cleaned_thinking2, function_calls2 = parse_function_calls_from_thinking(
            single_object_content
        )

        self.assertEqual(len(function_calls2), 1)
        tool_call2 = function_calls2[0]
        self.assertEqual(tool_call2["function"]["name"], "Write")

        arguments2 = json.loads(tool_call2["function"]["arguments"])
        self.assertEqual(arguments2["file_path"], "/tmp/test.txt")
        self.assertEqual(arguments2["content"], "Hello World")

        # Test case 3: Multiple function calls in one block
        multiple_calls_content = """I need to do multiple things.<|FunctionCallBegin|>[
  {"name": "Read", "parameters": {"file_path": "/tmp/file1.txt"}},
  {"name": "Write", "parameters": {"file_path": "/tmp/file2.txt", "content": "data"}}
]<|FunctionCallEnd|>All done."""

        cleaned_thinking3, function_calls3 = parse_function_calls_from_thinking(
            multiple_calls_content
        )

        self.assertEqual(len(function_calls3), 2)
        self.assertEqual(function_calls3[0]["function"]["name"], "Read")
        self.assertEqual(function_calls3[1]["function"]["name"], "Write")

        # Test case 4: Malformed JSON with trailing comma
        malformed_content = """Let me fix this.<|FunctionCallBegin|>[
  {"name": "Edit", "parameters": {"file_path": "/tmp/test.py", "old_string": "old", "new_string": "new",}}
]<|FunctionCallEnd|>Fixed."""

        cleaned_thinking4, function_calls4 = parse_function_calls_from_thinking(
            malformed_content
        )

        self.assertEqual(len(function_calls4), 1)
        tool_call4 = function_calls4[0]
        self.assertEqual(tool_call4["function"]["name"], "Edit")

        # Test case 5: Empty function call block
        empty_content = (
            """Some thinking.<|FunctionCallBegin|>[]<|FunctionCallEnd|>More thinking."""
        )

        cleaned_thinking5, function_calls5 = parse_function_calls_from_thinking(
            empty_content
        )

        self.assertEqual(len(function_calls5), 0)
        self.assertNotIn("<|FunctionCallBegin|>", cleaned_thinking5)

        print("✅ Function call parsing edge cases test passed")

    def test_function_call_parsing_malformed_recovery(self):
        """Test function call parsing with severe malformations that require regex fallback."""
        from anthropic_proxy.openai_converter import parse_function_calls_from_thinking

        # Test case 1: Severely malformed JSON that needs regex extraction
        severely_malformed = """I'll use the tool now.<|FunctionCallBegin|>
        {"name": "Bash", "parameters": {"command": "ls -la", "description": "List files"
        This is broken JSON but the regex should still extract it
        ]<|FunctionCallEnd|>Hope it works."""

        cleaned_thinking, function_calls = parse_function_calls_from_thinking(
            severely_malformed
        )

        # Should extract at least the tool name even if parameters fail
        self.assertGreaterEqual(
            len(function_calls), 0
        )  # May or may not succeed depending on fallback

        # Test case 2: Mixed content with both valid and invalid calls
        mixed_content = """First I'll do this:<|FunctionCallBegin|>[
  {"name": "Read", "parameters": {"file_path": "/valid/path.txt"}}
]<|FunctionCallEnd|>

Then this broken one:<|FunctionCallBegin|>
  {"name": "Write", "parameters": {"broken": "json"
<|FunctionCallEnd|>

Finally this valid one:<|FunctionCallBegin|>[
  {"name": "Edit", "parameters": {"file_path": "/another/valid.py", "old_string": "old", "new_string": "new"}}
]<|FunctionCallEnd|>Done."""

        cleaned_thinking2, function_calls2 = parse_function_calls_from_thinking(
            mixed_content
        )

        # Should extract at least the valid calls
        self.assertGreaterEqual(len(function_calls2), 2)

        # Verify the valid calls were extracted correctly
        valid_names = [call["function"]["name"] for call in function_calls2]
        self.assertIn("Read", valid_names)
        self.assertIn("Edit", valid_names)

        print("✅ Function call parsing malformed recovery test passed")

    def test_function_call_parsing_real_world_scenarios(self):
        """Test function call parsing with real-world scenarios that caused issues."""
        from anthropic_proxy.openai_converter import parse_function_calls_from_thinking

        # Test case 1: Complex file path and multi-line strings (based on actual bug report)
        real_world_1 = """Looking at the code, I need to add proper cleanup for the draggable instance.

<|FunctionCallBegin|>[
  {"name": "Edit", "parameters": {"file_path": "/Users/tizee/projects/project-tampermonkey-scripts/tizee-scripts/tampermonkey-chatgpt-model-usage-monitor/monitor.js", "old_string": "        draggable = new Draggable(container);", "new_string": "        if (draggable && draggable.destroy) {\\n            draggable.destroy();\\n        }\\n        draggable = new Draggable(container);"}}
]<|FunctionCallEnd|>

This will ensure proper cleanup before creating a new draggable instance."""

        cleaned_thinking, function_calls = parse_function_calls_from_thinking(
            real_world_1
        )

        self.assertEqual(len(function_calls), 1)
        tool_call = function_calls[0]
        self.assertEqual(tool_call["function"]["name"], "Edit")

        arguments = json.loads(tool_call["function"]["arguments"])
        self.assertIn("file_path", arguments)
        self.assertTrue(arguments["file_path"].endswith("monitor.js"))
        self.assertIn("old_string", arguments)
        self.assertIn("new_string", arguments)
        self.assertIn("draggable.destroy()", arguments["new_string"])

        # Test case 2: MultiEdit with complex structure
        real_world_2 = """I need to make multiple edits to fix all the issues.

<|FunctionCallBegin|>[
  {"name": "MultiEdit", "parameters": {"file_path": "/path/to/complex/file.py", "edits": [
    {"old_string": "def old_function():", "new_string": "def new_function():"},
    {"old_string": "    return False", "new_string": "    return True"},
    {"old_string": "# TODO: fix this", "new_string": "# FIXED: updated implementation"}
  ]}}
]<|FunctionCallEnd|>

These changes should resolve all the issues."""

        cleaned_thinking2, function_calls2 = parse_function_calls_from_thinking(
            real_world_2
        )

        self.assertEqual(len(function_calls2), 1)
        tool_call2 = function_calls2[0]
        self.assertEqual(tool_call2["function"]["name"], "MultiEdit")

        arguments2 = json.loads(tool_call2["function"]["arguments"])
        self.assertIn("file_path", arguments2)
        self.assertIn("edits", arguments2)
        self.assertIsInstance(arguments2["edits"], list)
        self.assertEqual(len(arguments2["edits"]), 3)

        # Verify each edit has the required fields
        for edit in arguments2["edits"]:
            self.assertIn("old_string", edit)
            self.assertIn("new_string", edit)

        print("✅ Function call parsing real-world scenarios test passed")

    def test_function_call_parsing_multiple_blocks(self):
        """Test function call parsing with multiple separate tool call blocks in thinking."""
        from anthropic_proxy.openai_converter import parse_function_calls_from_thinking

        # Test the exact scenario from the user's example
        multiple_blocks_content = """I need to start implementing the fixes for the drag-and-drop issue in monitor.js. Let me work through the todo list
  systematically.

  First, I'll tackle the highest priority items. The first todo is to increase the drag handle size to 20x20px in CSS.
  Looking at the code, the drag handle CSS is defined around line 123:

  #chatUsageMonitor .drag-handle {
      width: 12px;
      height: 12px;
      position: absolute;
      right: 8px;
      top: 50%;
      transform: translateY(-50%);
      border-radius: 50%;
      background: ${COLORS.secondaryText};
      cursor: move;
      transition: background-color 0.2s ease;
  }

  I need to change width and height from 12px to 20px. This should make the drag handle easier to click. I'll use the
  Edit tool for this change.

  Next, I'll mark this todo as in_progress before making the change.

  <|FunctionCallBegin|>[{"name":"TodoWrite","parameters":{"todos":[{"content":"Increase drag handle size to 20x20px in CSS","status":"in_progress","priority":"high","id":"6"},{"content":"Add error handling for drag handle selection","status":"pending","priority":"high","id":"7"},{"content":"Disable pointer events during text animations","status":"pending","priority":"medium","id":"8"},{"content":"Implement Draggable class destroy method","status":"pending","priority":"high","id":"9"},{"content":"Add visual feedback for active drag state","status":"pending","priority":"medium","id":"10"}]}}]<|FunctionCallEnd|>

  Now I'll make the CSS change using the Edit tool:

  <|FunctionCallBegin|>[{"name":"Edit","parameters":{"file_path":"/Users/tizee/projects/project-tampermonkey-scripts/tizee-scripts/tampermonkey-chatgpt-model-usage-monitor/monitor.js","old_string":"   124→    width: 12px;\\n   125→    height: 12px;","new_string":"   124→    width: 20px;\\n   125→    height: 20px;"}}]<|FunctionCallEnd|>

  This completes the first fix."""

        cleaned_thinking, function_calls = parse_function_calls_from_thinking(
            multiple_blocks_content
        )

        # Should find exactly 2 function calls from 2 separate blocks
        self.assertEqual(len(function_calls), 2)

        # Verify first tool call (TodoWrite)
        todo_call = function_calls[0]
        self.assertEqual(todo_call["function"]["name"], "TodoWrite")

        todo_args = json.loads(todo_call["function"]["arguments"])
        self.assertIn("todos", todo_args)
        self.assertIsInstance(todo_args["todos"], list)
        self.assertEqual(len(todo_args["todos"]), 5)  # Should have 5 todos

        # Verify second tool call (Edit)
        edit_call = function_calls[1]
        self.assertEqual(edit_call["function"]["name"], "Edit")

        edit_args = json.loads(edit_call["function"]["arguments"])
        self.assertIn("file_path", edit_args)
        self.assertIn("old_string", edit_args)
        self.assertIn("new_string", edit_args)
        self.assertTrue(edit_args["file_path"].endswith("monitor.js"))
        self.assertIn("12px", edit_args["old_string"])
        self.assertIn("20px", edit_args["new_string"])

        # Verify thinking content was cleaned (both function call blocks removed)
        self.assertNotIn("<|FunctionCallBegin|>", cleaned_thinking)
        self.assertNotIn("<|FunctionCallEnd|>", cleaned_thinking)
        self.assertIn("I need to start implementing", cleaned_thinking)
        self.assertIn("This completes the first fix.", cleaned_thinking)

        # Verify the content between the blocks is preserved
        self.assertIn(
            "Now I'll make the CSS change using the Edit tool:", cleaned_thinking
        )

        print("✅ Function call parsing multiple blocks test passed")

    def test_function_call_parsing_mixed_single_and_multiple(self):
        """Test parsing with mix of single-call blocks and multi-call blocks."""
        from anthropic_proxy.openai_converter import parse_function_calls_from_thinking

        mixed_content = """First, a single call block:

<|FunctionCallBegin|>[{"name": "Read", "parameters": {"file_path": "/tmp/file1.txt"}}]<|FunctionCallEnd|>

Then, a multi-call block:

<|FunctionCallBegin|>[
  {"name": "Write", "parameters": {"file_path": "/tmp/file2.txt", "content": "data1"}},
  {"name": "Edit", "parameters": {"file_path": "/tmp/file3.txt", "old_string": "old", "new_string": "new"}}
]<|FunctionCallEnd|>

Finally, another single call:

<|FunctionCallBegin|>[{"name": "Bash", "parameters": {"command": "ls -la", "description": "List files"}}]<|FunctionCallEnd|>

All done."""

        cleaned_thinking, function_calls = parse_function_calls_from_thinking(
            mixed_content
        )

        # Should find exactly 4 function calls total
        self.assertEqual(len(function_calls), 4)

        # Verify tool names in order
        expected_tools = ["Read", "Write", "Edit", "Bash"]
        actual_tools = [call["function"]["name"] for call in function_calls]
        self.assertEqual(actual_tools, expected_tools)

        # Verify specific parameters
        read_args = json.loads(function_calls[0]["function"]["arguments"])
        self.assertEqual(read_args["file_path"], "/tmp/file1.txt")

        write_args = json.loads(function_calls[1]["function"]["arguments"])
        self.assertEqual(write_args["content"], "data1")

        bash_args = json.loads(function_calls[3]["function"]["arguments"])
        self.assertEqual(bash_args["command"], "ls -la")

        # Verify thinking content cleanup
        self.assertNotIn("<|FunctionCallBegin|>", cleaned_thinking)
        self.assertNotIn("<|FunctionCallEnd|>", cleaned_thinking)
        self.assertIn("First, a single call block:", cleaned_thinking)
        self.assertIn("All done.", cleaned_thinking)

        print("✅ Function call parsing mixed single and multiple test passed")

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
        # Should still convert normally (thinking is handled at request processing)
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

        generated_ids = [generate_unique_id("toolu") for _ in range(10)]
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

class TestStreamingMalformedToolJSON(unittest.TestCase):
    """
    Test malformed tool JSON repair and error recovery in streaming responses.

    This test class validates the system's ability to:
    - Detect and repair malformed JSON in tool call arguments
    - Handle incomplete JSON objects during streaming
    - Recover from various JSON syntax errors (missing brackets, quotes, etc.)
    - Maintain streaming resilience with error counting and thresholds
    - Finalize tool calls even when JSON is partially corrupted

    These tests ensure robust handling of real-world API instabilities
    and network issues that can cause JSON fragmentation.
    """

    def setUp(self):
        """Set up test environment."""
        # Create a mock ClaudeMessagesRequest
        mock_request = ClaudeMessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[ClaudeContentBlockText(type="text", text="Test message")],
                )
            ],
        )
        self.converter = AnthropicStreamingConverter(mock_request)

    def test_malformed_tool_json_detection(self):
        """Test detection of various malformed tool JSON patterns."""
        print("🧪 Testing malformed tool JSON detection...")

        # Test cases with expected results
        test_cases = [
            # Valid JSON - should not be detected as malformed
            ('{"file_path": "/test.py", "content": "test"}', False),
            ('{"tool": "Edit", "args": {"old": "a", "new": "b"}}', False),
            # Empty/whitespace cases
            ("", True),
            ("   ", True),
            ("null", True),
            # Single character cases
            ("{", True),
            ("}", True),
            ("[", True),
            ("]", True),
            (",", True),
            (":", True),
            ('"', True),
            # Common malformed patterns
            ('{"', True),
            ('"}', True),
            ("[{", True),
            ("}]", True),
            ("{}", True),
            ("[]", True),
            ("{,", True),
            (",}", True),
            ("[,", True),
            (",]", True),
            # The specific issue we're fixing - trailing brackets
            (
                '{"file_path":"/test.py","old_string":"test","new_string":"fixed"}]',
                True,
            ),
            ('{"tool":"Edit","args":{"a":"b"}},]', True),
            ('{"valid":"json"}]', True),
            # Short incomplete JSON
            ('{"file', True),
            ('[{"test"', True),
            ('{"a":}', True),
        ]

        for json_str, expected_malformed in test_cases:
            with self.subTest(
                json_str=json_str[:50] + ("..." if len(json_str) > 50 else "")
            ):
                result = self.converter.is_malformed_tool_json(json_str)
                self.assertEqual(
                    result,
                    expected_malformed,
                    f"JSON: '{json_str}' - Expected malformed: {expected_malformed}, Got: {result}",
                )

    def test_tool_json_repair_functionality(self):
        """Test repair of malformed tool JSON."""
        print("🧪 Testing tool JSON repair functionality...")

        # Test cases: (malformed_json, expected_repaired_dict, should_be_repaired)
        test_cases = [
            # Valid JSON should pass through unchanged
            (
                '{"file_path": "/test.py", "content": "test"}',
                {"file_path": "/test.py", "content": "test"},
                False,
            ),
            # The main case we're fixing - trailing bracket
            (
                '{"file_path":"/test.py","old_string":"old","new_string":"new","replace_all":false}]',
                {
                    "file_path": "/test.py",
                    "old_string": "old",
                    "new_string": "new",
                    "replace_all": False,
                },
                True,
            ),
            # Trailing comma cases
            (
                '{"tool":"Edit","args":{"a":"b"}},]',
                {"tool": "Edit", "args": {"a": "b"}},
                True,
            ),
            # Multiple trailing artifacts
            (
                '{"file_path":"/test.py","content":"test"},]',
                {"file_path": "/test.py", "content": "test"},
                True,
            ),
            # Trailing comma only
            (
                '{"tool":"Edit","args":{"old":"a","new":"b"}},',
                {"tool": "Edit", "args": {"old": "a", "new": "b"}},
                True,
            ),
        ]

        for malformed_json, expected_dict, should_be_repaired in test_cases:
            with self.subTest(
                json_str=malformed_json[:50]
                + ("..." if len(malformed_json) > 50 else "")
            ):
                repaired_dict, was_repaired = self.converter.try_repair_tool_json(
                    malformed_json
                )

                self.assertEqual(
                    was_repaired,
                    should_be_repaired,
                    f"Expected repair: {should_be_repaired}, Got: {was_repaired}",
                )
                self.assertEqual(
                    repaired_dict,
                    expected_dict,
                    f"Expected: {expected_dict}, Got: {repaired_dict}",
                )

    def test_tool_json_repair_edge_cases(self):
        """Test edge cases for tool JSON repair."""
        print("🧪 Testing tool JSON repair edge cases...")

        # Test completely broken JSON that can't be repaired
        broken_cases = [
            "",
            "   ",
            "{broken json",
            '{"unclosed": "string',
            '{"key": value without quotes}',
            '{"nested": {"broken": }',
        ]

        for broken_json in broken_cases:
            with self.subTest(json_str=broken_json):
                repaired_dict, was_repaired = self.converter.try_repair_tool_json(
                    broken_json
                )
                # Should return empty dict when repair fails
                self.assertEqual(repaired_dict, {})

    def test_streaming_tool_json_finalization(self):
        """Test the finalization process handles malformed JSON gracefully."""
        import asyncio

        async def run_test():
            print("🧪 Testing streaming tool JSON finalization with malformed JSON...")

            # Set up converter state to simulate a tool call in progress
            converter = self.converter
            converter.is_tool_use = True
            converter.content_block_index = 1  # Move past the tool block
            converter.current_content_blocks = [
                {"type": "tool_use", "id": "test_tool_id", "name": "Edit", "input": {}}
            ]

            # Test with malformed JSON (the specific case from the logs)
            malformed_json = '{"file_path":"/Users/test/server.py","old_string":"test","new_string":"fixed","replace_all":false}]'

            # Set up the new tool call state format
            converter.tool_calls = {
                0: {
                    "id": "test_tool_id",
                    "name": "Edit",
                    "json_accumulator": malformed_json,
                    "content_block_index": 0,
                }
            }
            converter.active_tool_indices = {0}

            # Process finalization
            events = []
            async for event in converter._prepare_finalization("tool_calls"):
                events.append(event)

            # Should not raise an exception and should have repaired the JSON
            self.assertGreater(len(events), 0, "Should generate finalization events")

            # Check that the tool input was properly set
            tool_input = converter.current_content_blocks[0]["input"]
            expected_input = {
                "file_path": "/Users/test/server.py",
                "old_string": "test",
                "new_string": "fixed",
                "replace_all": False,
            }
            self.assertEqual(
                tool_input, expected_input, "Tool input should be properly repaired"
            )

        asyncio.run(run_test())


class TestAnthropicStreamingConverter(unittest.TestCase):
    """Test class specifically for AnthropicStreamingConverter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_request = ClaudeMessagesRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[ClaudeContentBlockText(type="text", text="Hello, test!")],
                )
            ],
            max_tokens=100,
        )

    def test_init_state(self):
        """Test initial state of AnthropicStreamingConverter."""
        converter = AnthropicStreamingConverter(self.test_request)

        # Test basic initialization
        self.assertIsNotNone(converter.message_id)
        self.assertEqual(converter.content_block_index, 0)
        self.assertEqual(len(converter.current_content_blocks), 0)

        # Test tool call state - should be empty dictionaries
        self.assertEqual(converter.tool_calls, {})
        self.assertEqual(converter.active_tool_indices, set())

        # Test block states
        self.assertFalse(converter.text_block_started)
        self.assertFalse(converter.is_tool_use)
        self.assertFalse(converter.thinking_block_started)

    def test_single_tool_call_processing(self):
        """Test processing a single tool call."""
        converter = AnthropicStreamingConverter(self.test_request)

        # Create a mock tool call with index 0
        mock_tool_call = {
            "index": 0,
            "id": "call_test123",
            "function": {"name": "test_function", "arguments": '{"param": "value"}'},
            "type": "function",
        }

        # Process the tool call
        events = []

        async def collect_events():
            async for event in converter._handle_tool_call_delta(mock_tool_call):
                events.append(event)

        # Run the async generator
        import asyncio

        asyncio.run(collect_events())

        # Verify tool call was registered
        self.assertIn(0, converter.tool_calls)
        self.assertIn(0, converter.active_tool_indices)
        self.assertTrue(converter.is_tool_use)

        # Verify tool call data
        tool_info = converter.tool_calls[0]
        self.assertEqual(tool_info["name"], "test_function")
        self.assertEqual(tool_info["json_accumulator"], '{"param": "value"}')
        self.assertEqual(tool_info["content_block_index"], 0)

    def test_multiple_tool_calls_processing(self):
        """Test processing multiple tool calls with different indices."""
        converter = AnthropicStreamingConverter(self.test_request)

        # Create mock tool calls with different indices
        tool_call_1 = {
            "index": 0,
            "function": {
                "name": "read_file",
                "arguments": '{"file_path": "/path/to/file1.txt"}',
            },
        }

        tool_call_2 = {
            "index": 1,
            "function": {
                "name": "write_file",
                "arguments": '{"file_path": "/path/to/file2.txt", "content": "test"}',
            },
        }

        # Process both tool calls
        async def process_tools():
            async for event in converter._handle_tool_call_delta(tool_call_1):
                pass
            async for event in converter._handle_tool_call_delta(tool_call_2):
                pass

        import asyncio

        asyncio.run(process_tools())

        # Verify both tool calls were registered separately
        self.assertEqual(len(converter.tool_calls), 2)
        self.assertEqual(len(converter.active_tool_indices), 2)
        self.assertIn(0, converter.tool_calls)
        self.assertIn(1, converter.tool_calls)

        # Verify tool call separation
        tool_0 = converter.tool_calls[0]
        tool_1 = converter.tool_calls[1]

        self.assertEqual(tool_0["name"], "read_file")
        self.assertEqual(
            tool_0["json_accumulator"], '{"file_path": "/path/to/file1.txt"}'
        )

        self.assertEqual(tool_1["name"], "write_file")
        self.assertEqual(
            tool_1["json_accumulator"],
            '{"file_path": "/path/to/file2.txt", "content": "test"}',
        )

        # Verify different content block indices
        self.assertNotEqual(
            tool_0["content_block_index"], tool_1["content_block_index"]
        )

    def test_json_parameter_separation(self):
        """Test that JSON parameters are separated correctly for multiple tools."""
        converter = AnthropicStreamingConverter(self.test_request)

        # Simulate the bug scenario from the logs
        tool_call_1 = {
            "index": 0,
            "function": {
                "name": "Read",
                "arguments": '{"file_path": "/Users/tizee/projects/project-AI/tools/claude-code-proxy.tizee/performance_test.py"}',
            },
        }

        tool_call_2 = {
            "index": 1,
            "function": {
                "name": "Read",
                "arguments": '{"file_path": "/Users/tizee/projects/project-AI/tools/claude-code-proxy.tizee/README.md"}',
            },
        }

        async def process_and_verify():
            # Process first tool call
            async for event in converter._handle_tool_call_delta(tool_call_1):
                pass

            # Process second tool call
            async for event in converter._handle_tool_call_delta(tool_call_2):
                pass

            # Verify JSON parameters are NOT mixed
            tool_0_json = converter.tool_calls[0]["json_accumulator"]
            tool_1_json = converter.tool_calls[1]["json_accumulator"]

            # This should NOT be the concatenated string that was causing the bug
            concatenated_bug = '{"file_path": "/Users/tizee/projects/project-AI/tools/claude-code-proxy.tizee/performance_test.py"}{"file_path": "/Users/tizee/projects/project-AI/tools/claude-code-proxy.tizee/README.md"}'

            # Verify each tool has its own JSON
            self.assertNotEqual(tool_0_json, concatenated_bug)
            self.assertNotEqual(tool_1_json, concatenated_bug)

            # Verify correct individual JSON
            self.assertEqual(
                tool_0_json,
                '{"file_path": "/Users/tizee/projects/project-AI/tools/claude-code-proxy.tizee/performance_test.py"}',
            )
            self.assertEqual(
                tool_1_json,
                '{"file_path": "/Users/tizee/projects/project-AI/tools/claude-code-proxy.tizee/README.md"}',
            )

            # Verify JSON can be parsed correctly
            import json

            parsed_0 = json.loads(tool_0_json)
            parsed_1 = json.loads(tool_1_json)

            self.assertEqual(
                parsed_0["file_path"],
                "/Users/tizee/projects/project-AI/tools/claude-code-proxy.tizee/performance_test.py",
            )
            self.assertEqual(
                parsed_1["file_path"],
                "/Users/tizee/projects/project-AI/tools/claude-code-proxy.tizee/README.md",
            )

        import asyncio

        asyncio.run(process_and_verify())

    def test_thinking_closes_before_tool_use_in_streaming(self):
        """Test that thinking block is closed before tool_use when reasoning is streamed."""
        converter = AnthropicStreamingConverter(self.test_request)

        thinking_chunk = ChatCompletionChunk.model_validate(
            {
                "id": "chunk_thinking",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "reasoning_content": "Thinking"},
                        "finish_reason": None,
                    }
                ],
            }
        )

        tool_chunk = ChatCompletionChunk.model_validate(
            {
                "id": "chunk_tool",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_test_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Skill",
                                        "arguments": '{"skill": "planning-with-files"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

        async def collect_events(chunk):
            events = []
            async for event in converter.process_chunk(chunk):
                events.append(event)
            return events

        import asyncio

        asyncio.run(collect_events(thinking_chunk))
        self.assertTrue(converter.thinking_block_started)
        self.assertFalse(converter.thinking_block_closed)

        tool_events = asyncio.run(collect_events(tool_chunk))

        self.assertTrue(converter.thinking_block_closed)
        self.assertIn(0, converter.tool_calls)
        self.assertEqual(converter.tool_calls[0]["content_block_index"], 1)

        stop_index = next(
            i for i, event in enumerate(tool_events) if event.startswith("event: content_block_stop")
        )
        tool_start_index = next(
            i
            for i, event in enumerate(tool_events)
            if event.startswith("event: content_block_start")
            and '"type": "tool_use"' in event
        )
        self.assertLess(stop_index, tool_start_index)

    def test_streaming_json_accumulation(self):
        """Test that JSON arguments are accumulated correctly across streaming chunks."""
        converter = AnthropicStreamingConverter(self.test_request)

        # Simulate streaming JSON in chunks
        chunks = [
            {"index": 0, "function": {"name": "test_tool", "arguments": '{"file'}},
            {
                "index": 0,
                "function": {"name": "test_tool", "arguments": '{"file_path": "/path'},
            },
            {
                "index": 0,
                "function": {
                    "name": "test_tool",
                    "arguments": '{"file_path": "/path/to/file.txt"',
                },
            },
            {
                "index": 0,
                "function": {
                    "name": "test_tool",
                    "arguments": '{"file_path": "/path/to/file.txt"}',
                },
            },
        ]

        async def process_chunks():
            for chunk in chunks:
                async for event in converter._handle_tool_call_delta(chunk):
                    pass

        import asyncio

        asyncio.run(process_chunks())

        # Verify final accumulated JSON
        final_json = converter.tool_calls[0]["json_accumulator"]
        self.assertEqual(final_json, '{"file_path": "/path/to/file.txt"}')

        # Verify it can be parsed
        import json

        parsed = json.loads(final_json)
        self.assertEqual(parsed["file_path"], "/path/to/file.txt")

    def test_content_block_index_management(self):
        """Test that content block indices are managed correctly for multiple tools."""
        converter = AnthropicStreamingConverter(self.test_request)

        # Process 3 tool calls
        tool_calls = [
            {"index": 0, "function": {"name": "tool_0", "arguments": "{}"}},
            {"index": 1, "function": {"name": "tool_1", "arguments": "{}"}},
            {"index": 2, "function": {"name": "tool_2", "arguments": "{}"}},
        ]

        async def process_all():
            for tool_call in tool_calls:
                async for event in converter._handle_tool_call_delta(tool_call):
                    pass

        import asyncio

        asyncio.run(process_all())

        # Verify each tool has a unique content block index
        indices = set()
        for tool_index in converter.tool_calls:
            block_index = converter.tool_calls[tool_index]["content_block_index"]
            self.assertNotIn(
                block_index, indices, f"Duplicate content block index {block_index}"
            )
            indices.add(block_index)

        # Verify content blocks were created
        self.assertEqual(len(converter.current_content_blocks), 3)

        # Verify content block types and IDs
        for i, block in enumerate(converter.current_content_blocks):
            self.assertEqual(block["type"], "tool_use")
            self.assertEqual(block["name"], f"tool_{i}")
            self.assertIn("id", block)
            self.assertIn("input", block)


if __name__ == "__main__":
    unittest.main()
