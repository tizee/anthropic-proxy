#!/usr/bin/env python3
"""
Test script specifically for testing the OpenAI SDK type integration in conversion functions.
"""

import sys
import json
from typing import Any, Dict

# Import our conversion functions and models
from server import (
    convert_anthropic_to_openai_request,
    convert_openai_to_anthropic,
    _extract_system_content,
    _extract_message_content
)
from models import (
    MessagesRequest,
    Message,
    ContentBlockText,
    Tool,
    ThinkingConfigEnabled
)


def create_mock_openai_response(content: str, tool_calls=None, reasoning_content: str = ""):
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
            if reasoning_content:
                self.reasoning_content = reasoning_content

    class MockChoice:
        def __init__(self, content, tool_calls=None, reasoning_content=None):
            self.message = MockMessage(content, tool_calls, reasoning_content)
            self.finish_reason = 'tool_calls' if tool_calls else 'stop'

    class MockUsage:
        def __init__(self):
            self.prompt_tokens = 10
            self.completion_tokens = 20

    class MockOpenAIResponse:
        def __init__(self, content, tool_calls=None, reasoning_content=None):
            self.choices = [MockChoice(content, tool_calls, reasoning_content)]
            self.usage = MockUsage()

    # Create tool calls if provided
    mock_tool_calls = None
    if tool_calls:
        mock_tool_calls = [
            MockToolCall(f"call_{i}", call['name'], json.dumps(call['arguments']))
            for i, call in enumerate(tool_calls)
        ]

    return MockOpenAIResponse(content, mock_tool_calls, reasoning_content)


def test_basic_anthropic_to_openai():
    """Test basic message conversion from Anthropic to OpenAI format."""
    print("🧪 Testing basic Anthropic to OpenAI conversion...")

    test_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[
            Message(role='user', content='Hello, world!')
        ]
    )

    result = convert_anthropic_to_openai_request(test_request, 'gpt-4')

    # Validate structure
    assert 'model' in result
    assert 'messages' in result
    assert 'max_tokens' in result
    assert result['model'] == 'gpt-4'
    assert len(result['messages']) == 1
    assert result['messages'][0]['role'] == 'user'
    assert result['messages'][0]['content'] == 'Hello, world!'

    print("✅ Basic conversion test passed")
    return True


def test_system_message_conversion():
    """Test system message conversion."""
    print("🧪 Testing system message conversion...")

    test_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[Message(role='user', content='Hello!')],
        system='You are a helpful assistant.'
    )

    result = convert_anthropic_to_openai_request(test_request, 'gpt-4')

    # Should have system message first
    assert len(result['messages']) == 2
    assert result['messages'][0]['role'] == 'system'
    assert result['messages'][0]['content'] == 'You are a helpful assistant.'
    assert result['messages'][1]['role'] == 'user'
    assert result['messages'][1]['content'] == 'Hello!'

    print("✅ System message conversion test passed")
    return True


def test_tool_conversion():
    """Test tool conversion from Anthropic to OpenAI format."""
    print("🧪 Testing tool conversion...")

    calculator_tool = Tool(
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

    test_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[Message(role='user', content='What is 2+2?')],
        tools=[calculator_tool],
        tool_choice={"type": "auto"}
    )

    result = convert_anthropic_to_openai_request(test_request, 'gpt-4')

    # Validate tools conversion
    assert 'tools' in result
    assert len(result['tools']) == 1

    tool = result['tools'][0]
    assert tool['type'] == 'function'
    assert 'function' in tool
    assert tool['function']['name'] == 'calculator'
    assert tool['function']['description'] == 'Evaluate mathematical expressions'
    assert 'parameters' in tool['function']

    print("✅ Tool conversion test passed")
    return True


def test_openai_to_anthropic_basic():
    """Test basic OpenAI to Anthropic response conversion."""
    print("🧪 Testing basic OpenAI to Anthropic conversion...")

    # Create mock OpenAI response
    mock_response = create_mock_openai_response("Hello! How can I help you?")

    original_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[Message(role='user', content='Hello')]
    )

    result = convert_openai_to_anthropic(mock_response, original_request)

    # Validate structure
    assert result.role == 'assistant'
    assert len(result.content) >= 1
    assert result.content[0].type == 'text'
    assert result.content[0].text == 'Hello! How can I help you?'
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20

    print("✅ Basic OpenAI to Anthropic conversion test passed")
    return True


def test_openai_to_anthropic_with_tools():
    """Test OpenAI to Anthropic conversion with tool calls."""
    print("🧪 Testing OpenAI to Anthropic conversion with tools...")

    # Create mock OpenAI response with tool calls
    tool_calls = [
        {
            'name': 'calculator',
            'arguments': {'expression': '2+2'}
        }
    ]
    mock_response = create_mock_openai_response("I'll calculate that for you.", tool_calls)

    original_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[Message(role='user', content='What is 2+2?')]
    )

    result = convert_openai_to_anthropic(mock_response, original_request)

    # Should have text content and tool use
    assert len(result.content) >= 2

    # Find text and tool blocks
    text_block = None
    tool_block = None

    for block in result.content:
        if block.type == 'text':
            text_block = block
        elif block.type == 'tool_use':
            tool_block = block

    assert text_block is not None
    assert tool_block is not None
    assert text_block.text == "I'll calculate that for you."
    assert tool_block.name == 'calculator'
    assert tool_block.input == {'expression': '2+2'}

    print("✅ OpenAI to Anthropic tool conversion test passed")
    return True


def test_content_extraction_helpers():
    """Test the helper functions for content extraction."""
    print("🧪 Testing content extraction helpers...")

    # Test system content extraction
    system_result = _extract_system_content('System prompt')
    assert system_result == 'System prompt'

    # Test list system content
    system_list = [{'type': 'text', 'text': 'Hello'}, {'type': 'text', 'text': ' World'}]
    system_result_list = _extract_system_content(system_list)
    assert system_result_list == 'Hello World'

    # Test message content extraction
    content_result = _extract_message_content('User message')
    assert content_result == 'User message'

    # Test list content
    content_list = [
        {'type': 'text', 'text': 'Calculate this: '},
        {'type': 'text', 'text': '2+2'}
    ]
    content_result_list = _extract_message_content(content_list)
    assert content_result_list == 'Calculate this: 2+2'

    print("✅ Content extraction helpers test passed")
    return True


def test_thinking_integration():
    """Test thinking functionality in conversion."""
    print("🧪 Testing thinking integration...")

    test_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[Message(role='user', content='Think about this problem...')],
        thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=500)
    )

    # Test that thinking request converts properly
    result = convert_anthropic_to_openai_request(test_request, 'gpt-4')

    # Should still convert normally (thinking is handled separately)
    assert 'messages' in result
    assert len(result['messages']) == 1
    assert result['messages'][0]['content'] == 'Think about this problem...'

    print("✅ Thinking integration test passed")
    return True


def test_reasoning_content_to_thinking():
    """Test conversion of OpenAI reasoning_content to Claude thinking content block."""
    print("🧪 Testing reasoning_content to thinking conversion...")

    reasoning_text = "Let me think about this step by step. First, I need to understand what the user is asking for. They want to know about 2+2, which is a simple arithmetic problem. I should calculate this: 2+2=4."

    # Create mock OpenAI response with reasoning_content
    mock_response = create_mock_openai_response(
        content="The answer is 4.",
        reasoning_content=reasoning_text
    )

    original_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[Message(role='user', content='What is 2+2?')]
    )

    result = convert_openai_to_anthropic(mock_response, original_request)

    # Should have both thinking and text content blocks
    assert len(result.content) >= 2

    # Find thinking and text blocks
    thinking_block = None
    text_block = None

    for block in result.content:
        if block.type == 'thinking':
            thinking_block = block
        elif block.type == 'text':
            text_block = block

    # Validate thinking block
    assert thinking_block is not None, "Should have thinking content block"
    assert thinking_block.thinking == reasoning_text, "Thinking content should match reasoning_content"
    assert hasattr(thinking_block, 'signature'), "Should have thinking signature"

    # Validate text block
    assert text_block is not None, "Should have text content block"
    assert text_block.text == "The answer is 4.", "Text content should match response content"

    print("✅ Reasoning content to thinking conversion test passed")
    return True


def run_all_conversion_tests():
    """Run all conversion tests."""
    print("🚀 Running OpenAI SDK type integration tests...\n")

    tests = [
        test_basic_anthropic_to_openai,
        test_system_message_conversion,
        test_tool_conversion,
        test_openai_to_anthropic_basic,
        test_openai_to_anthropic_with_tools,
        test_content_extraction_helpers,
        test_thinking_integration,
        test_reasoning_content_to_thinking,
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
    success = run_all_conversion_tests()
    sys.exit(0 if success else 1)
