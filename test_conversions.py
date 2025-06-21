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
    _extract_system_content
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

    result = convert_anthropic_to_openai_request(test_request, 'deepseek-v3-250324')

    # Validate structure
    assert 'model' in result
    assert 'messages' in result
    assert 'max_tokens' in result
    assert result['model'] == 'deepseek-v3-250324'
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

    result = convert_anthropic_to_openai_request(test_request, 'deepseek-v3-250324')

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

    result = convert_anthropic_to_openai_request(test_request, 'deepseek-v3-250324')

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

    # Test message content extraction using new Message methods
    test_message_str = Message(role='user', content='User message')
    content_result = test_message_str.extract_text_content()
    assert content_result == 'User message'

    # Test list content
    content_list = [
        {'type': 'text', 'text': 'Calculate this: '},
        {'type': 'text', 'text': '2+2'}
    ]
    test_message_list = Message(role='user', content=content_list)
    content_result_list = test_message_list.extract_text_content()
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
    result = convert_anthropic_to_openai_request(test_request, 'deepseek-v3-250324')

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


def test_official_docs_tool_use_scenario():
    """Test multi-turn conversation with tool use based on official documentation examples."""
    print("🧪 Testing official docs tool use scenario...")
    
    from models import ContentBlockText, ContentBlockToolUse, ContentBlockToolResult
    
    # Test scenario based on Claude docs:
    # 1. User asks: "What's the S&P 500 at today?"
    # 2. Assistant responds with tool_use
    # 3. User provides tool_result
    # 4. Assistant responds with final answer
    
    test_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[
            # Initial user question
            Message(role='user', content="What's the S&P 500 at today?"),
            
            # Assistant response with tool use (Claude format)
            Message(role='assistant', content=[
                ContentBlockToolUse(
                    type="tool_use",
                    id="toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                    name="get_stock_price",
                    input={"ticker": "^GSPC"}
                )
            ]),
            
            # User message with tool result (Claude format)
            Message(role='user', content=[
                ContentBlockToolResult(
                    type="tool_result",
                    tool_use_id="toolu_01D7FLrfh4GYq7yT1ULFeyMV",
                    content="259.75 USD"
                )
            ]),
            
            # Assistant final response
            Message(role='assistant', content="The S&P 500 is currently at 259.75 USD.")
        ],
        tools=[
            Tool(
                name="get_stock_price",
                description="Get the current stock price for a given ticker symbol.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
                        }
                    },
                    "required": ["ticker"]
                }
            )
        ]
    )
    
    # Convert to OpenAI format
    result = convert_anthropic_to_openai_request(test_request, 'deepseek-v3-250324')
    
    # Validate conversion
    assert 'messages' in result
    assert 'tools' in result
    
    messages = result['messages']
    
    # Should have: user, assistant with tool_calls, tool with result, assistant final
    # Expected structure:
    # 1. user: "What's the S&P 500 at today?"
    # 2. assistant: tool_calls=[{...}], content=None
    # 3. tool: tool_call_id="toolu_01D7FLrfh4GYq7yT1ULFeyMV", content="259.75 USD" 
    # 4. assistant: "The S&P 500 is currently at 259.75 USD."
    
    # Find messages by role and characteristics
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
    tool_messages = [msg for msg in messages if msg['role'] == 'tool']
    
    # Validate user messages
    assert len(user_messages) == 1, f"Expected 1 user message, got {len(user_messages)}"
    assert user_messages[0]['content'] == "What's the S&P 500 at today?"
    
    # Validate assistant messages
    assert len(assistant_messages) == 2, f"Expected 2 assistant messages, got {len(assistant_messages)}"
    
    # First assistant message should have tool_calls
    first_assistant = assistant_messages[0]
    assert 'tool_calls' in first_assistant, "First assistant message should have tool_calls"
    assert len(first_assistant['tool_calls']) == 1, "Should have exactly 1 tool call"
    
    tool_call = first_assistant['tool_calls'][0]
    assert tool_call['id'] == "toolu_01D7FLrfh4GYq7yT1ULFeyMV", "Tool call ID should be preserved"
    assert tool_call['type'] == "function", "Tool call type should be function"
    assert tool_call['function']['name'] == "get_stock_price", "Function name should match"
    assert '"ticker":"^GSPC"' in tool_call['function']['arguments'], "Arguments should be JSON string with ticker"
    
    # Second assistant message should have final response
    second_assistant = assistant_messages[1]
    assert second_assistant['content'] == "The S&P 500 is currently at 259.75 USD."
    assert 'tool_calls' not in second_assistant, "Second assistant message should not have tool_calls"
    
    # Validate tool messages
    assert len(tool_messages) == 1, f"Expected 1 tool message, got {len(tool_messages)}"
    tool_message = tool_messages[0]
    assert tool_message['tool_call_id'] == "toolu_01D7FLrfh4GYq7yT1ULFeyMV", "Tool call ID should match"
    assert tool_message['content'] == "259.75 USD", "Tool result content should match"
    
    # Validate tools
    tools = result['tools']
    assert len(tools) == 1, f"Expected 1 tool, got {len(tools)}"
    tool = tools[0]
    assert tool['function']['name'] == "get_stock_price", "Tool name should match"
    assert 'ticker' in tool['function']['parameters']['properties'], "Tool should have ticker parameter"
    
    print("✅ Official docs tool use scenario test passed")
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
        test_official_docs_tool_use_scenario,
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
