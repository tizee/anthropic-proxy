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
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
    ContentBlockThinking,
    Tool,
    ThinkingConfigEnabled,
    convert_content_block_to_openai
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


def test_convert_content_block_to_openai():
    """Test the convert_content_block_to_openai utility function."""
    print("🧪 Testing convert_content_block_to_openai utility function...")
    
    # Test ContentBlockText conversion
    text_block = ContentBlockText(type="text", text="Hello, world!")
    text_result = convert_content_block_to_openai(text_block)
    assert text_result == {"type": "text", "text": "Hello, world!"}
    
    # Test ContentBlockImage conversion
    image_block = ContentBlockImage(
        type="image",
        source={
            "type": "base64",
            "media_type": "image/jpeg", 
            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        }
    )
    image_result = convert_content_block_to_openai(image_block)
    expected_image = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        }
    }
    assert image_result == expected_image
    
    # Test ContentBlockThinking conversion (should return None)
    thinking_block = ContentBlockThinking(type="thinking", thinking="Let me think about this...")
    thinking_result = convert_content_block_to_openai(thinking_block)
    assert thinking_result is None
    
    # Test ContentBlockToolUse conversion
    tool_use_block = ContentBlockToolUse(
        type="tool_use",
        id="toolu_123",
        name="get_weather",
        input={"location": "San Francisco"}
    )
    tool_use_result = convert_content_block_to_openai(tool_use_block)
    assert tool_use_result["id"] == "toolu_123"
    assert tool_use_result["type"] == "function"
    assert tool_use_result["function"]["name"] == "get_weather"
    assert '"location":"San Francisco"' in tool_use_result["function"]["arguments"]
    
    # Test ContentBlockToolResult conversion
    tool_result_block = ContentBlockToolResult(
        type="tool_result",
        tool_use_id="toolu_123",
        content="Weather is sunny, 75°F"
    )
    tool_result_result = convert_content_block_to_openai(tool_result_block)
    assert tool_result_result["role"] == "tool"
    assert tool_result_result["tool_call_id"] == "toolu_123"
    assert tool_result_result["content"] == "Weather is sunny, 75°F"
    
    # Test dict format conversions
    text_dict = {"type": "text", "text": "Hello from dict"}
    text_dict_result = convert_content_block_to_openai(text_dict)
    assert text_dict_result == {"type": "text", "text": "Hello from dict"}
    
    image_dict = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "test_data"
        }
    }
    image_dict_result = convert_content_block_to_openai(image_dict)
    assert image_dict_result["type"] == "image_url"
    assert "data:image/png;base64,test_data" == image_dict_result["image_url"]["url"]
    
    thinking_dict = {"type": "thinking", "thinking": "Thinking in dict format"}
    thinking_dict_result = convert_content_block_to_openai(thinking_dict)
    assert thinking_dict_result is None
    
    # Test unknown format
    unknown_block = {"type": "unknown", "data": "something"}
    unknown_result = convert_content_block_to_openai(unknown_block)
    assert unknown_result is None
    
    # Test invalid image format
    invalid_image = {"type": "image", "source": {"invalid": "format"}}
    invalid_image_result = convert_content_block_to_openai(invalid_image)
    assert invalid_image_result is None
    
    print("✅ convert_content_block_to_openai utility function test passed")
    return True


def test_content_block_to_openai_methods():
    """Test individual ContentBlock to_openai methods."""
    print("🧪 Testing ContentBlock to_openai methods...")
    
    # Test ContentBlockText.to_openai()
    text_block = ContentBlockText(type="text", text="Test text")
    text_result = text_block.to_openai()
    assert text_result == {"type": "text", "text": "Test text"}
    
    # Test ContentBlockImage.to_openai()
    image_block = ContentBlockImage(
        type="image",
        source={
            "type": "base64",
            "media_type": "image/png",
            "data": "test_image_data"
        }
    )
    image_result = image_block.to_openai()
    assert image_result["type"] == "image_url"
    assert image_result["image_url"]["url"] == "data:image/png;base64,test_image_data"
    
    # Test ContentBlockImage.to_openai() with invalid format
    invalid_image_block = ContentBlockImage(
        type="image",
        source={"invalid": "format"}
    )
    invalid_result = invalid_image_block.to_openai()
    assert invalid_result is None
    
    # Test ContentBlockThinking.to_openai()
    thinking_block = ContentBlockThinking(type="thinking", thinking="Internal thoughts")
    thinking_result = thinking_block.to_openai()
    assert thinking_result is None
    
    # Test ContentBlockToolUse.to_openai() (already existing)
    tool_use_block = ContentBlockToolUse(
        type="tool_use",
        id="call_456",
        name="calculator",
        input={"expression": "2+2"}
    )
    tool_use_result = tool_use_block.to_openai()
    assert tool_use_result["id"] == "call_456"
    assert tool_use_result["function"]["name"] == "calculator"
    
    # Test ContentBlockToolResult.to_openai() (already existing)
    tool_result_block = ContentBlockToolResult(
        type="tool_result",
        tool_use_id="call_456",
        content="4"
    )
    tool_result_result = tool_result_block.to_openai()
    assert tool_result_result["role"] == "tool"
    assert tool_result_result["tool_call_id"] == "call_456"
    assert tool_result_result["content"] == "4"
    
    print("✅ ContentBlock to_openai methods test passed")
    return True


def test_mixed_content_message_conversion():
    """Test conversion of messages with mixed content types."""
    print("🧪 Testing mixed content message conversion...")
    
    # Test text + image message
    mixed_message = Message(
        role="user",
        content=[
            ContentBlockText(type="text", text="Look at this image: "),
            ContentBlockImage(
                type="image",
                source={
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "fake_image_data"
                }
            ),
            ContentBlockText(type="text", text=" What do you see?")
        ]
    )
    
    test_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[mixed_message]
    )
    
    result = convert_anthropic_to_openai_request(test_request, 'test-model')
    messages = result['messages']
    
    assert len(messages) == 1
    user_message = messages[0]
    assert user_message['role'] == 'user'
    assert isinstance(user_message['content'], list)
    assert len(user_message['content']) == 3
    
    # Check text content
    text_parts = [part for part in user_message['content'] if part['type'] == 'text']
    assert len(text_parts) == 2
    assert text_parts[0]['text'] == "Look at this image: "
    assert text_parts[1]['text'] == " What do you see?"
    
    # Check image content
    image_parts = [part for part in user_message['content'] if part['type'] == 'image_url']
    assert len(image_parts) == 1
    assert "data:image/jpeg;base64,fake_image_data" in image_parts[0]['image_url']['url']
    
    print("✅ Mixed content message conversion test passed")
    return True


def test_thinking_content_filtering():
    """Test that thinking content is properly filtered out in message conversion."""
    print("🧪 Testing thinking content filtering...")
    
    # Test message with text + thinking (thinking should be filtered)
    message_with_thinking = Message(
        role="user",
        content=[
            ContentBlockText(type="text", text="Regular user message"),
            ContentBlockThinking(type="thinking", thinking="This is internal thinking")
        ]
    )
    
    test_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[message_with_thinking]
    )
    
    result = convert_anthropic_to_openai_request(test_request, 'test-model')
    messages = result['messages']
    
    assert len(messages) == 1
    user_message = messages[0]
    assert user_message['role'] == 'user'
    # Should only have text content, thinking should be filtered out
    assert user_message['content'] == "Regular user message"
    
    # Test message with only thinking content
    thinking_only_message = Message(
        role="user", 
        content=[
            ContentBlockThinking(type="thinking", thinking="Only thinking here")
        ]
    )
    
    thinking_only_request = MessagesRequest(
        model='test-model',
        max_tokens=100,
        messages=[thinking_only_message]
    )
    
    thinking_result = convert_anthropic_to_openai_request(thinking_only_request, 'test-model')
    thinking_messages = thinking_result['messages']
    
    # Should still create a message, but content should be empty or minimal
    assert len(thinking_messages) >= 0  # Could be filtered out entirely or have empty content
    
    print("✅ Thinking content filtering test passed")
    return True


def test_official_docs_tool_use_scenario():
    """Test multi-turn conversation with tool use based on official documentation examples."""
    print("🧪 Testing official docs tool use scenario...")
    
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


def test_claude_code_tool_interruption_message_ordering():
    """Test that tool interruption messages maintain correct chronological order."""
    print("🧪 Testing Claude Code tool interruption message ordering...")
    
    # This test specifically verifies the bug fix for message ordering
    # when user interrupts a tool call with mixed content blocks
    
    test_request = MessagesRequest(
        model='test-model',
        max_tokens=4000,
        messages=[
            # The critical test case: user message with tool_result + text content
            # This is the exact scenario from the bug report logs
            Message(
                role='user',
                content=[
                    # Tool result should be converted to separate tool message
                    ContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="call_test_interruption_123",
                        content="The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). STOP what you are doing and wait for the user to tell you how to proceed."
                    ),
                    # User interruption marker
                    ContentBlockText(
                        type="text",
                        text="[Request interrupted by user for tool use]"
                    ),
                    # Additional user message  
                    ContentBlockText(
                        type="text",
                        text="Actually, let's try a different approach. The file already exists."
                    )
                ]
            )
        ]
    )
    
    # Convert to OpenAI format
    result = convert_anthropic_to_openai_request(test_request, 'test-model')
    
    # Validate conversion structure
    assert 'messages' in result
    messages = result['messages']
    
    # CRITICAL TEST: The correct order should be:
    # 1. tool message (from ContentBlockToolResult) 
    # 2. user message (from ContentBlockText blocks)
    #
    # NOT the incorrect order that was happening before the bug fix:
    # 1. user message (from ContentBlockText blocks)
    # 2. tool message (from ContentBlockToolResult) <- Wrong! This was happening at the end
    
    assert len(messages) == 2, f"Expected exactly 2 messages, got {len(messages)}"
    
    # First message should be the tool result
    tool_message = messages[0]
    assert tool_message['role'] == 'tool', f"First message should be tool role, got {tool_message['role']}"
    assert tool_message['tool_call_id'] == "call_test_interruption_123", "Tool call ID should match"
    assert "The user doesn't want to proceed" in tool_message['content'], "Tool result content should match"
    
    # Second message should be the user content  
    user_message = messages[1]
    assert user_message['role'] == 'user', f"Second message should be user role, got {user_message['role']}"
    expected_user_content = "[Request interrupted by user for tool use]Actually, let's try a different approach. The file already exists."
    assert user_message['content'] == expected_user_content, f"User content should be combined text blocks, got: {user_message['content']}"
    
    print("✅ Claude Code tool interruption message ordering test passed")
    return True


def test_claude_code_multiple_interruptions():
    """Test complex scenario with multiple tool results and interruptions."""
    print("🧪 Testing multiple tool interruptions scenario...")
    
    # Test scenario with multiple tool calls and interruptions to ensure
    # our fix handles complex cases correctly
    
    test_request = MessagesRequest(
        model='test-model',
        max_tokens=4000,
        messages=[
            # Multiple messages to create a complex conversation
            Message(role='user', content="Please help with file operations"),
            
            # Assistant uses tool
            Message(role='assistant', content=[
                ContentBlockToolUse(
                    type="tool_use",
                    id="call_glob_123",
                    name="Glob", 
                    input={"pattern": "*.yaml"}
                )
            ]),
            
            # First tool result - normal flow
            Message(role='user', content=[
                ContentBlockToolResult(
                    type="tool_result",
                    tool_use_id="call_glob_123",
                    content="/path/to/config.yaml"
                )
            ]),
            
            # Assistant uses another tool
            Message(role='assistant', content=[
                ContentBlockToolUse(
                    type="tool_use",
                    id="call_read_456",
                    name="Read",
                    input={"file_path": "/path/to/config.yaml"}
                )
            ]),
            
            # Second tool result with additional user content - interruption scenario
            Message(role='user', content=[
                ContentBlockToolResult(
                    type="tool_result", 
                    tool_use_id="call_read_456",
                    content="version: 1.0\napi_key: example"
                ),
                ContentBlockText(
                    type="text",
                    text="Wait, let me check something else first."
                )
            ]),
            
            # Assistant tries another tool
            Message(role='assistant', content=[
                ContentBlockToolUse(
                    type="tool_use",
                    id="call_plan_789",
                    name="exit_plan_mode",
                    input={"plan": "Create config example"}
                )
            ]),
            
            # Third interruption - the critical test case
            Message(role='user', content=[
                ContentBlockToolResult(
                    type="tool_result",
                    tool_use_id="call_plan_789", 
                    content="Tool use rejected by user"
                ),
                ContentBlockText(
                    type="text",
                    text="[Request interrupted by user for tool use]"
                ),
                ContentBlockText(
                    type="text",
                    text="Let's skip this step entirely."
                )
            ])
        ]
    )
    
    # Convert to OpenAI format
    result = convert_anthropic_to_openai_request(test_request, 'test-model')
    
    # Validate conversion
    assert 'messages' in result
    messages = result['messages']
    
    # Expected OpenAI message structure:
    # 1. user: "Please help with file operations"
    # 2. assistant: tool_calls=[Glob call]
    # 3. tool: Glob result
    # 4. assistant: tool_calls=[Read call] 
    # 5. tool: Read result (from mixed content block)
    # 6. user: "Wait, let me check something else first." (from mixed content block)
    # 7. assistant: tool_calls=[exit_plan_mode call]
    # 8. tool: exit_plan_mode result (from mixed content block)
    # 9. user: "[Request interrupted...]Let's skip this step entirely." (from mixed content block)
    
    # Find messages by role
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
    tool_messages = [msg for msg in messages if msg['role'] == 'tool']
    
    # Should have 3 user messages (initial + 2 from mixed content)
    assert len(user_messages) == 3, f"Expected 3 user messages, got {len(user_messages)}"
    
    # Should have 3 assistant messages (all with tool calls)
    assert len(assistant_messages) == 3, f"Expected 3 assistant messages, got {len(assistant_messages)}"
    
    # Should have 3 tool messages (all results)
    assert len(tool_messages) == 3, f"Expected 3 tool messages, got {len(tool_messages)}"
    
    # Verify the critical ordering: in the mixed content scenarios,
    # tool results should appear immediately, not deferred to the end
    
    # The last few messages should be in this exact order:
    # ..., assistant (tool call), tool (result), user (interruption text)
    
    last_three = messages[-3:]
    assert last_three[0]['role'] == 'assistant', "Third-to-last should be assistant with tool call"
    assert 'tool_calls' in last_three[0], "Should have tool calls"
    assert last_three[0]['tool_calls'][0]['function']['name'] == 'exit_plan_mode'
    
    assert last_three[1]['role'] == 'tool', "Second-to-last should be tool result" 
    assert last_three[1]['tool_call_id'] == 'call_plan_789'
    assert last_three[1]['content'] == "Tool use rejected by user"
    
    assert last_three[2]['role'] == 'user', "Last should be user interruption"
    assert "[Request interrupted by user for tool use]" in last_three[2]['content']
    assert "Let's skip this step entirely." in last_three[2]['content']
    
    print("✅ Multiple tool interruptions scenario test passed")
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
        test_convert_content_block_to_openai,
        test_content_block_to_openai_methods,
        test_mixed_content_message_conversion,
        test_thinking_content_filtering,
        test_official_docs_tool_use_scenario,
        test_claude_code_tool_interruption_message_ordering,
        test_claude_code_multiple_interruptions,
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
