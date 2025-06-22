#!/usr/bin/env python3
"""
Comprehensive test suite for Claude-on-OpenAI Proxy.

This script provides tests for both streaming and non-streaming requests,
with various scenarios including tool use, multi-turn conversations,
and content blocks.

Usage:
  python tests.py                                       # Run all tests
  python tests.py --no-streaming                        # Skip streaming tests
  python tests.py --streaming-only                      # Run only streaming tests
  python tests.py --simple                              # Run only simple tests (no tools)
  python tests.py --tools-only                          # Run only tool-related tests
  python tests.py --thinking-only                       # Run only thinking-related tests
  python tests.py --test <test_name>                    # Run only a specific test (e.g. calculator)
  python tests.py --test <test_name> --streaming-only   # Run only streaming version of a test
  python tests.py --list-tests                          # List all available tests
  python tests.py --model <model_id>                    # Use a custom model from custom_models.yaml
  python tests.py --model <model_id> --compare          # Use custom model and compare with official one
  python tests.py --test <test_name> --model <model_id> # Test a custom model on a specific test
  python tests.py --test <test_name> --model <model_id> --compare # Combine all three flags
"""

import os
import json
import time
import httpx
import argparse
import asyncio
import sys
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dotenv import load_dotenv
from pydantic import BaseModel

# Import Pydantic models from models.py (separate from server.py to avoid starting the server)
from models import (
    ClaudeToolChoice,
    Tool,
    MessagesRequest,
    ThinkingConfigEnabled,
    ThinkingConfigDisabled,
    ContentBlockText,
    ContentBlockToolUse,
    ContentBlockToolResult,
    Message,
    SystemContent,
    ToolChoiceAuto,
    ToolChoiceAny,
)

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
PROXY_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # Using same key for proxy
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
PROXY_API_URL = "http://127.0.0.1:8082/v1/messages"
PROXY_TEST_API_URL = "http://127.0.0.1:8082/v1/messages/test_conversion"
ANTHROPIC_VERSION = "2023-06-01"
MODEL_THINKING = "claude-3-7-sonnet-20250219"
MODEL = "claude-3-haiku-20240307"  # Change to your preferred model

# Headers
anthropic_headers = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

tool_choice_required = ToolChoiceAny(type="any")
tool_choice_auto = ToolChoiceAuto(type="auto")

proxy_headers = {
    "x-api-key": PROXY_API_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

# Tool definitions using Pydantic models
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

# Gemini-incompatible tool - contains fields that Gemini doesn't support
gemini_incompatible_tool = Tool(
    name="complex_data_processor",
    description="Process complex data with advanced validation",
    input_schema={
        "type": "object",
        "additionalProperties": False,  # Gemini doesn't support this
        "title": "Complex Data Processor Input",  # Gemini doesn't support this
        "$schema": "http://json-schema.org/draft-07/schema#",  # Gemini doesn't support this
        "properties": {
            "data": {
                "type": "string",
                "description": "The data to process",
                "default": "empty",  # Gemini doesn't support default values
                "examples": [
                    "sample data",
                    "test input",
                ],  # Gemini doesn't support examples
            },
            "validation_level": {
                "type": "string",
                "enum": ["strict", "normal", "lenient"],
                "description": "Validation strictness level",
            },
            "config": {
                "type": "object",
                "additionalProperties": True,  # Gemini doesn't support this
                "properties": {
                    "timeout": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 3600,
                        "multipleOf": 5,  # This should be kept as it's simple
                    },
                    "advanced_options": {
                        "allOf": [  # Gemini doesn't support allOf
                            {"type": "object"},
                            {"properties": {"debug": {"type": "boolean"}}},
                        ]
                    },
                },
                "patternProperties": {  # Gemini doesn't support this
                    "^opt_": {"type": "string"}
                },
            },
        },
        "required": ["data", "validation_level"],
        "dependencies": {  # Gemini doesn't support this
            "validation_level": ["config"]
        },
    },
)

weather_tool = Tool(
    name="weather",
    description="Get weather information for a location",
    input_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location to get weather for",
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units",
            },
        },
        "required": ["location"],
    },
)

search_tool = Tool(
    name="search",
    description="Search for information on the web",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    },
)

# Claude Code tools for testing
read_tool = Tool(
    name="Read",
    description="Reads a file from the local filesystem",
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to read",
            }
        },
        "required": ["file_path"],
    },
)

bash_tool = Tool(
    name="Bash",
    description="Executes a bash command",
    input_schema={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to execute"},
            "description": {
                "type": "string",
                "description": "Clear description of what this command does",
            },
        },
        "required": ["command"],
    },
)

ls_tool = Tool(
    name="LS",
    description="Lists files and directories in a given path",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The absolute path to the directory to list",
            }
        },
        "required": ["path"],
    },
)

grep_tool = Tool(
    name="Grep",
    description="Fast content search tool using regular expressions",
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regular expression pattern to search for",
            },
            "path": {"type": "string", "description": "The directory to search in"},
        },
        "required": ["pattern"],
    },
)

glob_tool = Tool(
    name="Glob",
    description="Fast file pattern matching tool",
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The glob pattern to match files against",
            }
        },
        "required": ["pattern"],
    },
)

todo_write_tool = Tool(
    name="TodoWrite",
    description="Writes todo items to the todo list",
    input_schema={
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "minLength": 1},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "id": {"type": "string"},
                    },
                    "required": ["content", "status", "priority", "id"],
                },
                "description": "The updated todo list",
            }
        },
        "required": ["todos"],
    },
)

todo_read_tool = Tool(
    name="TodoRead",
    description="Reads the current todo list",
    input_schema={"type": "object", "properties": {}},
)

exit_plan_mode_tool = Tool(
    name="exit_plan_mode",
    description="Use this tool when you are in plan mode and have finished presenting your plan and are ready to code. This will prompt the user to exit plan mode.",
    input_schema={
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "The plan you came up with, that you want to run by the user for approval. Supports markdown. The plan should be pretty concise."
            }
        },
        "required": ["plan"]
    },
)

# Define behavioral difference tests that should warn instead of fail
BEHAVIORAL_DIFFERENCE_TESTS = {
    "multi_turn",
    "thinking_with_tools",
    "thinking_with_tools_stream",
    "calculator",
    "calculator_stream",
    "content_blocks",
    "multi_tool",
    "todo_write",
    "todo_read",
    "todo_write_stream",
    "todo_read_stream",
    "gemini_tool_test",
    "gemini_tool_test_stream",
    "gemini_incompatible_schema_test",
    "gemini_incompatible_schema_test_stream",
    "deepseek_thinking_tools",
    "deepseek_thinking_tools_stream",
    "claude_code_read_test_stream",
    "claude_code_bash_test_stream",
    "claude_code_ls_test_stream",
    "claude_code_grep_test_stream",
    "claude_code_glob_test_stream",
    "claude_code_todowrite_test_stream",
    "claude_code_todoread_test_stream",
    "claude_code_interruption_test",
    "claude_code_interruption_only_test",
    "deepseek_task_summarization_stream",
}

# Test scenarios using MessagesRequest Pydantic models
TEST_SCENARIOS = {
    # Simple text response
    "simple": MessagesRequest(
        model=MODEL,
        max_tokens=1025,
        messages=[
            Message(
                role="user",
                content="Hello, world! Can you tell me about Paris in 2-3 sentences?",
            )
        ],
    ),
    # Basic tool use
    "calculator": MessagesRequest(
        model=MODEL,
        max_tokens=1025,
        messages=[Message(role="user", content="What is 135 + 7.5 divided by 2.5?")],
        tools=[calculator_tool],
        tool_choice=tool_choice_required,
    ),
    # Todo tools
    "todo_write": MessagesRequest(
        model=MODEL,
        max_tokens=1025,
        messages=[
            Message(
                role="user",
                content="Create a todo list: 1. Buy milk, 2. Pay bills, 3. Call mom",
            )
        ],
        tools=[todo_write_tool],
        tool_choice=tool_choice_required,
    ),
    "todo_read": MessagesRequest(
        model=MODEL,
        max_tokens=1025,
        messages=[Message(role="user", content="What's on my todo list?")],
        tools=[todo_read_tool],
        tool_choice=tool_choice_required,
    ),
    # Multiple tools
    "multi_tool": MessagesRequest(
        model=MODEL,
        max_tokens=1025,
        temperature=0.7,
        top_p=0.95,
        system="You are a helpful assistant that uses tools when appropriate. Be concise and precise.",
        messages=[
            Message(
                role="user",
                content="I'm planning a trip to New York next week. What's the weather like and what are some interesting places to visit?",
            )
        ],
        tools=[weather_tool, search_tool],
        tool_choice=tool_choice_required,
    ),
    # Multi-turn conversation
    "multi_turn": MessagesRequest(
        model=MODEL,
        max_tokens=1025,
        messages=[
            Message(
                role="user",
                content="Let's do some math. What is 240 divided by 8?",
            ),
            Message(
                role="assistant",
                content="To calculate 240 divided by 8, I'll perform the division:\n\n240 ÷ 8 = 30\n\nSo the result is 30.",
            ),
            Message(
                role="user",
                content="Now multiply that by 4 and tell me the result.",
            ),
        ],
        tools=[calculator_tool],
        tool_choice=tool_choice_required,
    ),
    # Content blocks
    "content_blocks": MessagesRequest(
        model=MODEL,
        max_tokens=1025,
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockText(
                        type="text",
                        text="I need to know the weather in Los Angeles and calculate 75.5 / 5. Can you help with both?",
                    )
                ],
            )
        ],
        tools=[calculator_tool, weather_tool],
        tool_choice=tool_choice_required,
    ),
    # Simple streaming test
    "simple_stream": MessagesRequest(
        model=MODEL,
        max_tokens=100,
        stream=True,
        messages=[
            Message(role="user", content="Count from 1 to 5, with one number per line.")
        ],
    ),
    # Tool use with streaming
    "calculator_stream": MessagesRequest(
        model=MODEL,
        max_tokens=1025,
        stream=True,
        messages=[Message(role="user", content="What is 135 + 17.5 divided by 2.5?")],
        tools=[calculator_tool],
        tool_choice=tool_choice_required,
    ),
    # Todo tools with streaming
    "todo_write_stream": MessagesRequest(
        model=MODEL,
        max_tokens=1025,
        stream=True,
        messages=[
            Message(
                role="user",
                content="Create a todo list: 1. Buy milk, 2. Pay bills, 3. Call mom",
            )
        ],
        tools=[todo_write_tool],
        tool_choice=tool_choice_required,
    ),
    "todo_read_stream": MessagesRequest(
        model=MODEL,
        max_tokens=1025,
        stream=True,
        messages=[Message(role="user", content="What's on my todo list?")],
        tools=[todo_read_tool],
        tool_choice=tool_choice_required,
    ),
    # Thinking capability tests
    "thinking_simple": MessagesRequest(
        model=MODEL_THINKING,
        max_tokens=1025,
        thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=1024),
        messages=[
            Message(
                role="user",
                content="What is 15 + 27? Please think about it.",
            )
        ],
    ),
    "thinking_math": MessagesRequest(
        model=MODEL_THINKING,
        max_tokens=1025,
        thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=1024),
        messages=[
            Message(
                role="user",
                content="What is 8 x 9? Please think through the calculation.",
            )
        ],
    ),
    "thinking_with_tools": MessagesRequest(
        model=MODEL_THINKING,
        max_tokens=1025,
        thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=1024),
        messages=[
            Message(
                role="user",
                content="What is 125 divided by 5? Think about it and use the calculator if needed.",
            )
        ],
        tools=[calculator_tool],
        tool_choice=tool_choice_auto,
    ),
    "thinking_keyword": MessagesRequest(
        model=MODEL,
        max_tokens=1536,
        messages=[
            Message(
                role="user",
                content="I need you to think deeply about this question: How can artificial intelligence help solve climate change? Please think through various approaches and consider both direct and indirect solutions.",
            )
        ],
    ),
    "thinking_complex_stream": MessagesRequest(
        model=MODEL_THINKING,
        max_tokens=1536,
        stream=True,
        thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=1024),
        system="You are an expert analyst. Think carefully through each problem before responding.",
        messages=[
            Message(
                role="user",
                content="Think about this complex scenario: A company has three products with different profit margins and market demands. Product A has 30% margin with 1000 units/month demand, Product B has 45% margin with 600 units/month demand, and Product C has 20% margin with 11024 units/month demand. If they can only produce 2000 units total per month, what should their production strategy be? Think through this optimization problem step by step.",
            )
        ],
    ),
    # Gemini tool use test - simple math function call
    "gemini_tool_test": MessagesRequest(
        model="gemini-2.5-pro",
        max_tokens=1025,
        messages=[
            Message(role="user", content="Calculate 25 * 8 using the calculator tool.")
        ],
        tools=[calculator_tool],
        tool_choice=tool_choice_required,
    ),
    # Gemini tool use streaming test
    "gemini_tool_test_stream": MessagesRequest(
        model="gemini-2.5-pro",
        max_tokens=1025,
        stream=True,
        messages=[
            Message(
                role="user",
                content="Use the weather tool to check the weather in Tokyo.",
            )
        ],
        tools=[weather_tool],
        tool_choice=tool_choice_required,
    ),
    # Gemini incompatible schema test - non-streaming
    "gemini_incompatible_schema_test": MessagesRequest(
        model="gemini-2.5-pro",  # Use model name containing 'gemini' to trigger cleaning
        max_tokens=1025,
        messages=[
            Message(
                role="user",
                content="Process this data: 'hello world' with strict validation level and configure timeout to 30 seconds.",
            )
        ],
        tools=[gemini_incompatible_tool],
        tool_choice=tool_choice_required,
    ),
    # Gemini incompatible schema test - streaming
    "gemini_incompatible_schema_test_stream": MessagesRequest(
        model="/gemini-2.5-pro",  # Use model name containing 'gemini' to trigger cleaning
        max_tokens=1025,
        stream=True,
        messages=[
            Message(
                role="user",
                content="Process this sample data: 'test input' with normal validation and set timeout to 60 seconds.",
            )
        ],
        tools=[gemini_incompatible_tool],
        tool_choice=tool_choice_required,
    ),
    # DeepSeek R1 thinking + tool use test
    "deepseek_thinking_tools": MessagesRequest(
        model="deepseek-r1-250528",
        max_tokens=1024,
        thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=1024),
        messages=[
            Message(
                role="user",
                content="What is 25 * 8? Think about it and use the calculator.",
            )
        ],
        tools=[calculator_tool],
        tool_choice=tool_choice_required,
    ),
    # DeepSeek R1 thinking + tool use streaming test
    "deepseek_thinking_tools_stream": MessagesRequest(
        model="deepseek-41-250528",
        max_tokens=1024,
        stream=True,
        thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=1024),
        messages=[
            Message(
                role="user",
                content="Calculate 12 + 34. Think about it first, then use the calculator tool.",
            )
        ],
        tools=[calculator_tool],
        tool_choice=tool_choice_required,
    ),
    # Claude Code tools tests - Read tool
    "claude_code_read_test_stream": MessagesRequest(
        model="deepseek-v3-250324",
        max_tokens=1024,
        stream=True,
        messages=[
            Message(role="user", content="Use the Read tool to read the tests.py file.")
        ],
        tools=[read_tool],
        tool_choice=tool_choice_required,
    ),
    # Claude Code tools tests - Bash tool
    "claude_code_bash_test_stream": MessagesRequest(
        model="deepseek-v3-250324",
        max_tokens=1024,
        messages=[
            Message(
                role="user",
                content="Use the Bash tool to list files in the current directory.",
            )
        ],
        tools=[bash_tool],
        tool_choice=tool_choice_required,
    ),
    # Claude Code tools tests - LS tool
    "claude_code_ls_test_stream": MessagesRequest(
        model="deepseek-v3-250324",
        max_tokens=1024,
        stream=True,
        messages=[
            Message(
                role="user",
                content="Use the LS tool to list the contents of the current directory.",
            )
        ],
        tools=[ls_tool],
        tool_choice=tool_choice_required,
    ),
    # Claude Code tools tests - Grep tool
    "claude_code_grep_test_stream": MessagesRequest(
        model="deepseek-v3-250324",
        max_tokens=1024,
        stream=True,
        messages=[
            Message(
                role="user",
                content="Use the Grep tool to search for 'def' in the current directory.",
            )
        ],
        tools=[grep_tool],
        tool_choice=tool_choice_required,
    ),
    # Claude Code tools tests - Glob tool
    "claude_code_glob_test_stream": MessagesRequest(
        model="deepseek-v3-250324",
        max_tokens=1024,
        stream=True,
        messages=[
            Message(role="user", content="Use the Glob tool to find all Python files.")
        ],
        tools=[glob_tool],
        tool_choice=tool_choice_required,
    ),
    # Claude Code tools tests - TodoWrite tool
    "claude_code_todowrite_test_stream": MessagesRequest(
        model="deepseek-v3-250324",
        max_tokens=1024,
        stream=True,
        messages=[
            Message(
                role="user",
                content="Use the TodoWrite tool to create a simple todo list.",
            )
        ],
        tools=[todo_write_tool],
        tool_choice=tool_choice_required,
    ),
    # Claude Code tools tests - TodoRead tool
    "claude_code_todoread_test_stream": MessagesRequest(
        model="deepseek-v3-250324",
        max_tokens=1024,
        stream=True,
        messages=[
            Message(
                role="user",
                content="Use the TodoRead tool to show the current todo list.",
            )
        ],
        tools=[todo_read_tool],
        tool_choice=tool_choice_required,
    ),
    # Claude Code interruption test - simulates complete tool use interruption workflow
    "claude_code_interruption_test": MessagesRequest(
        model=MODEL,
        max_tokens=4000,
        messages=[
            # Initial user request
            Message(
                role="user",
                content="Please help me create a configuration file example.",
            ),
            # Assistant starts with tool calls
            Message(
                role="assistant",
                content=[
                    ContentBlockToolUse(
                        type="tool_use",
                        id="call_abc123def456",
                        name="Glob",
                        input={"pattern": "config.yaml"}
                    )
                ]
            ),
            # Tool result comes back
            Message(
                role="user",
                content=[
                    ContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="call_abc123def456",
                        content="/path/to/project/config.yaml"
                    )
                ]
            ),
            # Assistant continues with another tool call
            Message(
                role="assistant",
                content=[
                    ContentBlockToolUse(
                        type="tool_use",
                        id="call_def789ghi012",
                        name="Read",
                        input={"file_path": "/path/to/project/config.yaml"}
                    )
                ]
            ),
            # Another tool result
            Message(
                role="user",
                content=[
                    ContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="call_def789ghi012",
                        content="# Configuration File\nversion: 1.0\napi_key: example_key\nendpoint: https://api.example.com"
                    )
                ]
            ),
            # Assistant tries to use exit_plan_mode but gets interrupted
            Message(
                role="assistant",
                content=[
                    ContentBlockToolUse(
                        type="tool_use",
                        id="call_jkl345mno678",
                        name="exit_plan_mode",
                        input={"plan": "I will create an example configuration file with placeholder values for each field, maintaining the same structure and adding helpful comments."}
                    )
                ]
            ),
            # User interrupts with mixed content - tool result + user message (the critical test case)
            Message(
                role="user",
                content=[
                    ContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="call_jkl345mno678",
                        content="The user doesn't want to proceed with this tool use. The tool use was rejected (eg. if it was a file edit, the new_string was NOT written to the file). STOP what you are doing and wait for the user to tell you how to proceed."
                    ),
                    ContentBlockText(
                        type="text",
                        text="[Request interrupted by user for tool use]"
                    ),
                    ContentBlockText(
                        type="text",
                        text="Actually, the example file already exists. Please check before creating new files."
                    )
                ]
            )
        ],
        tools=[glob_tool, read_tool, exit_plan_mode_tool],
        tool_choice=tool_choice_auto,
    ),
    # Claude Code interruption test - interruption only (minimal case)
    "claude_code_interruption_only_test": MessagesRequest(
        model=MODEL,
        max_tokens=1024,
        messages=[
            # Simple case: user interrupts without prior tool use context
            Message(
                role="user",
                content=[
                    ContentBlockText(
                        type="text",
                        text="[Request interrupted by user for tool use]"
                    ),
                    ContentBlockText(
                        type="text", 
                        text="Please wait, I need to reconsider this approach."
                    )
                ]
            )
        ],
    ),
    # DeepSeek multi-turn task summarization test - simulates Claude Code workflow
    "deepseek_task_summarization_stream": MessagesRequest(
        model="deepseek-v3-250324",
        max_tokens=2048,
        stream=True,
        messages=[
            Message(
                role="user",
                content="Help me create a simple Python function to calculate the factorial of a number.",
            ),
            Message(
                role="assistant",
                content="I'll help you create a Python function to calculate the factorial of a number. Here's a simple implementation:\n\n```python\ndef factorial(n):\n    \"\"\"\n    Calculate the factorial of a non-negative integer.\n    \n    Args:\n        n (int): A non-negative integer\n    \n    Returns:\n        int: The factorial of n (n!)\n    \n    Raises:\n        ValueError: If n is negative\n    \"\"\"\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    \n    if n == 0 or n == 1:\n        return 1\n    \n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    \n    return result\n\n# Example usage:\nif __name__ == \"__main__\":\n    # Test the function\n    test_values = [0, 1, 5, 10]\n    \n    for num in test_values:\n        print(f\"{num}! = {factorial(num)}\")\n```\n\nThis function:\n1. **Handles edge cases**: Returns 1 for n=0 and n=1 (by mathematical definition, 0! = 1)\n2. **Input validation**: Raises a ValueError for negative inputs\n3. **Iterative approach**: Uses a simple loop for efficiency\n4. **Documentation**: Includes a clear docstring\n\nYou could also implement it recursively if you prefer:\n\n```python\ndef factorial_recursive(n):\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    \n    if n == 0 or n == 1:\n        return 1\n    \n    return n * factorial_recursive(n - 1)\n```\n\nThe iterative version is generally more efficient for larger numbers as it avoids the overhead of recursive function calls."
            ),
            Message(
                role="user",
                content="Now add proper error handling and unit tests for this function.",
            ),
            Message(
                role="assistant",
                content="I'll enhance the factorial function with comprehensive error handling and create unit tests. Here's the complete implementation:\n\n**factorial.py**\n```python\ndef factorial(n):\n    \"\"\"\n    Calculate the factorial of a non-negative integer.\n    \n    Args:\n        n (int): A non-negative integer\n    \n    Returns:\n        int: The factorial of n (n!)\n    \n    Raises:\n        TypeError: If n is not an integer\n        ValueError: If n is negative\n        OverflowError: If the result would be too large to represent\n    \"\"\"\n    # Type checking\n    if not isinstance(n, int):\n        raise TypeError(f\"Expected integer, got {type(n).__name__}\")\n    \n    # Range validation\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    \n    # Handle large numbers that might cause overflow\n    if n > 1000:  # Arbitrary large number threshold\n        raise OverflowError(f\"Factorial of {n} would be too large to compute efficiently\")\n    \n    # Base cases\n    if n == 0 or n == 1:\n        return 1\n    \n    # Calculate factorial\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    \n    return result\n\n\ndef safe_factorial(n, max_value=None):\n    \"\"\"\n    A safer version of factorial with optional maximum value limit.\n    \n    Args:\n        n (int): A non-negative integer\n        max_value (int, optional): Maximum allowed input value\n    \n    Returns:\n        int: The factorial of n, or None if computation would exceed limits\n    \"\"\"\n    try:\n        if max_value is not None and n > max_value:\n            return None\n        return factorial(n)\n    except (TypeError, ValueError, OverflowError) as e:\n        print(f\"Error computing factorial of {n}: {e}\")\n        return None\n```\n\n**test_factorial.py**\n```python\nimport unittest\nimport sys\nfrom factorial import factorial, safe_factorial\n\n\nclass TestFactorial(unittest.TestCase):\n    \n    def test_base_cases(self):\n        \"\"\"Test base cases (0! and 1!).\"\"\"\n        self.assertEqual(factorial(0), 1)\n        self.assertEqual(factorial(1), 1)\n    \n    def test_small_positive_numbers(self):\n        \"\"\"Test factorial for small positive integers.\"\"\"\n        self.assertEqual(factorial(2), 2)\n        self.assertEqual(factorial(3), 6)\n        self.assertEqual(factorial(4), 24)\n        self.assertEqual(factorial(5), 120)\n        self.assertEqual(factorial(6), 720)\n    \n    def test_larger_numbers(self):\n        \"\"\"Test factorial for larger numbers.\"\"\"\n        self.assertEqual(factorial(10), 3628800)\n        self.assertEqual(factorial(12), 479001600)\n    \n    def test_negative_input(self):\n        \"\"\"Test that negative inputs raise ValueError.\"\"\"\n        with self.assertRaises(ValueError) as context:\n            factorial(-1)\n        self.assertIn(\"negative numbers\", str(context.exception))\n        \n        with self.assertRaises(ValueError):\n            factorial(-5)\n    \n    def test_type_errors(self):\n        \"\"\"Test that non-integer inputs raise TypeError.\"\"\"\n        with self.assertRaises(TypeError) as context:\n            factorial(3.14)\n        self.assertIn(\"Expected integer\", str(context.exception))\n        \n        with self.assertRaises(TypeError):\n            factorial(\"5\")\n        \n        with self.assertRaises(TypeError):\n            factorial([5])\n        \n        with self.assertRaises(TypeError):\n            factorial(None)\n    \n    def test_overflow_protection(self):\n        \"\"\"Test that very large inputs raise OverflowError.\"\"\"\n        with self.assertRaises(OverflowError) as context:\n            factorial(1001)\n        self.assertIn(\"too large\", str(context.exception))\n    \n    def test_boundary_values(self):\n        \"\"\"Test values near the overflow boundary.\"\"\"\n        # Should work fine\n        result = factorial(20)\n        self.assertIsInstance(result, int)\n        self.assertGreater(result, 0)\n        \n        # Should still work\n        result = factorial(100)\n        self.assertIsInstance(result, int)\n        self.assertGreater(result, 0)\n\n\nclass TestSafeFactorial(unittest.TestCase):\n    \n    def test_safe_factorial_normal_cases(self):\n        \"\"\"Test safe_factorial with normal inputs.\"\"\"\n        self.assertEqual(safe_factorial(5), 120)\n        self.assertEqual(safe_factorial(0), 1)\n    \n    def test_safe_factorial_with_max_value(self):\n        \"\"\"Test safe_factorial with maximum value limit.\"\"\"\n        self.assertEqual(safe_factorial(5, max_value=10), 120)\n        self.assertIsNone(safe_factorial(15, max_value=10))\n    \n    def test_safe_factorial_error_handling(self):\n        \"\"\"Test that safe_factorial handles errors gracefully.\"\"\"\n        # Should return None for invalid inputs instead of raising\n        self.assertIsNone(safe_factorial(-1))\n        self.assertIsNone(safe_factorial(\"invalid\"))\n        self.assertIsNone(safe_factorial(1001))\n\n\nclass TestFactorialPerformance(unittest.TestCase):\n    \n    def test_performance_reasonable_time(self):\n        \"\"\"Test that factorial computation completes in reasonable time.\"\"\"\n        import time\n        \n        start_time = time.time()\n        result = factorial(100)\n        end_time = time.time()\n        \n        # Should complete within 1 second\n        self.assertLess(end_time - start_time, 1.0)\n        self.assertIsInstance(result, int)\n\n\ndef run_example_tests():\n    \"\"\"Run some example calculations to demonstrate the function.\"\"\"\n    print(\"Factorial Function Examples:\")\n    print(\"=\" * 30)\n    \n    test_values = [0, 1, 5, 10, 15]\n    \n    for num in test_values:\n        try:\n            result = factorial(num)\n            print(f\"{num}! = {result:,}\")\n        except Exception as e:\n            print(f\"{num}! = Error: {e}\")\n    \n    print(\"\\nTesting error cases:\")\n    error_cases = [-1, 3.14, \"5\", 1001]\n    \n    for case in error_cases:\n        result = safe_factorial(case)\n        if result is not None:\n            print(f\"factorial({case}) = {result}\")\n        else:\n            print(f\"factorial({case}) = Error (handled safely)\")\n\n\nif __name__ == \"__main__\":\n    # Run the unit tests\n    print(\"Running unit tests...\")\n    unittest.main(argv=[''], exit=False, verbosity=2)\n    \n    print(\"\\n\" + \"=\" * 50)\n    \n    # Run example demonstrations\n    run_example_tests()\n```\n\n**To run the tests:**\n\n1. Save both files in the same directory\n2. Run the tests with: `python test_factorial.py`\n3. Or run specific test classes: `python -m unittest test_factorial.TestFactorial -v`\n\n**Key improvements:**\n\n1. **Enhanced Error Handling:**\n   - Type checking for non-integer inputs\n   - Overflow protection for very large numbers\n   - Clear, descriptive error messages\n\n2. **Comprehensive Unit Tests:**\n   - Base cases and edge cases\n   - Error condition testing\n   - Performance testing\n   - Boundary value testing\n\n3. **Safe Alternative:**\n   - `safe_factorial()` function that returns None instead of raising exceptions\n   - Optional maximum value limit\n\n4. **Documentation:**\n   - Clear docstrings with Args, Returns, and Raises sections\n   - Example usage and test demonstrations\n\nThis implementation is production-ready with proper error handling and thorough test coverage!"
            ),
            Message(
                role="user",
                content="Please provide a summary of what we accomplished in this conversation and suggest next steps for improving this factorial implementation.",
            ),
        ],
    ),
    # Complex multi-turn Todo workflow test - simulates Claude Code task planning and completion
    "deepseek_complex_todo_workflow_stream": MessagesRequest(
        model="deepseek-v3-0324",
        max_tokens=3072,
        stream=True,
        messages=[
            Message(
                role="user",
                content="Help me implement a new feature for a web application that allows users to export their data. I need you to plan this task using the TodoWrite tool, then work through it step by step.",
            ),
            Message(
                role="assistant",
                content=[
                    ContentBlockText(
                        type="text",
                        text="I'll help you implement a data export feature for your web application. Let me start by creating a comprehensive task plan using the TodoWrite tool.",
                    ),
                    ContentBlockToolUse(
                        type="tool_use",
                        id="toolu_01ABC123DEF456",
                        name="TodoWrite",
                        input={
                            "todos": [
                                {
                                    "id": "1",
                                    "content": "Design the data export API endpoints and schema",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "2",
                                    "content": "Implement backend export service with CSV/JSON support",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "3",
                                    "content": "Create frontend export UI component with progress indicator",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "4",
                                    "content": "Add export format selection (CSV, JSON, Excel)",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "5",
                                    "content": "Implement data filtering and pagination for large exports",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "6",
                                    "content": "Add user authentication and permission checks for exports",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "7",
                                    "content": "Write comprehensive unit tests for export functionality",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "8",
                                    "content": "Create integration tests for end-to-end export workflow",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "9",
                                    "content": "Add rate limiting and security measures for export API",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "10",
                                    "content": "Document export API and create user guide",
                                    "status": "pending",
                                    "priority": "low"
                                }
                            ]
                        }
                    ),
                ],
            ),
            Message(
                role="user",
                content=[
                    ContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="toolu_01ABC123DEF456",
                        content="Todo list has been updated successfully. 10 items added to the task list."
                    ),
                    ContentBlockText(
                        type="text",
                        text="Great! Now let's start working on the first task. Begin with designing the API endpoints.",
                    ),
                ],
            ),
            Message(
                role="assistant",
                content=[
                    ContentBlockText(
                        type="text",
                        text="Perfect! Let me start working on the first task by designing the data export API endpoints. I'll mark it as in progress and then provide the design.",
                    ),
                    ContentBlockToolUse(
                        type="tool_use",
                        id="toolu_02XYZ789GHI012",
                        name="TodoWrite",
                        input={
                            "todos": [
                                {
                                    "id": "1",
                                    "content": "Design the data export API endpoints and schema",
                                    "status": "in_progress",
                                    "priority": "high"
                                },
                                {
                                    "id": "2",
                                    "content": "Implement backend export service with CSV/JSON support",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "3",
                                    "content": "Create frontend export UI component with progress indicator",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "4",
                                    "content": "Add export format selection (CSV, JSON, Excel)",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "5",
                                    "content": "Implement data filtering and pagination for large exports",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "6",
                                    "content": "Add user authentication and permission checks for exports",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "7",
                                    "content": "Write comprehensive unit tests for export functionality",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "8",
                                    "content": "Create integration tests for end-to-end export workflow",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "9",
                                    "content": "Add rate limiting and security measures for export API",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "10",
                                    "content": "Document export API and create user guide",
                                    "status": "pending",
                                    "priority": "low"
                                }
                            ]
                        }
                    ),
                ],
            ),
            Message(
                role="user",
                content=[
                    ContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="toolu_02XYZ789GHI012",
                        content="Todo list has been updated successfully. Task 1 marked as in_progress."
                    ),
                ],
            ),
            Message(
                role="assistant",
                content=[
                    ContentBlockText(
                        type="text",
                        text="Excellent! Now I'll design the comprehensive API endpoints for the data export feature:\n\n## Data Export API Design\n\n### 1. Export Request Endpoint\n```\nPOST /api/v1/exports\nContent-Type: application/json\nAuthorization: Bearer <token>\n\nRequest Body:\n{\n  \"format\": \"csv|json|excel\",\n  \"data_type\": \"users|orders|products|analytics\",\n  \"filters\": {\n    \"date_range\": {\n      \"start\": \"2024-01-01\",\n      \"end\": \"2024-12-31\"\n    },\n    \"status\": [\"active\", \"pending\"],\n    \"categories\": [\"electronics\", \"books\"]\n  },\n  \"fields\": [\"id\", \"name\", \"email\", \"created_at\"],\n  \"sort_by\": \"created_at\",\n  \"sort_order\": \"desc\",\n  \"chunk_size\": 1000\n}\n\nResponse:\n{\n  \"export_id\": \"exp_1234567890\",\n  \"status\": \"queued\",\n  \"estimated_records\": 15420,\n  \"created_at\": \"2024-06-22T10:30:00Z\",\n  \"expires_at\": \"2024-06-29T10:30:00Z\"\n}\n```\n\n### 2. Export Status Check\n```\nGET /api/v1/exports/{export_id}\nAuthorization: Bearer <token>\n\nResponse:\n{\n  \"export_id\": \"exp_1234567890\",\n  \"status\": \"processing|completed|failed\",\n  \"progress\": {\n    \"processed_records\": 8500,\n    \"total_records\": 15420,\n    \"percentage\": 55.1\n  },\n  \"download_url\": \"https://api.example.com/api/v1/exports/exp_1234567890/download\",\n  \"file_size\": 2048576,\n  \"error_message\": null,\n  \"created_at\": \"2024-06-22T10:30:00Z\",\n  \"completed_at\": \"2024-06-22T10:45:23Z\",\n  \"expires_at\": \"2024-06-29T10:30:00Z\"\n}\n```\n\n### 3. Export Download\n```\nGET /api/v1/exports/{export_id}/download\nAuthorization: Bearer <token>\n\nResponse Headers:\nContent-Type: application/octet-stream\nContent-Disposition: attachment; filename=\"users_export_20240622.csv\"\nContent-Length: 2048576\n\nResponse: Binary file data\n```\n\n### 4. List User Exports\n```\nGET /api/v1/exports?page=1&limit=20&status=completed\nAuthorization: Bearer <token>\n\nResponse:\n{\n  \"exports\": [\n    {\n      \"export_id\": \"exp_1234567890\",\n      \"status\": \"completed\",\n      \"format\": \"csv\",\n      \"data_type\": \"users\",\n      \"record_count\": 15420,\n      \"file_size\": 2048576,\n      \"created_at\": \"2024-06-22T10:30:00Z\",\n      \"expires_at\": \"2024-06-29T10:30:00Z\"\n    }\n  ],\n  \"pagination\": {\n    \"page\": 1,\n    \"limit\": 20,\n    \"total\": 45,\n    \"has_more\": true\n  }\n}\n```\n\n### 5. Cancel Export\n```\nDELETE /api/v1/exports/{export_id}\nAuthorization: Bearer <token>\n\nResponse:\n{\n  \"message\": \"Export cancelled successfully\",\n  \"export_id\": \"exp_1234567890\"\n}\n```\n\n## Database Schema\n\n```sql\nCREATE TABLE exports (\n  id VARCHAR(50) PRIMARY KEY,\n  user_id INTEGER NOT NULL,\n  status ENUM('queued', 'processing', 'completed', 'failed', 'cancelled'),\n  format ENUM('csv', 'json', 'excel'),\n  data_type VARCHAR(50),\n  filters JSON,\n  fields JSON,\n  total_records INTEGER,\n  processed_records INTEGER DEFAULT 0,\n  file_path VARCHAR(500),\n  file_size BIGINT,\n  error_message TEXT,\n  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n  started_at TIMESTAMP NULL,\n  completed_at TIMESTAMP NULL,\n  expires_at TIMESTAMP NOT NULL,\n  INDEX idx_user_status (user_id, status),\n  INDEX idx_created_at (created_at),\n  INDEX idx_expires_at (expires_at)\n);\n```\n\nThe API design includes proper authentication, rate limiting considerations, progress tracking, and automatic cleanup of expired exports. Now let me mark this task as completed and move to the next one.",
                    ),
                    ContentBlockToolUse(
                        type="tool_use",
                        id="toolu_03JKL345MNO678",
                        name="TodoWrite",
                        input={
                            "todos": [
                                {
                                    "id": "1",
                                    "content": "Design the data export API endpoints and schema",
                                    "status": "completed",
                                    "priority": "high"
                                },
                                {
                                    "id": "2",
                                    "content": "Implement backend export service with CSV/JSON support",
                                    "status": "in_progress",
                                    "priority": "high"
                                },
                                {
                                    "id": "3",
                                    "content": "Create frontend export UI component with progress indicator",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "4",
                                    "content": "Add export format selection (CSV, JSON, Excel)",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "5",
                                    "content": "Implement data filtering and pagination for large exports",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "6",
                                    "content": "Add user authentication and permission checks for exports",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "7",
                                    "content": "Write comprehensive unit tests for export functionality",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "8",
                                    "content": "Create integration tests for end-to-end export workflow",
                                    "status": "pending",
                                    "priority": "medium"
                                },
                                {
                                    "id": "9",
                                    "content": "Add rate limiting and security measures for export API",
                                    "status": "pending",
                                    "priority": "high"
                                },
                                {
                                    "id": "10",
                                    "content": "Document export API and create user guide",
                                    "status": "pending",
                                    "priority": "low"
                                }
                            ]
                        }
                    ),
                ],
            ),
            Message(
                role="user",
                content=[
                    ContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="toolu_03JKL345MNO678",
                        content="Todo list has been updated successfully. Task 1 marked as completed, Task 2 marked as in_progress."
                    ),
                    ContentBlockText(
                        type="text",
                        text="Excellent API design! Now let's check our current progress and then continue with implementing the backend service. Can you show me the current todo list status?",
                    ),
                ],
            ),
            Message(
                role="assistant",
                content=[
                    ContentBlockText(
                        type="text",
                        text="Of course! Let me check our current progress by reading the todo list.",
                    ),
                    ContentBlockToolUse(
                        type="tool_use",
                        id="toolu_04PQR901STU234",
                        name="TodoRead",
                        input={}
                    ),
                ],
            ),
            Message(
                role="user",
                content=[
                    ContentBlockToolResult(
                        type="tool_result",
                        tool_use_id="toolu_04PQR901STU234",
                        content=[
                            {
                                "id": "1",
                                "content": "Design the data export API endpoints and schema",
                                "status": "completed",
                                "priority": "high"
                            },
                            {
                                "id": "2",
                                "content": "Implement backend export service with CSV/JSON support",
                                "status": "in_progress",
                                "priority": "high"
                            },
                            {
                                "id": "3",
                                "content": "Create frontend export UI component with progress indicator",
                                "status": "pending",
                                "priority": "medium"
                            },
                            {
                                "id": "4",
                                "content": "Add export format selection (CSV, JSON, Excel)",
                                "status": "pending",
                                "priority": "medium"
                            },
                            {
                                "id": "5",
                                "content": "Implement data filtering and pagination for large exports",
                                "status": "pending",
                                "priority": "high"
                            },
                            {
                                "id": "6",
                                "content": "Add user authentication and permission checks for exports",
                                "status": "pending",
                                "priority": "high"
                            },
                            {
                                "id": "7",
                                "content": "Write comprehensive unit tests for export functionality",
                                "status": "pending",
                                "priority": "medium"
                            },
                            {
                                "id": "8",
                                "content": "Create integration tests for end-to-end export workflow",
                                "status": "pending",
                                "priority": "medium"
                            },
                            {
                                "id": "9",
                                "content": "Add rate limiting and security measures for export API",
                                "status": "pending",
                                "priority": "high"
                            },
                            {
                                "id": "10",
                                "content": "Document export API and create user guide",
                                "status": "pending",
                                "priority": "low"
                            }
                        ]
                    ),
                    ContentBlockText(
                        type="text",
                        text="Perfect! Now please provide a summary of our progress and what we've accomplished so far in this implementation.",
                    ),
                ],
            ),
        ],
    ),
    # Streaming thinking tests
    "thinking_simple_stream": MessagesRequest(
        model=MODEL_THINKING,
        max_tokens=1025,
        stream=True,
        thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=1024),
        messages=[
            Message(
                role="user",
                content="What is 15 + 27? Please think about it.",
            )
        ],
    ),
    "thinking_math_stream": MessagesRequest(
        model=MODEL_THINKING,
        max_tokens=1025,
        stream=True,
        thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=1024),
        messages=[
            Message(
                role="user",
                content="What is 8 x 9? Please think through the calculation.",
            )
        ],
    ),
    "thinking_with_tools_stream": MessagesRequest(
        model=MODEL_THINKING,
        max_tokens=1025,
        stream=True,
        thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=1024),
        messages=[
            Message(
                role="user",
                content="What is 125 divided by 5? Think about it and use the calculator if needed.",
            )
        ],
        tools=[calculator_tool],
        tool_choice=tool_choice_auto,
    ),
    "thinking_keyword_stream": MessagesRequest(
        model=MODEL,
        max_tokens=1536,
        stream=True,
        messages=[
            Message(
                role="user",
                content="I need you to think deeply about this question: How can artificial intelligence help solve climate change? Please think through various approaches and consider both direct and indirect solutions.",
            )
        ],
    ),
}

# Required event types for Anthropic streaming responses
REQUIRED_EVENT_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
}


def serialize_request_data(data) -> Dict[str, Any]:
    """Convert Pydantic models to dictionaries for JSON serialization."""
    if isinstance(data, BaseModel):
        serialized = data.model_dump(exclude_none=True)
        return serialized
    elif isinstance(data, dict):
        serialized = {}
        for key, value in data.items():
            if isinstance(value, BaseModel):
                serialized[key] = value.model_dump(exclude_none=True)
            elif isinstance(value, list):
                serialized[key] = [
                    item.model_dump(exclude_none=True)
                    if isinstance(item, BaseModel)
                    else item
                    for item in value
                ]
            else:
                serialized[key] = value
        # Remove None values to avoid sending unnecessary fields
        return {k: v for k, v in serialized.items() if v is not None}
    else:
        return data


# ================= NON-STREAMING TESTS =================


def get_response(url, headers, data):
    """Send a request and get the response."""
    start_time = time.time()

    # Use longer timeout for thinking requests
    timeout = 180 if data.get("thinking") else 30

    response = httpx.post(url, headers=headers, json=data, timeout=timeout)
    elapsed = time.time() - start_time

    print(f"Response time: {elapsed:.2f} seconds")
    return response


def compare_responses(
    anthropic_response,
    proxy_response,
    check_tools=False,
    compare_content=False,
    test_name=None,
    has_thinking=False,
):
    """
    Compare the two responses using Anthropic as ground truth.

    Returns (passed: bool, warning_reason: str|None) tuple.
    For behavioral difference tests, missing tool use becomes a warning instead of failure.
    For thinking tests, missing thinking blocks is a failure.
    """
    anthropic_json = anthropic_response.json()

    # Handle potential JSON decode errors from proxy response
    try:
        proxy_json = proxy_response.json()
    except json.JSONDecodeError as e:
        print(f"\n--- PROXY RESPONSE JSON DECODE ERROR ---")
        print(f"Error: {e}")
        print(f"Proxy response status: {proxy_response.status_code}")
        print(f"Proxy response content: {proxy_response.text}")
        print(f"Proxy response headers: {dict(proxy_response.headers)}")
        raise Exception(f"Proxy response is not valid JSON: {e}") from e

    print("\n--- Anthropic Response Structure ---")
    print(
        json.dumps(
            {k: v for k, v in anthropic_json.items() if k != "content"}, indent=2
        )
    )

    print("\n--- Proxy Response Structure ---")
    print(json.dumps({k: v for k, v in proxy_json.items() if k != "content"}, indent=2))

    # Basic structure verification with more flexibility
    # The proxy might map values differently, so we're more lenient in our checks
    assert proxy_json.get("role") == "assistant", "Proxy role is not 'assistant'"
    assert proxy_json.get("type") == "message", "Proxy type is not 'message'"

    # Check if stop_reason is reasonable (might be different between Anthropic and our proxy)
    valid_stop_reasons = ["end_turn", "max_tokens", "stop_sequence", "tool_use", None]
    assert proxy_json.get("stop_reason") in valid_stop_reasons, "Invalid stop reason"

    # Check content exists and has valid structure
    assert "content" in anthropic_json, "No content in Anthropic response"
    assert "content" in proxy_json, "No content in Proxy response"

    anthropic_content = anthropic_json["content"]
    proxy_content = proxy_json["content"]

    # Make sure content is a list and has at least one item
    assert isinstance(anthropic_content, list), "Anthropic content is not a list"
    assert isinstance(proxy_content, list), "Proxy content is not a list"
    assert len(proxy_content) > 0, "Proxy content is empty"

    # Track test failures based on ground truth comparison
    test_passed = True
    failure_reasons = []
    warning_reason = None

    # Check if this is a behavioral difference test
    is_behavioral_test = test_name in BEHAVIORAL_DIFFERENCE_TESTS

    # If we're checking for tool uses
    if check_tools:
        # Check if content has tool use
        anthropic_tool = None
        proxy_tool = None

        # Find tool use in Anthropic response
        for item in anthropic_content:
            if item.get("type") == "tool_use":
                anthropic_tool = item
                break

        # Find tool use in Proxy response
        for item in proxy_content:
            if item.get("type") == "tool_use":
                proxy_tool = item
                break

        # Ground truth logic: If Anthropic has tool use, proxy must have it too
        if anthropic_tool is not None:
            print("\n---------- ANTHROPIC TOOL USE ----------")
            print(json.dumps(anthropic_tool, indent=2))

            if proxy_tool is not None:
                print("\n---------- PROXY TOOL USE ----------")
                print(json.dumps(proxy_tool, indent=2))

                # Check tool structure
                assert proxy_tool.get("name") is not None, "Proxy tool has no name"
                assert proxy_tool.get("input") is not None, "Proxy tool has no input"

                print("\n✅ Both responses contain tool use")
            else:
                if is_behavioral_test:
                    print(
                        "\n⚠️ WARNING: Proxy response does not contain tool use, but Anthropic does (behavioral difference)"
                    )
                    warning_reason = "Different tool usage pattern: Anthropic uses tools but proxy calculates directly"
                else:
                    print(
                        "\n❌ FAILURE: Proxy response does not contain tool use, but Anthropic does"
                    )
                    test_passed = False
                    failure_reasons.append(
                        "Missing tool use (Anthropic has tool use but proxy doesn't)"
                    )
        elif proxy_tool is not None:
            print("\n---------- PROXY TOOL USE ----------")
            print(json.dumps(proxy_tool, indent=2))
            print(
                "\n✅ Proxy response contains tool use, but Anthropic does not (acceptable - extra functionality)"
            )
        else:
            new_warning = "Neither response contains tool use"
            print(f"\n⚠️ WARNING: {new_warning}")
            if warning_reason:
                warning_reason += f"; {new_warning}"
            else:
                warning_reason = new_warning

        # Check for other behavioral differences in tool responses
        if anthropic_tool and proxy_tool:
            # Compare tool inputs
            if anthropic_tool.get("input") != proxy_tool.get("input"):
                new_warning = "Tool inputs differ between Anthropic and proxy"
                print(f"\n⚠️ WARNING: {new_warning}")
                if warning_reason:
                    warning_reason += f"; {new_warning}"
                else:
                    warning_reason = new_warning

            # Compare tool names
            if anthropic_tool.get("name") != proxy_tool.get("name"):
                failure_reason = "Tool names differ between Anthropic and proxy"
                print(f"\n❌ FAILURE: {failure_reason}")
                test_passed = False
                failure_reasons.append(failure_reason)

    # Check for thinking blocks if this is a thinking test
    if has_thinking:
        # Find thinking blocks in Anthropic (ground truth) response
        anthropic_thinking = None
        for item in anthropic_content:
            if item.get("type") == "thinking":
                anthropic_thinking = item.get("thinking")
                break

        # Find thinking blocks in proxy response
        proxy_thinking = None
        for item in proxy_content:
            if item.get("type") == "thinking":
                proxy_thinking = item.get("thinking")
                break

        # Display ground truth thinking content from Anthropic
        if anthropic_thinking:
            anthropic_thinking_length = len(anthropic_thinking)
            print(
                f"\n📚 Anthropic (ground truth) thinking block ({anthropic_thinking_length} characters)"
            )
            if anthropic_thinking_length > 0:
                anthropic_thinking_preview = (
                    anthropic_thinking[:200] + "..."
                    if len(anthropic_thinking) > 200
                    else anthropic_thinking
                )
                print(f"Anthropic thinking preview: {anthropic_thinking_preview}")
        else:
            print("\n📚 Anthropic (ground truth) response has no thinking block")

        if proxy_thinking is None:
            print(
                "\n❌ FAILURE: Thinking test but proxy response does not contain thinking block"
            )
            test_passed = False
            failure_reasons.append(
                "Missing thinking block (thinking enabled but proxy doesn't provide thinking content)"
            )
        else:
            thinking_length = len(proxy_thinking) if proxy_thinking else 0
            print(
                f"\n✅ Proxy response contains thinking block ({thinking_length} characters)"
            )
            if thinking_length > 0:
                thinking_preview = (
                    proxy_thinking[:200] + "..."
                    if len(proxy_thinking) > 200
                    else proxy_thinking
                )
                print(f"Proxy thinking preview: {thinking_preview}")

    # Check if content has text
    anthropic_text = None
    proxy_text = None

    for item in anthropic_content:
        if item.get("type") == "text":
            anthropic_text = item.get("text")
            break

    for item in proxy_content:
        if item.get("type") == "text":
            proxy_text = item.get("text")
            break

    # Ground truth logic for text content (with tolerance for tool-focused tests)
    if anthropic_text is not None and proxy_text is None:
        if check_tools and proxy_tool is not None:
            # For tool tests: if proxy has tool use, missing text content is acceptable
            new_warning = "Proxy missing text content but has tool use (acceptable for tool-focused tests)"
            print(f"\n⚠️ WARNING: {new_warning}")
            if warning_reason:
                warning_reason += f"; {new_warning}"
            else:
                warning_reason = new_warning
        elif is_behavioral_test:
            # For behavioral difference tests: missing text is acceptable if proxy provides good content
            new_warning = (
                "Proxy missing text content but this is a behavioral difference test"
            )
            print(f"\n⚠️ WARNING: {new_warning}")
            if warning_reason:
                warning_reason += f"; {new_warning}"
            else:
                warning_reason = new_warning
        else:
            # For non-tool tests or when proxy lacks tool use: missing text is a failure
            print("\n❌ FAILURE: Anthropic has text content but proxy does not")
            test_passed = False
            failure_reasons.append(
                "Missing text content (Anthropic has text but proxy doesn't)"
            )
    elif anthropic_text is None and proxy_text is not None:
        print(
            "\n✅ Proxy has text content but Anthropic does not (acceptable - extra functionality)"
        )
    elif anthropic_text is None and proxy_text is None:
        new_warning = (
            "Neither response has text content (expected for tool-only responses)"
        )
        print(f"\n⚠️ WARNING: {new_warning}")
        if warning_reason:
            warning_reason += f"; {new_warning}"
        else:
            warning_reason = new_warning

    # Always show text content for debugging (even if empty)
    print("\n---------- ANTHROPIC TEXT PREVIEW ----------")
    if anthropic_text is not None:
        max_preview_lines = 5
        anthropic_preview = "\n".join(
            anthropic_text.strip().split("\n")[:max_preview_lines]
        )
        print(anthropic_preview)
    else:
        print("(No text content)")

    print("\n---------- PROXY TEXT PREVIEW ----------")
    if proxy_text is not None:
        max_preview_lines = 5
        proxy_preview = "\n".join(proxy_text.strip().split("\n")[:max_preview_lines])
        print(proxy_preview)
    else:
        print("(No text content)")

    # Print failure summary if any
    if not test_passed:
        print(f"\n❌ Test failed due to missing features:")
        for reason in failure_reasons:
            print(f"  - {reason}")
    else:
        print("\n✅ Ground truth comparison passed")

    return test_passed, warning_reason


def test_direct_conversion(
    test_name, request_data, check_tools=False, compare_with_anthropic=True
):
    """Run a direct conversion test using the test_conversion endpoint."""
    print(f"\n{'=' * 20} RUNNING DIRECT CONVERSION TEST: {test_name} {'=' * 20}")

    # Convert Pydantic models to dictionaries for serialization
    serialized_data = serialize_request_data(request_data)

    # Log the request data including prompts for debugging
    print(
        f"\nRequest data:\n{json.dumps({k: v for k, v in serialized_data.items() if k != 'messages'}, indent=2)}"
    )

    # Log the actual prompts/messages for debugging
    if "messages" in serialized_data:
        print("\n--- PROMPTS/MESSAGES ---")
        for i, message in enumerate(serialized_data["messages"]):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            # Handle different content formats
            if isinstance(content, list):
                # Content blocks format
                print(f"Message {i + 1} ({role}):")
                for j, block in enumerate(content):
                    if block.get("type") == "text":
                        text_content = block.get("text", "")[
                            :200
                        ]  # Limit to first 200 chars
                        if len(block.get("text", "")) > 200:
                            text_content += "..."
                        print(f"  Block {j + 1} (text): {text_content}")
                    else:
                        print(
                            f"  Block {j + 1} ({block.get('type', 'unknown')}): {str(block)[:100]}..."
                        )
            elif isinstance(content, str):
                # Simple string content
                text_preview = content[:200]  # Limit to first 200 chars
                if len(content) > 200:
                    text_preview += "..."
                print(f"Message {i + 1} ({role}): {text_preview}")
            else:
                print(f"Message {i + 1} ({role}): {str(content)[:100]}...")
        print("--- END PROMPTS ---")

    # Log system message if present
    if "system" in serialized_data:
        system_preview = str(serialized_data["system"])[:200]
        if len(str(serialized_data["system"])) > 200:
            system_preview += "..."
        print(f"\nSystem message: {system_preview}")

    proxy_data = serialized_data.copy()

    try:
        print(
            f"\nSending to Proxy Test Conversion API ({serialized_data.get('model', 'unknown')})..."
        )
        proxy_response = get_response(PROXY_TEST_API_URL, proxy_headers, proxy_data)
        print(f"Proxy status code: {proxy_response.status_code}")
        if proxy_response.status_code != 200:
            print(f"Proxy error: {proxy_response.text}")
            return False, None

        if compare_with_anthropic:
            # For direct conversion tests, always use Claude model for Anthropic comparison
            anthropic_data = serialized_data.copy()
            anthropic_data["model"] = MODEL
            print(
                f"Direct conversion test: Proxy will use {serialized_data.get('model')}, Anthropic will use {MODEL}"
            )

            print("\nSending to Anthropic API...")
            anthropic_response = get_response(
                ANTHROPIC_API_URL, anthropic_headers, anthropic_data
            )
            print(f"Anthropic status code: {anthropic_response.status_code}")
            if anthropic_response.status_code != 200:
                print(f"Anthropic error: {anthropic_response.text}")
                return False, None

            # Check if this is a thinking test
            has_thinking = (
                hasattr(request_data, "thinking")
                and request_data.thinking is not None
                and getattr(request_data.thinking, "type", None) == "enabled"
            )

            # Compare the responses
            result, warning = compare_responses(
                anthropic_response,
                proxy_response,
                check_tools=check_tools,
                compare_content=True,
                test_name=test_name,
                has_thinking=has_thinking,
            )

        else:  # Just validate the proxy response for a custom model
            print("\n(Direct conversion test, skipping comparison with Anthropic API)")
            try:
                proxy_json = proxy_response.json()
            except json.JSONDecodeError as e:
                print(f"\n--- PROXY RESPONSE JSON DECODE ERROR ---")
                print(f"Error: {e}")
                print(f"Proxy response status: {proxy_response.status_code}")
                print(f"Proxy response content: {proxy_response.text}")
                print(f"Proxy response headers: {dict(proxy_response.headers)}")
                return False, None
            print("\n--- Proxy Response Structure ---")
            print(
                json.dumps(
                    {k: v for k, v in proxy_json.items() if k != "content"}, indent=2
                )
            )

            # Basic validation
            assert proxy_json.get("role") == "assistant", (
                "Proxy role is not 'assistant'"
            )
            assert proxy_json.get("type") == "message", "Proxy type is not 'message'"
            assert "content" in proxy_json, "No content in Proxy response"
            assert isinstance(proxy_json["content"], list), (
                "Proxy content is not a list"
            )
            assert len(proxy_json["content"]) > 0, "Proxy content is empty"

            proxy_content = proxy_json["content"]
            has_text = any(
                item.get("type") == "text" and item.get("text")
                for item in proxy_content
            )
            has_tool_use = any(item.get("type") == "tool_use" for item in proxy_content)
            has_thinking = any(item.get("type") == "thinking" for item in proxy_content)

            # Check if this is a thinking test
            is_thinking_test = (
                hasattr(request_data, "thinking")
                and request_data.thinking is not None
                and getattr(request_data.thinking, "type", None) == "enabled"
            )

            if is_thinking_test:
                if not has_thinking:
                    assert False, (
                        "Thinking test but proxy response does not contain thinking block"
                    )
                else:
                    # Find thinking content
                    thinking_content = None
                    for item in proxy_content:
                        if item.get("type") == "thinking":
                            thinking_content = item.get("thinking")
                            break
                    thinking_length = len(thinking_content) if thinking_content else 0
                    print(
                        f"\n✅ Proxy response contains thinking block ({thinking_length} characters)"
                    )

            if check_tools:
                if not has_tool_use and not has_text:
                    assert False, "Expected tool use or text response"
                if has_tool_use:
                    print("\n✅ Proxy response contains tool use.")
                if has_text:
                    print("\n✅ Proxy response contains text.")
            else:
                assert has_text, "Expected text response"
                print("\n✅ Proxy response contains text.")

            result = True
            warning = None

        if result:
            if warning:
                print(f"\n⚠️ Test {test_name} passed with warning: {warning}")
                return True, warning
            else:
                print(f"\n✅ Test {test_name} passed!")
                return True, None
        else:
            print(f"\n❌ Test {test_name} failed!")
            return False, None

    except Exception as e:
        print(f"\n❌ Error in test {test_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False, None


def test_request(
    test_name, request_data, check_tools=False, compare_with_anthropic=True
):
    """Run a test with the given request data."""
    print(f"\n{'=' * 20} RUNNING TEST: {test_name} {'=' * 20}")

    # Check if this is a third-party model (not the default MODEL)
    model_name = (
        getattr(request_data, "model", "") if hasattr(request_data, "model") else ""
    )
    if model_name != MODEL and model_name != MODEL_THINKING:
        print(
            f"Third-party model ({model_name}) detected, using direct conversion test endpoint"
        )
        return test_direct_conversion(
            test_name, request_data, check_tools, compare_with_anthropic=False
        )

    # Convert Pydantic models to dictionaries for serialization
    serialized_data = serialize_request_data(request_data)

    # Log the request data including prompts for debugging
    print(
        f"\nRequest data:\n{json.dumps({k: v for k, v in serialized_data.items() if k != 'messages'}, indent=2)}"
    )

    # Log the actual prompts/messages for debugging
    if "messages" in serialized_data:
        print("\n--- PROMPTS/MESSAGES ---")
        for i, message in enumerate(serialized_data["messages"]):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            # Handle different content formats
            if isinstance(content, list):
                # Content blocks format
                print(f"Message {i + 1} ({role}):")
                for j, block in enumerate(content):
                    if block.get("type") == "text":
                        text_content = block.get("text", "")[
                            :200
                        ]  # Limit to first 200 chars
                        if len(block.get("text", "")) > 200:
                            text_content += "..."
                        print(f"  Block {j + 1} (text): {text_content}")
                    else:
                        print(
                            f"  Block {j + 1} ({block.get('type', 'unknown')}): {str(block)[:100]}..."
                        )
            elif isinstance(content, str):
                # Simple string content
                text_preview = content[:200]  # Limit to first 200 chars
                if len(content) > 200:
                    text_preview += "..."
                print(f"Message {i + 1} ({role}): {text_preview}")
            else:
                print(f"Message {i + 1} ({role}): {str(content)[:100]}...")
        print("--- END PROMPTS ---")

    # Log system message if present
    if "system" in serialized_data:
        system_preview = str(serialized_data["system"])[:200]
        if len(str(serialized_data["system"])) > 200:
            system_preview += "..."
        print(f"\nSystem message: {system_preview}")

    proxy_data = serialized_data.copy()

    try:
        print("\nSending to Proxy...")
        proxy_response = get_response(PROXY_API_URL, proxy_headers, proxy_data)
        print(f"Proxy status code: {proxy_response.status_code}")
        if proxy_response.status_code != 200:
            print(f"Proxy error: {proxy_response.text}")
            return False, None

        if compare_with_anthropic:
            # For Claude models, use the same model for Anthropic API comparison
            anthropic_data = serialized_data.copy()
            print("\nSending to Anthropic API...")
            anthropic_response = get_response(
                ANTHROPIC_API_URL, anthropic_headers, anthropic_data
            )
            print(f"Anthropic status code: {anthropic_response.status_code}")
            if anthropic_response.status_code != 200:
                print(f"Anthropic error: {anthropic_response.text}")
                return False, None

            # Check if this is a thinking test
            has_thinking = (
                hasattr(request_data, "thinking")
                and request_data.thinking is not None
                and getattr(request_data.thinking, "type", None) == "enabled"
            )

            # Compare the responses
            result, warning = compare_responses(
                anthropic_response,
                proxy_response,
                check_tools=check_tools,
                compare_content=True,
                test_name=test_name,
                has_thinking=has_thinking,
            )

        else:  # Just validate the proxy response for a custom model
            print("\n(Custom model test, skipping comparison with Anthropic API)")
            try:
                proxy_json = proxy_response.json()
            except json.JSONDecodeError as e:
                print(f"\n--- PROXY RESPONSE JSON DECODE ERROR ---")
                print(f"Error: {e}")
                print(f"Proxy response status: {proxy_response.status_code}")
                print(f"Proxy response content: {proxy_response.text}")
                print(f"Proxy response headers: {dict(proxy_response.headers)}")
                return False, None
            print("\n--- Proxy Response Structure ---")
            print(
                json.dumps(
                    {k: v for k, v in proxy_json.items() if k != "content"}, indent=2
                )
            )

            # Basic validation
            assert proxy_json.get("role") == "assistant", (
                "Proxy role is not 'assistant'"
            )
            assert proxy_json.get("type") == "message", "Proxy type is not 'message'"
            assert "content" in proxy_json, "No content in Proxy response"
            assert isinstance(proxy_json["content"], list), (
                "Proxy content is not a list"
            )
            assert len(proxy_json["content"]) > 0, "Proxy content is empty"

            proxy_content = proxy_json["content"]
            has_text = any(
                item.get("type") == "text" and item.get("text")
                for item in proxy_content
            )
            has_tool_use = any(item.get("type") == "tool_use" for item in proxy_content)
            has_thinking = any(item.get("type") == "thinking" for item in proxy_content)

            # Check if this is a thinking test
            is_thinking_test = (
                hasattr(request_data, "thinking")
                and request_data.thinking is not None
                and getattr(request_data.thinking, "type", None) == "enabled"
            )

            if is_thinking_test:
                if not has_thinking:
                    assert False, (
                        "Thinking test but proxy response does not contain thinking block"
                    )
                else:
                    # Find thinking content
                    thinking_content = None
                    for item in proxy_content:
                        if item.get("type") == "thinking":
                            thinking_content = item.get("thinking")
                            break
                    thinking_length = len(thinking_content) if thinking_content else 0
                    print(
                        f"\n✅ Proxy response contains thinking block ({thinking_length} characters)"
                    )

            if check_tools:
                if not has_tool_use and not has_text:
                    assert False, "Expected tool use or text response"
                if has_tool_use:
                    print("\n✅ Proxy response contains tool use.")
                if has_text:
                    print("\n✅ Proxy response contains text.")
            else:
                assert has_text, "Expected text response"
                print("\n✅ Proxy response contains text.")

            result = True
            warning = None

        if result:
            if warning:
                print(f"\n⚠️ Test {test_name} passed with warning: {warning}")
                return True, warning
            else:
                print(f"\n✅ Test {test_name} passed!")
                return True, None
        else:
            print(f"\n❌ Test {test_name} failed!")
            return False, None

    except Exception as e:
        print(f"\n❌ Error in test {test_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False, None


# ================= STREAMING TESTS =================


class StreamStats:
    """Track statistics about a streaming response."""

    def __init__(self):
        self.event_types = set()
        self.event_counts = {}
        self.first_event_time = None
        self.last_event_time = None
        self.total_chunks = 0
        self.events = []
        self.text_content = ""
        self.content_blocks = {}
        self.has_tool_use = False
        self.has_thinking = False
        self.thinking_content = ""
        self.has_error = False
        self.error_message = ""
        self.text_content_by_block = {}
        self.thinking_content_by_block = {}

    def add_event(self, event_data):
        """Track information about each received event."""
        now = datetime.now()
        if self.first_event_time is None:
            self.first_event_time = now
        self.last_event_time = now

        self.total_chunks += 1

        # Record event type and increment count
        if "type" in event_data:
            event_type = event_data["type"]
            self.event_types.add(event_type)
            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

            # Track specific event data
            if event_type == "content_block_start":
                block_idx = event_data.get("index")
                content_block = event_data.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    self.has_tool_use = True
                elif content_block.get("type") == "thinking":
                    self.has_thinking = True

                self.content_blocks[block_idx] = content_block
                self.text_content_by_block[block_idx] = ""
                self.thinking_content_by_block[block_idx] = ""

            elif event_type == "content_block_delta":
                block_idx = event_data.get("index")
                delta = event_data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    self.text_content += text
                    # Also track text by block ID
                    if block_idx in self.text_content_by_block:
                        self.text_content_by_block[block_idx] += text
                elif delta.get("type") == "thinking_delta":  # For thinking deltas
                    thinking_text = delta.get("thinking", "")
                    self.thinking_content += thinking_text
                    # Also track thinking by block ID
                    if block_idx in self.thinking_content_by_block:
                        self.thinking_content_by_block[block_idx] += thinking_text

        # Keep track of all events for debugging
        self.events.append(event_data)

    def get_duration(self):
        """Calculate the total duration of the stream in seconds."""
        if self.first_event_time is None or self.last_event_time is None:
            return 0
        return (self.last_event_time - self.first_event_time).total_seconds()

    def summarize(self):
        """Print a summary of the stream statistics."""
        print(f"Total chunks: {self.total_chunks}")
        print(f"Unique event types: {sorted(list(self.event_types))}")
        print(f"Event counts: {json.dumps(self.event_counts, indent=2)}")
        print(f"Duration: {self.get_duration():.2f} seconds")
        print(f"Has tool use: {self.has_tool_use}")
        print(f"Has thinking: {self.has_thinking}")

        # Print the first few lines of thinking content
        if self.thinking_content:
            max_preview_lines = 3
            thinking_preview = "\n".join(
                self.thinking_content.strip().split("\n")[:max_preview_lines]
            )
            print(
                f"Thinking preview ({len(self.thinking_content)} chars):\n{thinking_preview}"
            )
        elif self.has_thinking:
            print("Thinking detected but no content extracted")

        # Print the first few lines of text content
        if self.text_content:
            max_preview_lines = 5
            text_preview = "\n".join(
                self.text_content.strip().split("\n")[:max_preview_lines]
            )
            print(f"Text preview:\n{text_preview}")
        else:
            print("No text content extracted")

        if self.has_error:
            print(f"Error: {self.error_message}")


async def stream_response(url, headers, data, stream_name):
    """Send a streaming request and process the response."""
    print(f"\nStarting {stream_name} stream...")
    stats = StreamStats()
    error = None

    try:
        async with httpx.AsyncClient() as client:
            # Add stream flag to ensure it's streamed
            request_data = data.copy()
            request_data["stream"] = True

            # Use longer timeout for thinking requests
            timeout = 180 if request_data.get("thinking") else 30

            start_time = time.time()
            async with client.stream(
                "POST", url, json=request_data, headers=headers, timeout=timeout
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    stats.has_error = True
                    stats.error_message = (
                        f"HTTP {response.status_code}: {error_text.decode('utf-8')}"
                    )
                    error = stats.error_message
                    print(f"Error: {stats.error_message}")
                    return stats, error

                print(f"{stream_name} connected, receiving events...")

                # Process each chunk
                buffer = ""
                async for chunk in response.aiter_text():
                    if not chunk.strip():
                        continue

                    # Handle multiple events in one chunk
                    buffer += chunk
                    events = buffer.split("\n\n")

                    # Process all complete events
                    for event_text in events[
                        :-1
                    ]:  # All but the last (possibly incomplete) event
                        if not event_text.strip():
                            continue

                        # Parse server-sent event format
                        if "data: " in event_text:
                            # Extract the data part
                            data_parts = []
                            for line in event_text.split("\n"):
                                if line.startswith("data: "):
                                    data_part = line[len("data: ") :]
                                    # Skip the "[DONE]" marker
                                    if data_part == "[DONE]":
                                        break
                                    data_parts.append(data_part)

                            if data_parts:
                                try:
                                    event_data = json.loads("".join(data_parts))
                                    stats.add_event(event_data)
                                except json.JSONDecodeError as e:
                                    print(
                                        f"Error parsing event: {e}\nRaw data: {''.join(data_parts)}"
                                    )

                    # Keep the last (potentially incomplete) event for the next iteration
                    buffer = events[-1] if events else ""

                # Process any remaining complete events in the buffer
                if buffer.strip():
                    lines = buffer.strip().split("\n")
                    data_lines = [
                        line[len("data: ") :]
                        for line in lines
                        if line.startswith("data: ")
                    ]
                    if data_lines and data_lines[0] != "[DONE]":
                        try:
                            event_data = json.loads("".join(data_lines))
                            stats.add_event(event_data)
                        except:
                            pass

            elapsed = time.time() - start_time
            print(f"{stream_name} stream completed in {elapsed:.2f} seconds")
    except Exception as e:
        stats.has_error = True
        stats.error_message = str(e)
        error = str(e)
        print(f"Error in {stream_name} stream: {e}")

    return stats, error


def compare_stream_stats(
    anthropic_stats, proxy_stats, test_name=None, has_thinking=False
):
    """
    Compare the streaming statistics using Anthropic as ground truth.

    Returns (passed: bool, warning_reason: str|None) tuple.
    For behavioral difference tests, missing tool use becomes a warning instead of failure.
    For thinking tests, missing thinking blocks is a failure.
    """

    print("\n--- Stream Comparison ---")

    # Required events
    anthropic_missing = REQUIRED_EVENT_TYPES - anthropic_stats.event_types
    proxy_missing = REQUIRED_EVENT_TYPES - proxy_stats.event_types

    print(f"Anthropic missing event types: {anthropic_missing}")
    print(f"Proxy missing event types: {proxy_missing}")

    # Track test failures based on ground truth comparison
    test_passed = True
    failure_reasons = []
    warning_reason = None

    # Check if this is a behavioral difference test
    is_behavioral_test = test_name in BEHAVIORAL_DIFFERENCE_TESTS

    # Check if proxy has the required events
    if proxy_missing:
        print(f"❌ FAILURE: Proxy is missing required event types: {proxy_missing}")
        test_passed = False
        failure_reasons.append(f"Missing required event types: {proxy_missing}")
    else:
        print("✅ Proxy has all required event types")

    # Check for thinking blocks if this is a thinking test
    if has_thinking:
        # Display ground truth thinking content from Anthropic stream
        if anthropic_stats.has_thinking and anthropic_stats.thinking_content:
            anthropic_thinking_length = len(anthropic_stats.thinking_content)
            print(
                f"\n📚 Anthropic (ground truth) stream thinking blocks ({anthropic_thinking_length} characters)"
            )
            if anthropic_thinking_length > 0:
                anthropic_thinking_preview = (
                    anthropic_stats.thinking_content[:200] + "..."
                    if len(anthropic_stats.thinking_content) > 200
                    else anthropic_stats.thinking_content
                )
                print(
                    f"Anthropic stream thinking preview: {anthropic_thinking_preview}"
                )
        else:
            print("\n📚 Anthropic (ground truth) stream has no thinking blocks")

        if not proxy_stats.has_thinking:
            print(
                "❌ FAILURE: Thinking test but proxy stream does not contain thinking blocks"
            )
            test_passed = False
            failure_reasons.append(
                "Missing thinking blocks (thinking enabled but proxy doesn't provide thinking content)"
            )
        else:
            thinking_length = (
                len(proxy_stats.thinking_content) if proxy_stats.thinking_content else 0
            )
            print(
                f"✅ Proxy stream contains thinking blocks ({thinking_length} characters)"
            )
            if thinking_length > 0 and proxy_stats.thinking_content:
                thinking_preview = (
                    proxy_stats.thinking_content[:200] + "..."
                    if len(proxy_stats.thinking_content) > 200
                    else proxy_stats.thinking_content
                )
                print(f"Proxy stream thinking preview: {thinking_preview}")

    # Compare content using ground truth logic (with tolerance for tool-focused tests)
    if anthropic_stats.text_content and proxy_stats.text_content:
        anthropic_preview = "\n".join(
            anthropic_stats.text_content.strip().split("\n")[:5]
        )
        proxy_preview = "\n".join(proxy_stats.text_content.strip().split("\n")[:5])

        print("\n--- Anthropic Content Preview ---")
        print(anthropic_preview)

        print("\n--- Proxy Content Preview ---")
        print(proxy_preview)
    elif anthropic_stats.text_content and not proxy_stats.text_content:
        if proxy_stats.has_tool_use:
            # For streams with tool use: missing text content is acceptable
            new_warning = "Proxy missing text content but has tool use (acceptable for tool-focused streams)"
            print(f"\n⚠️ WARNING: {new_warning}")
            if warning_reason:
                warning_reason += f"; {new_warning}"
            else:
                warning_reason = new_warning
        else:
            print("\n❌ FAILURE: Anthropic has text content but proxy does not")
            test_passed = False
            failure_reasons.append(
                "Missing text content (Anthropic has text but proxy doesn't)"
            )
    elif not anthropic_stats.text_content and proxy_stats.text_content:
        print(
            "\n✅ Proxy has text content but Anthropic does not (acceptable - extra functionality)"
        )

    # Compare tool use using ground truth logic
    if anthropic_stats.has_tool_use and proxy_stats.has_tool_use:
        print("✅ Both have tool use")
    elif anthropic_stats.has_tool_use and not proxy_stats.has_tool_use:
        if is_behavioral_test:
            print(
                "⚠️ WARNING: Anthropic has tool use but proxy does not (behavioral difference)"
            )
            warning_reason = "Different tool usage pattern: Anthropic uses tools but proxy calculates directly"
        else:
            print("❌ FAILURE: Anthropic has tool use but proxy does not")
            test_passed = False
            failure_reasons.append(
                "Missing tool use (Anthropic has tool use but proxy doesn't)"
            )
    elif not anthropic_stats.has_tool_use and proxy_stats.has_tool_use:
        print(
            "✅ Proxy has tool use but Anthropic does not (acceptable - extra functionality)"
        )

    # Check for errors
    if proxy_stats.has_error:
        print(f"❌ FAILURE: Proxy stream had an error: {proxy_stats.error_message}")
        test_passed = False
        failure_reasons.append(f"Proxy stream error: {proxy_stats.error_message}")

    # Print failure summary if any
    if not test_passed:
        print(f"\n❌ Stream comparison failed due to missing features:")
        for reason in failure_reasons:
            print(f"  - {reason}")
    else:
        print("\n✅ Stream ground truth comparison passed")

    return test_passed, warning_reason


async def test_streaming(test_name, request_data, compare_with_anthropic=True):
    """Run a streaming test with the given request data."""
    print(f"\n{'=' * 20} RUNNING STREAMING TEST: {test_name} {'=' * 20}")

    # Convert Pydantic models to dictionaries for serialization
    serialized_data = serialize_request_data(request_data)

    # Log the request data including prompts for debugging
    print(
        f"\nRequest data:\n{json.dumps({k: v for k, v in serialized_data.items() if k != 'messages'}, indent=2)}"
    )

    # Log the actual prompts/messages for debugging
    if "messages" in serialized_data:
        print("\n--- PROMPTS/MESSAGES ---")
        for i, message in enumerate(serialized_data["messages"]):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            # Handle different content formats
            if isinstance(content, list):
                # Content blocks format
                print(f"Message {i + 1} ({role}):")
                for j, block in enumerate(content):
                    if block.get("type") == "text":
                        text_content = block.get("text", "")[
                            :200
                        ]  # Limit to first 200 chars
                        if len(block.get("text", "")) > 200:
                            text_content += "..."
                        print(f"  Block {j + 1} (text): {text_content}")
                    else:
                        print(
                            f"  Block {j + 1} ({block.get('type', 'unknown')}): {str(block)[:100]}..."
                        )
            elif isinstance(content, str):
                # Simple string content
                text_preview = content[:200]  # Limit to first 200 chars
                if len(content) > 200:
                    text_preview += "..."
                print(f"Message {i + 1} ({role}): {text_preview}")
            else:
                print(f"Message {i + 1} ({role}): {str(content)[:100]}...")
        print("--- END PROMPTS ---")

    # Log system message if present
    if "system" in serialized_data:
        system_preview = str(serialized_data["system"])[:200]
        if len(str(serialized_data["system"])) > 200:
            system_preview += "..."
        print(f"\nSystem message: {system_preview}")

    proxy_data = serialized_data.copy()
    if not proxy_data.get("stream"):
        proxy_data["stream"] = True

    try:
        proxy_stats, proxy_error = await stream_response(
            PROXY_API_URL, proxy_headers, proxy_data, "Proxy"
        )

        if proxy_error:
            print(f"\n❌ Test {test_name} failed! Proxy had an error: {proxy_error}")
            return False, None

        print("\n--- Proxy Stream Statistics ---")
        proxy_stats.summarize()

        if not proxy_stats.total_chunks > 0:
            print(f"\n❌ Test {test_name} failed! Proxy stream was empty.")
            return False, None

        # Check if this is a third-party model (not the default MODEL)
        model_name = serialized_data.get("model", "")
        if model_name != MODEL:
            print(
                f"Third-party model streaming test ({model_name}): Skipping comparison with Anthropic API"
            )
            compare_with_anthropic = False

        if compare_with_anthropic:
            anthropic_data = serialized_data.copy()
            if not anthropic_data.get("stream"):
                anthropic_data["stream"] = True

            anthropic_stats, anthropic_error = await stream_response(
                ANTHROPIC_API_URL, anthropic_headers, anthropic_data, "Anthropic"
            )

            if anthropic_error:
                print(f"\n⚠️ Anthropic stream had an error: {anthropic_error}")
                print(
                    f"\n✅ Test {test_name} passed! (Proxy worked even though Anthropic failed)"
                )
                return True, None

            print("\n--- Anthropic Stream Statistics ---")
            anthropic_stats.summarize()

            # Check if this is a thinking test
            has_thinking = (
                hasattr(request_data, "thinking")
                and request_data.thinking is not None
                and getattr(request_data.thinking, "type", None) == "enabled"
            )

            result, warning = compare_stream_stats(
                anthropic_stats, proxy_stats, test_name, has_thinking
            )

        else:  # Custom model, no comparison
            print("\n(Custom model test, skipping comparison with Anthropic stream)")
            # Basic validation of proxy stream
            if proxy_stats.has_error:
                print(
                    f"\n❌ Test {test_name} failed! Proxy stream had an error: {proxy_stats.error_message}"
                )
                return False, None

            # Check if this is a thinking test for custom model
            is_thinking_test = (
                hasattr(request_data, "thinking")
                and request_data.thinking is not None
                and getattr(request_data.thinking, "type", None) == "enabled"
            )

            if is_thinking_test:
                if not proxy_stats.has_thinking:
                    print(
                        "\n❌ Test failed! Thinking test but proxy stream does not contain thinking blocks"
                    )
                    return False, None
                else:
                    thinking_length = (
                        len(proxy_stats.thinking_content)
                        if proxy_stats.thinking_content
                        else 0
                    )
                    print(
                        f"\n✅ Proxy stream contains thinking blocks ({thinking_length} characters)"
                    )

            result = True
            warning = None

        if result:
            if warning:
                print(f"\n⚠️ Test {test_name} passed with warning: {warning}")
                return True, warning
            else:
                print(f"\n✅ Test {test_name} passed!")
                return True, None
        else:
            print(f"\n❌ Test {test_name} failed!")
            return False, None

    except Exception as e:
        print(f"\n❌ Error in test {test_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return False, None


# ================= MAIN =================


async def run_tests(args):
    """Run all tests based on command-line arguments."""

    # Handle --list-tests
    if args.list_tests:
        print("Available tests:")
        for test_name, test_data in TEST_SCENARIOS.items():
            test_type = (
                "streaming"
                if (hasattr(test_data, "stream") and test_data.stream)
                else "non-streaming"
            )
            has_tools = hasattr(test_data, "tools") and test_data.tools is not None
            tools_info = " (with tools)" if has_tools else " (no tools)"
            print(f"  {test_name:<20} - {test_type}{tools_info}")
        return True

    # Validate custom model if specified
    if args.model:
        # Load custom models config
        try:
            with open("custom_models.yaml", "r") as f:
                custom_models = yaml.safe_load(f)

            # Check if model exists in config
            model_exists = any(m["model_id"] == args.model for m in custom_models)
            if not model_exists:
                print(f"Error: Model '{args.model}' not found in custom_models.yaml")
                return False

            # Override model in test scenarios (but skip custom models)
            for test_name, test_data in TEST_SCENARIOS.items():
                # Don't override custom models with --model parameter
                current_model = (
                    getattr(test_data, "model", "")
                    if hasattr(test_data, "model")
                    else test_data.get("model", "")
                )
                if not current_model.startswith("custom/"):
                    if hasattr(test_data, "model"):
                        test_data.model = args.model
                    else:
                        TEST_SCENARIOS[test_name]["model"] = args.model

            print(f"\nUsing custom model: {args.model}")

        except Exception as e:
            print(f"Error loading custom_models.yaml: {str(e)}")
            return False

    # Track test results
    results = {}
    warnings = {}

    # Determine if we should compare with Anthropic
    compare_with_anthropic = not args.model or args.compare

    # Filter tests if --test or --only specified
    if args.test or args.only:
        if args.test:
            if args.test not in TEST_SCENARIOS:
                print(f"Error: Test '{args.test}' not found.")
                print("Available tests:")
                for test_name in TEST_SCENARIOS.keys():
                    print(f"  {test_name}")
                return False

            test_names = [args.test]
        else:  # --only
            test_names = [
                name
                for name in TEST_SCENARIOS.keys()
                if args.only.lower() in name.lower()
            ]
            if not test_names:
                print(f"Error: No tests found containing keyword '{args.only}'")
                print("Available tests:")
                for test_name in TEST_SCENARIOS.keys():
                    print(f"  {test_name}")
                return False
            print(f"Running tests containing keyword: '{args.only}'")
            print(f"Matched tests: {', '.join(test_names)}")

        # Run all matched tests
        for test_name in test_names:
            test_data = TEST_SCENARIOS[test_name]

            # Run non-streaming version if not streaming-only
            if not args.streaming_only and not test_name.endswith("_stream"):
                print(f"\n\n=========== RUNNING TEST: {test_name} ===========\n")
                check_tools = (
                    hasattr(test_data, "tools") and test_data.tools is not None
                )
                result, warning = test_request(
                    test_name,
                    test_data,
                    check_tools=check_tools,
                    compare_with_anthropic=compare_with_anthropic,
                )
                results[test_name] = result
                if warning:
                    warnings[test_name] = warning

            # Run streaming version if not no-streaming
            if not args.no_streaming and test_name.endswith("_stream"):
                print(
                    f"\n\n=========== RUNNING STREAMING TEST: {test_name} ===========\n"
                )
                # Create streaming version of test data
                if isinstance(test_data, MessagesRequest):
                    streaming_data = test_data.model_copy(update={"stream": True})
                else:
                    streaming_data = test_data.copy()
                    streaming_data["stream"] = True
                result, warning = await test_streaming(
                    test_name,
                    streaming_data,
                    compare_with_anthropic=compare_with_anthropic,
                )
                results[f"{test_name}"] = result
                if warning:
                    warnings[f"{test_name}"] = warning

    else:
        # Run all tests based on filters

        # First run non-streaming tests
        if not args.streaming_only:
            print("\n\n=========== RUNNING NON-STREAMING TESTS ===========\n")
            for test_name, test_data in TEST_SCENARIOS.items():
                # Skip streaming tests
                if hasattr(test_data, "stream") and test_data.stream:
                    continue

                # Skip tool tests if requested
                if args.simple and hasattr(test_data, "tools") and test_data.tools:
                    continue

                # Skip non-tool tests if tools_only
                if args.tools_only and (
                    not hasattr(test_data, "tools") or not test_data.tools
                ):
                    continue

                # Skip non-thinking tests if thinking_only
                if args.thinking_only and (
                    not hasattr(test_data, "thinking") or not test_data.thinking
                ):
                    continue

                # Run the test
                check_tools = (
                    hasattr(test_data, "tools") and test_data.tools is not None
                )
                result, warning = test_request(
                    test_name,
                    test_data,
                    check_tools=check_tools,
                    compare_with_anthropic=compare_with_anthropic,
                )
                results[test_name] = result
                if warning:
                    warnings[test_name] = warning

        # Now run streaming tests
        if not args.no_streaming:
            print("\n\n=========== RUNNING STREAMING TESTS ===========\n")
            for test_name, test_data in TEST_SCENARIOS.items():
                # Only select streaming tests, or force streaming
                if not (
                    hasattr(test_data, "stream") and test_data.stream
                ) and not test_name.endswith("_stream"):
                    continue

                # Skip tool tests if requested
                if args.simple and hasattr(test_data, "tools") and test_data.tools:
                    continue

                # Skip non-tool tests if tools_only
                if args.tools_only and (
                    not hasattr(test_data, "tools") or not test_data.tools
                ):
                    continue

                # Skip non-thinking tests if thinking_only
                if args.thinking_only and (
                    not hasattr(test_data, "thinking") or not test_data.thinking
                ):
                    continue

                # Run the streaming test
                result, warning = await test_streaming(
                    test_name, test_data, compare_with_anthropic=compare_with_anthropic
                )
                results[f"{test_name}"] = result
                if warning:
                    warnings[f"{test_name}"] = warning

    # Print summary
    print("\n\n=========== TEST SUMMARY ===========\n")
    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        if test in warnings:
            status += f" (⚠️ WARNING)"
        print(f"{test}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # Print warnings summary
    if warnings:
        print(f"\n⚠️ BEHAVIORAL DIFFERENCES DETECTED ({len(warnings)} warnings)")
        print("=" * 60)
        print(
            "The following tests passed but showed different behavior compared to Claude:"
        )
        for test_name, warning_msg in warnings.items():
            print(f"  • {test_name}: {warning_msg}")
        print(
            "\nNote: These warnings indicate that your proxy model handles tool usage"
        )
        print("differently than Claude models. This is expected behavior when using")
        print("different underlying models like DeepSeek, which may calculate simple")
        print("math directly instead of using tools.")

    if passed == total:
        if warnings:
            print(
                f"\n🎉 All tests passed! ({len(warnings)} behavioral differences noted)"
            )
        else:
            print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        return False


async def main():
    # Check that API key is set
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set in .env file")
        return

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the Claude-on-OpenAI proxy")
    parser.add_argument(
        "--no-streaming", action="store_true", help="Skip streaming tests"
    )
    parser.add_argument(
        "--streaming-only", action="store_true", help="Only run streaming tests"
    )
    parser.add_argument(
        "--simple", action="store_true", help="Only run simple tests (no tools)"
    )
    parser.add_argument("--tools-only", action="store_true", help="Only run tool tests")
    parser.add_argument(
        "--thinking-only", action="store_true", help="Only run thinking tests"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run only a specific test by name (e.g., 'calculator', 'todo_write')",
    )
    parser.add_argument(
        "--only",
        type=str,
        help="Run all tests containing the specified keyword in their name",
    )
    parser.add_argument(
        "--list-tests", action="store_true", help="List all available tests"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specify a custom model ID from custom_models.yaml for testing",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare responses with official model when using custom model",
    )
    args = parser.parse_args()

    # Run tests
    success = await run_tests(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
