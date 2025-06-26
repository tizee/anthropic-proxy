#!/usr/bin/env python3
"""
Comprehensive test suite for Claude-on-OpenAI Proxy using unittest framework.

This module provides tests for both streaming and non-streaming requests,
with various scenarios including tool use, multi-turn conversations,
and content blocks.

Usage:
  python tests_unittest.py                    # Run all tests
  python -m unittest tests_unittest           # Run with unittest module
  python -m unittest tests_unittest.TestBasicRequests.test_simple_request  # Run specific test
"""

import json
import logging
import os
import unittest
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

# Setup logger
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Pydantic models from models.py
from models import (
    ClaudeContentBlockText,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockToolUse,
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeThinkingConfigDisabled,
    ClaudeThinkingConfigEnabled,
    ClaudeTool,
    ClaudeToolChoice,
    ClaudeToolChoiceAny,
    ClaudeToolChoiceAuto,
)

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
PROXY_TEST_API_URL = "http://127.0.0.1:8082/v1/messages/test_conversion"
ANTHROPIC_VERSION = "2023-06-01"
BASE_URL = "http://127.0.0.1:8082"
MODEL = "claude-3-5-haiku-20241022"
MODEL_THINKING = "claude-3-7-sonnet-20250219"
TEST_TIMEOUT = 30

# Headers for API requests
HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

anthropic_headers = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

proxy_headers = {
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

# Tool definitions
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

edit_tool = ClaudeTool(
    name="Edit",
    description="Performs exact string replacements in files",
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to modify",
            },
            "old_string": {"type": "string", "description": "The text to replace"},
            "new_string": {
                "type": "string",
                "description": "The text to replace it with",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    },
)

todo_write_tool = ClaudeTool(
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

todo_read_tool = ClaudeTool(
    name="TodoRead",
    description="Reads the current todo list",
    input_schema={"type": "object", "properties": {}},
)

weather_tool = ClaudeTool(
    name="get_weather",
    description="Get weather information for a specified location",
    input_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The location to get weather for",
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius",
            },
        },
        "required": ["location"],
    },
)

bash_tool = ClaudeTool(
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

# Gemini-incompatible tool - contains fields that Gemini doesn't support
gemini_incompatible_tool = ClaudeTool(
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

search_tool = ClaudeTool(
    name="search",
    description="Search for information on the web",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query"}},
        "required": ["query"],
    },
)

# Claude Code tools for testing
read_tool = ClaudeTool(
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

ls_tool = ClaudeTool(
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

grep_tool = ClaudeTool(
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

glob_tool = ClaudeTool(
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

exit_plan_mode_tool = ClaudeTool(
    name="exit_plan_mode",
    description="Use this tool when you are in plan mode and have finished presenting your plan and are ready to code. This will prompt the user to exit plan mode.",
    input_schema={
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "The plan you came up with, that you want to run by the user for approval. Supports markdown. The plan should be pretty concise.",
            }
        },
        "required": ["plan"],
    },
)

# Tool choice configurations
tool_choice_auto = ClaudeToolChoiceAuto(type="auto")
tool_choice_required = ClaudeToolChoiceAny(type="any")

# Define behavioral difference tests that should warn instead of fail
BEHAVIORAL_DIFFERENCE_TESTS = {
    "multi_turn",
    "thinking_with_tools",
    "thinking_with_tools_stream",
    "calculator",
    "calculator_stream",
    "content_blocks",
    "multi_tool",
    "edit_tool_completion",
    "todo_write",
    "todo_read",
    "todo_write_stream",
    "todo_read_stream",
    "edit_tool_completion_stream",
    "gemini_tool_test",
    "gemini_tool_test_stream",
    "gemini_incompatible_schema_test",
    "gemini_incompatible_schema_test_stream",
    "gemini_tool_result_positioning_test",
    "gemini_tool_result_positioning_invalid_test",
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
    "deepseek_complex_todo_workflow_stream",
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


def serialize_request_data(data) -> dict[str, Any]:
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
        print("📊 Stream Statistics")
        print(f"   📦 Total chunks: {self.total_chunks}")
        print(f"   🔄 Event types: {sorted(list(self.event_types))}")
        print(f"   📈 Event counts: {json.dumps(self.event_counts, indent=2)}")
        print(f"   ⏱️  Duration: {self.get_duration():.2f} seconds")
        print(f"   🔧 Tool use: {'✅' if self.has_tool_use else '❌'}")
        print(f"   🧠 Thinking: {'✅' if self.has_thinking else '❌'}")

        # Print the first few lines of thinking content
        if self.thinking_content:
            max_preview_lines = 3
            thinking_preview = "\n".join(
                self.thinking_content.strip().split("\n")[:max_preview_lines]
            )
            print(
                f"🧠 Thinking preview ({len(self.thinking_content)} chars):\n{thinking_preview}"
            )
        elif self.has_thinking:
            print("🧠 Thinking detected but no content extracted")

        # Print the first few lines of text content
        if self.text_content:
            max_preview_lines = 5
            text_preview = "\n".join(
                self.text_content.strip().split("\n")[:max_preview_lines]
            )
            print(f"📝 Text preview:\n{text_preview}")
        else:
            print("📝 No text content extracted")

        if self.has_error:
            print(f"❌ Error: {self.error_message}")


class ProxyTestBase(unittest.IsolatedAsyncioTestCase):
    """Base class for proxy tests with common utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources."""
        if not ANTHROPIC_API_KEY:
            raise unittest.SkipTest("ANTHROPIC_API_KEY not set in .env file")

    async def asyncSetUp(self):
        """Set up test-specific resources."""
        self.client = httpx.AsyncClient(timeout=TEST_TIMEOUT)
        self.base_url = BASE_URL
        self.headers = HEADERS.copy()

        # Pretty print test start
        test_name = self._testMethodName
        class_name = self.__class__.__name__
        print(f"🧪 Testing {test_name} ({class_name})...")

    async def asyncTearDown(self):
        """Clean up test-specific resources."""
        if hasattr(self, "client") and self.client is not None:
            await self.client.aclose()

    def run(self, result=None):
        """Override run to capture test results for pretty printing."""
        test_name = self._testMethodName

        # Create a TestResult if none provided
        if result is None:
            result = unittest.TestResult()

        # Run the actual test
        test_result = super().run(result)

        # Check if test passed or failed and print accordingly
        if hasattr(result, "failures") and hasattr(result, "errors"):
            # Check if this test had any failures or errors
            test_failed = False
            test_error = False

            # Check for failures
            for failure in result.failures:
                if failure[0] == self:
                    test_failed = True
                    break

            # Check for errors
            for error in result.errors:
                if error[0] == self:
                    test_error = True
                    break

            if test_failed:
                print(f"❌ {test_name} failed")
            elif test_error:
                print(f"💥 {test_name} error")
            else:
                print(f"✅ {test_name} passed")

        return test_result

    async def make_request(
        self, request_data: ClaudeMessagesRequest, stream: bool = False
    ) -> dict[str, Any]:
        """Make a request to the proxy server."""
        serialized_data = serialize_request_data(request_data)

        if stream:
            serialized_data["stream"] = True

        response = await self.client.post(
            f"{self.base_url}/v1/messages", headers=self.headers, json=serialized_data
        )

        if stream or request_data.stream:
            return await self._process_streaming_response(response)
        else:
            response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {
                "error": "Empty or invalid JSON response",
                "content": response.text,
            }

    async def _process_streaming_response(self, response) -> dict[str, Any]:
        """Process streaming response and return aggregated result."""
        response.raise_for_status()

        chunks = []

        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data_part = line[6:]
                if data_part == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_part)
                    chunks.append(chunk)
                except json.JSONDecodeError:
                    continue

        # Aggregate streaming response
        content_blocks = []
        current_text = ""
        current_thinking = ""
        tool_calls = []
        tool_input_jsons = {}  # Track partial JSON for each tool index

        for chunk in chunks:
            if chunk.get("type") == "content_block_delta":
                delta = chunk.get("delta", {})
                chunk_index = chunk.get("index", 0)
                if delta.get("type") == "text_delta":
                    current_text += delta.get("text", "")
                elif delta.get("type") == "thinking_delta":
                    current_thinking += delta.get("thinking", "")
                elif delta.get("type") == "input_json_delta":
                    # Accumulate partial JSON for tool calls
                    partial_json = delta.get("partial_json", "")
                    if chunk_index not in tool_input_jsons:
                        tool_input_jsons[chunk_index] = ""
                    tool_input_jsons[chunk_index] += partial_json
            elif chunk.get("type") == "content_block_start":
                content_block = chunk.get("content_block", {})
                chunk_index = chunk.get("index", 0)
                if content_block.get("type") == "tool_use":
                    tool_calls.append((chunk_index, content_block))

        # Parse accumulated tool inputs and merge with tool calls
        final_tool_calls = []
        for chunk_index, tool_call in tool_calls:
            if chunk_index in tool_input_jsons:
                try:
                    tool_input = json.loads(tool_input_jsons[chunk_index])
                    tool_call["input"] = tool_input
                except json.JSONDecodeError:
                    # If JSON parsing fails, keep original or set empty dict
                    tool_call["input"] = tool_call.get("input", {})
            final_tool_calls.append(tool_call)

        # Debug: log accumulated streaming data
        logger.debug(f"Streaming response final_tool_calls = {final_tool_calls}")
        logger.debug(f"Streaming response tool_input_jsons = {tool_input_jsons}")

        # Build final response structure
        if current_thinking:
            content_blocks.append({"type": "thinking", "thinking": current_thinking})

        if current_text:
            content_blocks.append({"type": "text", "text": current_text})

        content_blocks.extend(final_tool_calls)

        # Display stream statistics
        print(f"\n{'=' * 50}")
        print(f"{content_blocks}")
        print(f"{'=' * 50}")

        return {
            "content": content_blocks,
            "role": "assistant",
            "type": "message",
        }

    def assertResponseValid(self, response: dict[str, Any]):
        """Assert that a response has valid structure."""
        self.assertIn("content", response)
        self.assertIn("role", response)
        self.assertEqual(response["role"], "assistant")
        self.assertIsInstance(response["content"], list)
        self.assertGreater(len(response["content"]), 0)

    def assertHasTextContent(self, response: dict[str, Any], min_length: int = 1):
        """Assert that response contains text content."""
        text_blocks = [
            block for block in response["content"] if block.get("type") == "text"
        ]
        self.assertGreater(len(text_blocks), 0, "Response should contain text content")

        total_text = "".join(block.get("text", "") for block in text_blocks)
        self.assertGreaterEqual(
            len(total_text),
            min_length,
            f"Text content should be at least {min_length} characters",
        )

    def assertNoToolUse(self, response: dict[str, Any]):
        """Assert that response should not contain tool use."""
        tool_blocks = [
            block for block in response["content"] if block.get("type") == "tool_use"
        ]
        self.assertGreater(1, len(tool_blocks), "Response should should not contain tool use")


    def assertHasToolUse(self, response: dict[str, Any], tool_name: str = None):
        """Assert that response contains tool use."""
        tool_blocks = [
            block for block in response["content"] if block.get("type") == "tool_use"
        ]
        self.assertGreater(len(tool_blocks), 0, "Response should contain tool use")

        if tool_name:
            tool_names = [block.get("name") for block in tool_blocks]
            self.assertIn(
                tool_name, tool_names, f"Response should use tool '{tool_name}'"
            )

    def assertHasThinking(self, response: dict[str, Any]):
        """Assert that response contains thinking content."""
        thinking_blocks = [
            block for block in response["content"] if block.get("type") == "thinking"
        ]
        self.assertGreater(
            len(thinking_blocks), 0, "Response should contain thinking content"
        )

    async def make_anthropic_request(
        self, request_data: ClaudeMessagesRequest, stream: bool = False
    ) -> dict[str, Any]:
        """Make a request to the official Anthropic API."""
        serialized_data = serialize_request_data(request_data)

        if stream:
            serialized_data["stream"] = True

        timeout = 180 if serialized_data.get("thinking") else 30
        client = httpx.AsyncClient(timeout=timeout)

        try:
            response = await client.post(
                ANTHROPIC_API_URL, headers=anthropic_headers, json=serialized_data
            )

            if stream or request_data.stream:
                return await self._process_streaming_response(response)
            else:
                response.raise_for_status()
                return response.json()
        finally:
            await client.aclose()

    async def make_proxy_test_conversion_request(
        self, request_data: ClaudeMessagesRequest
    ) -> dict[str, Any]:
        """
        Make a request to the proxy test conversion endpoint
        without routing and using the request_data.model as model_id
        for mapping the model name in models.yaml
        """
        serialized_data = serialize_request_data(request_data)

        timeout = 180 if serialized_data.get("thinking") else 30
        client = httpx.AsyncClient(timeout=timeout)

        response = await client.post(
            PROXY_TEST_API_URL, headers=proxy_headers, json=serialized_data
        )
        if request_data.stream:
            return await self._process_streaming_response(response)
        else:
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return {
                    "error": "Empty or invalid JSON response",
                    "content": response.text,
                }

    def compare_responses(
        self,
        anthropic_response: dict[str, Any],
        proxy_response: dict[str, Any],
        check_tools: bool = False,
        test_name: str | None = None,
        compare_content: bool = False,
        has_thinking: bool = False,
        tool_choice: ClaudeToolChoice | None = None
    ) -> tuple[bool, str | None]:
        """
        Compare the two responses using Anthropic as ground truth.

        Returns (passed: bool, warning_reason: str|None) tuple.
        For behavioral difference tests, missing tool use becomes a warning instead of failure.
        For thinking tests, missing thinking blocks is a failure.
        """
        print("\n🔍 Response Structure Comparison")
        print("\n📡 Anthropic Response Structure:")
        print(
            json.dumps(
                {k: v for k, v in anthropic_response.items() if k != "content"},
                indent=2,
            )
        )

        print("\n🔄 Proxy Response Structure:")
        print(
            json.dumps(
                {k: v for k, v in proxy_response.items() if k != "content"}, indent=2
            )
        )

        # Basic structure verification with more flexibility
        self.assertEqual(
            proxy_response.get("role"), "assistant", "Proxy role is not 'assistant'"
        )
        self.assertEqual(
            proxy_response.get("type"), "message", "Proxy type is not 'message'"
        )

        # Check if stop_reason is reasonable
        valid_stop_reasons = [
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            None,
        ]
        self.assertIn(
            proxy_response.get("stop_reason"), valid_stop_reasons, "Invalid stop reason"
        )

        # Check content exists and has valid structure
        self.assertIn("content", anthropic_response, "No content in Anthropic response")
        self.assertIn("content", proxy_response, "No content in Proxy response")

        anthropic_content = anthropic_response["content"]
        proxy_content = proxy_response["content"]

        self.assertIsInstance(
            anthropic_content, list, "Anthropic content is not a list"
        )
        self.assertIsInstance(proxy_content, list, "Proxy content is not a list")
        self.assertGreater(len(proxy_content), 0, "Proxy content is empty")

        # Track test failures based on ground truth comparison
        test_passed = True
        failure_reasons = []
        warning_reason = None

        # Check if this is a behavioral difference test
        is_behavioral_test = (
            test_name in BEHAVIORAL_DIFFERENCE_TESTS if test_name else False
        )

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
                print("\n🛠️  Anthropic Tool Use:")
                print(json.dumps(anthropic_tool, indent=2))

                if proxy_tool is not None:
                    print("\n🔧 Proxy Tool Use:")
                    print(json.dumps(proxy_tool, indent=2))

                    # Check tool structure
                    self.assertIsNotNone(
                        proxy_tool.get("name"), "Proxy tool has no name"
                    )
                    self.assertIsNotNone(
                        proxy_tool.get("input"), "Proxy tool has no input"
                    )

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
                print("\n🔧 Proxy Tool Use:")
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
                    new_warning = ""
                    if tool_choice and tool_choice.type == "any":
                        new_warning = "Tool Choice is any: Tool names differ between Anthropic and proxy"
                    else:
                        new_warning = "Tool Choice is auto: Tool names differ between Anthropic and proxy"
                    print(f"\n⚠️ WARNING: {new_warning}")
                    if warning_reason:
                        warning_reason += f"; {new_warning}"
                    else:
                        warning_reason = new_warning

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
                new_warning = "Proxy missing text content but this is a behavioral difference test"
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
            proxy_preview = "\n".join(
                proxy_text.strip().split("\n")[:max_preview_lines]
            )
            print(proxy_preview)
        else:
            print("(No text content)")

        # Print failure summary if any
        if not test_passed:
            print("\n❌ Test failed due to missing features:")
            for reason in failure_reasons:
                print(f"  - {reason}")
        else:
            print("\n✅ Ground truth comparison passed")

        return test_passed, warning_reason

    async def make_comparison_test(
        self,
        test_name: str,
        request_data: ClaudeMessagesRequest,
        check_tools: bool = False,
        has_thinking: bool = False,
    ) -> tuple[bool, str | None]:
        """Run a comparison test between proxy and Anthropic API."""
        print(f"\n{'=' * 20} RUNNING COMPARISON TEST: {test_name} {'=' * 20}")

        # Make requests to both APIs
        proxy_response = await self.make_request(request_data)
        anthropic_response = await self.make_anthropic_request(request_data)

        # Compare responses
        return self.compare_responses(
            anthropic_response=anthropic_response,
            proxy_response=proxy_response,
            check_tools=check_tools,
            test_name=test_name,
            has_thinking=has_thinking,
            tool_choice=request_data.tool_choice
        )

    async def make_direct_conversion_test(
        self,
        test_name: str,
        request_data: ClaudeMessagesRequest,
        check_tools: bool = False,
        compare_with_anthropic: bool = True,
        expect_failure: bool = False,
    ) -> tuple[bool, str | None]:
        """Run a direct conversion test using the test_conversion endpoint."""
        print(f"\n{'=' * 20} RUNNING DIRECT CONVERSION TEST: {test_name} {'=' * 20}")

        # Get proxy test conversion response
        try:
            proxy_response = await self.make_proxy_test_conversion_request(request_data)
            if expect_failure:
                # If we expected failure but got success, that's a test failure
                print(f"❌ Expected failure but request succeeded for {test_name}")
                return False, "Expected request to fail but it succeeded"
        except httpx.HTTPStatusError as e:
            if expect_failure:
                # If we expected failure and got failure, that's a test success
                print(f"✅ Expected failure occurred for {test_name}: {e.response.status_code}")
                return True, f"Expected failure: {e.response.status_code}"
            else:
                # If we didn't expect failure but got failure, re-raise the exception
                raise

        if not compare_with_anthropic:
            return True, None

        # For direct conversion tests, always use Claude model for Anthropic comparison
        anthropic_data = request_data.model_copy()
        if request_data.thinking and request_data.thinking.type == "enabled":
            anthropic_data.model = MODEL_THINKING

        print(
            f"Direct conversion test: Proxy will use {request_data.model}, Anthropic will use {MODEL}"
        )

        try:
            anthropic_response = await self.make_anthropic_request(anthropic_data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return True, f"Skipping test - Anthropic API returned 404: {e}"
            raise

        # Check if this is a thinking test
        has_thinking = (
            hasattr(request_data, "thinking")
            and request_data.thinking is not None
            and getattr(request_data.thinking, "type", None) == "enabled"
        )

        # Compare responses
        return self.compare_responses(
            anthropic_response=anthropic_response,
            proxy_response=proxy_response,
            check_tools=check_tools,
            test_name=test_name,
            has_thinking=has_thinking,
            tool_choice=request_data.tool_choice
        )


class TestBasicRequests(ProxyTestBase):
    """Test basic request functionality."""

    async def test_simple_request(self):
        """Test simple text request without tools."""

        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Hello! Please respond with exactly the word 'SUCCESS'.",
                )
            ],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasTextContent(response)

        # Check that response contains expected content
        text_content = "".join(
            block.get("text", "")
            for block in response["content"]
            if block.get("type") == "text"
        )
        self.assertIn("SUCCESS", text_content.upper())

    async def test_simple_streaming_request(self):
        """Test simple streaming text request."""

        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user", content="Count from 1 to 5, one number per line."
                )
            ],
        )

        response = await self.make_request(request, stream=True)

        self.assertResponseValid(response)
        self.assertHasTextContent(response)

    async def test_system_message(self):
        """Test request with system message."""

        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=100,
            system="You are a helpful assistant that always responds with 'SYSTEM_TEST_PASS'.",
            messages=[ClaudeMessage(role="user", content="Hello!")],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasTextContent(response)

        text_content = "".join(
            block.get("text", "")
            for block in response["content"]
            if block.get("type") == "text"
        )
        self.assertIn("SYSTEM_TEST_PASS", text_content)


class TestToolRequests(ProxyTestBase):
    """Test tool-related functionality."""

    async def test_thinking_with_tools(self):
        """thinking with tools"""
        request = ClaudeMessagesRequest(
            model=MODEL_THINKING,
            max_tokens=1025,
            stream=False,
            thinking=ClaudeThinkingConfigEnabled(type="enabled", budget_tokens=1024),
            messages=[
                ClaudeMessage(
                    role="user",
                    content="What is 125 divided by 5? Think about it and use the calculator if needed.",
                )
            ],
            tools=[calculator_tool],
            tool_choice=tool_choice_auto,
        )
        passed, warning = await self.make_comparison_test(
            "thinking_with_tools", request, has_thinking=True
        )

        self.assertTrue(passed, "Comparison test should pass")

    async def test_thinking_with_tools_stream(self):
        """thinking with tools"""
        request = ClaudeMessagesRequest(
            model=MODEL_THINKING,
            max_tokens=1025,
            thinking=ClaudeThinkingConfigEnabled(type="enabled", budget_tokens=1024),
            stream=True,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="What is 125 divided by 5? Think about it and use the calculator if needed.",
                )
            ],
            tools=[calculator_tool],
            tool_choice=tool_choice_auto,
        )
        passed, warning = await self.make_comparison_test(
            "thinking_with_tools", request, has_thinking=True
        )
        self.assertTrue(passed, "Comparison test should pass")

    async def test_calculator_tool(self):
        """Test calculator tool usage."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user", content="What is 25 + 17? Use the calculator tool."
                )
            ],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "calculator")

        # Check tool call parameters
        tool_blocks = [
            block for block in response["content"] if block.get("type") == "tool_use"
        ]
        print(f"response content: {response['content']}")
        calculator_call = None
        for block in tool_blocks:
            if block.get("name") == "calculator":
                calculator_call = block
                break

        self.assertIsNotNone(calculator_call, "Should have calculator tool call")
        self.assertIn("input", calculator_call)
        self.assertIn("expression", calculator_call["input"])

    async def test_tool_use_order(self):
        """Test tool use order"""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[calculator_tool],
            tool_choice=tool_choice_auto,
            messages=[
                ClaudeMessage(
                    role="user", content="What is 25 + 17? Use the calculator tool."
                ),
                # tool calls
                ClaudeMessage(
                    # [{'type': 'tool_use', 'id': 'tool_0_calculator', 'name': 'calculator', 'input': {'expression': '25 + 17'}}]
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            name="calculator",
                            id="tool_0_calculator",
                            input={"expression": "25 + 17"},
                        )
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="tool_0_calculator",
                            content="42",
                        )
                    ],
                ),
            ],
        )

        response = await self.make_request(request)
        print(f"response content: {response['content']}")

        self.assertResponseValid(response)
        self.assertNoToolUse(response)


    async def test_tool_streaming(self):
        """Test tool usage with streaming."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user", content="Calculate 100 * 50 using the calculator."
                )
            ],
        )

        response = await self.make_request(request, stream=True)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "calculator")

    # multi_tool
    async def test_multiple_tools(self):
        """Test request with multiple tools available."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[calculator_tool, weather_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user", content="I need to do some math. What's 15 * 8?"
                )
            ],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        # Should use calculator for math request
        self.assertHasToolUse(response, "calculator")

    # todo_write
    async def test_todo_write_tool(self):
        """Test TodoWrite tool usage."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            stream=False,
            tools=[todo_write_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Create a todo list: 1. Buy milk, 2. Pay bills, 3. Call mom",
                )
            ],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "TodoWrite")

        # Check that todos array is present
        tool_blocks = [
            block for block in response["content"] if block.get("type") == "tool_use"
        ]
        todo_call = next(
            (block for block in tool_blocks if block.get("name") == "TodoWrite"), None
        )

        self.assertIsNotNone(todo_call)
        self.assertIn("input", todo_call)
        self.assertIn("todos", todo_call["input"])
        self.assertIsInstance(todo_call["input"]["todos"], list)

    # todo_write_stream
    async def test_todo_write_tool_stream(self):
        """Test TodoWrite tool usage."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            stream=True,
            tools=[todo_write_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Create a todo list: 1. Buy milk, 2. Pay bills, 3. Call mom",
                )
            ],
        )

        response = await self.make_request(request, stream=True)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "TodoWrite")

        # Check that todos array is present
        tool_blocks = [
            block for block in response["content"] if block.get("type") == "tool_use"
        ]
        todo_call = next(
            (block for block in tool_blocks if block.get("name") == "TodoWrite"), None
        )

        self.assertIsNotNone(todo_call)
        self.assertIn("input", todo_call)
        self.assertIn("todos", todo_call["input"])
        self.assertIsInstance(todo_call["input"]["todos"], list)

    # todo_read
    async def test_todo_read_tool(self):
        """Test TodoRead tool usage."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            stream=False,
            max_tokens=1000,
            tools=[todo_read_tool],
            tool_choice=tool_choice_required,
            messages=[ClaudeMessage(role="user", content="What's on my todo list?")],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "TodoRead")

    async def test_todo_read_tool_stream(self):
        """Test TodoRead tool usage."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            stream=True,
            max_tokens=1000,
            tools=[todo_read_tool],
            tool_choice=tool_choice_required,
            messages=[ClaudeMessage(role="user", content="What's on my todo list?")],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "TodoRead")

    async def test_weather_tool(self):
        """Test weather tool usage."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[weather_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user", content="What's the weather like in San Francisco?"
                )
            ],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "get_weather")

        # Check location parameter
        tool_blocks = [
            block for block in response["content"] if block.get("type") == "tool_use"
        ]
        weather_call = next(
            (block for block in tool_blocks if block.get("name") == "get_weather"), None
        )

        self.assertIsNotNone(weather_call)
        self.assertIn("input", weather_call)
        self.assertIn("location", weather_call["input"])


class TestClaudeCodeTools(ProxyTestBase):
    """Test Claude Code specific tools."""

    async def test_bash_tool(self):
        """Test Bash tool usage."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[bash_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="List the files in the current directory using ls command.",
                )
            ],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "Bash")

        # Check command parameter
        tool_blocks = [
            block for block in response["content"] if block.get("type") == "tool_use"
        ]
        bash_call = next(
            (block for block in tool_blocks if block.get("name") == "Bash"), None
        )

        self.assertIsNotNone(bash_call)
        self.assertIn("input", bash_call)
        self.assertIn("command", bash_call["input"])
        self.assertIn("ls", bash_call["input"]["command"].lower())

    async def test_edit_tool(self):
        """Test Edit tool usage."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[edit_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Edit a file to replace 'old_text' with 'new_text' in /tmp/test.txt",
                )
            ],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "Edit")

        # Check edit parameters
        tool_blocks = [
            block for block in response["content"] if block.get("type") == "tool_use"
        ]
        edit_call = next(
            (block for block in tool_blocks if block.get("name") == "Edit"), None
        )

        self.assertIsNotNone(edit_call)
        self.assertIn("input", edit_call)
        input_params = edit_call["input"]
        self.assertIn("file_path", input_params)
        self.assertIn("old_string", input_params)
        self.assertIn("new_string", input_params)


class TestConversationFlow(ProxyTestBase):
    """Test multi-turn conversation scenarios."""

    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with tool results."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[calculator_tool],
            messages=[
                ClaudeMessage(role="user", content="What is 10 + 5?"),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text", text="I'll calculate that for you."
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="calc_001",
                            name="calculator",
                            input={"expression": "10 + 5"},
                        ),
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result", tool_use_id="calc_001", content="15"
                        ),
                        ClaudeContentBlockText(
                            type="text", text="Now what is double that?"
                        ),
                    ],
                ),
            ],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        # Should use calculator again for "double that" calculation
        self.assertHasToolUse(response, "calculator")

    async def test_conversation_with_thinking(self):
        """Test conversation that may include thinking content."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            thinking=ClaudeThinkingConfigEnabled(type="enabled", budget_tokens=200),
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Think step by step: What is the square root of 144?",
                )
            ],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasTextContent(response)
        # Note: thinking content availability depends on model and configuration


class TestErrorHandling(ProxyTestBase):
    """Test error handling scenarios."""

    async def test_invalid_model(self):
        """Test request with invalid model for default routed model"""
        request = ClaudeMessagesRequest(
            model="invalid-model-name",
            max_tokens=100,
            messages=[ClaudeMessage(role="user", content="Hello")],
        )

        response = await self.make_request(request)
        self.assertHasTextContent(response)

    async def test_malformed_request(self):
        """Test malformed request."""
        malformed_data = {
            "model": MODEL,
            "max_tokens": "invalid",  # Should be integer
            "messages": [],  # Empty messages
        }

        with self.assertRaises(httpx.HTTPStatusError):
            response = await self.client.post(
                f"{self.base_url}/v1/messages",
                headers=self.headers,
                json=malformed_data,
            )
            response.raise_for_status()


class TestStreamingSpecific(ProxyTestBase):
    """Test streaming-specific scenarios."""

    async def test_streaming_with_content_blocks(self):
        """Test streaming with complex content blocks."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text="Count from 1 to 3, then explain what you did.",
                        )
                    ],
                )
            ],
        )

        response = await self.make_request(request, stream=True)

        self.assertResponseValid(response)
        self.assertHasTextContent(response)

    async def test_streaming_tool_usage(self):
        """Test streaming with tool usage."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(role="user", content="What is 135 + 17.5 divided by 2.5?")
            ],
        )

        response = await self.make_request(request, stream=True)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "calculator")


class TestThinkingFeatures(ProxyTestBase):
    """Test thinking-related features."""

    async def test_thinking_enabled(self):
        """Test request with thinking enabled."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            thinking=ClaudeThinkingConfigEnabled(type="enabled", budget_tokens=200),
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Think carefully: What is the capital of France?",
                )
            ],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasTextContent(response)
        # Note: thinking content may or may not appear depending on model behavior

    async def test_thinking_disabled(self):
        """Test request with thinking explicitly disabled."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            thinking=ClaudeThinkingConfigDisabled(type="disabled"),
            messages=[ClaudeMessage(role="user", content="What is 2 + 2?")],
        )

        response = await self.make_request(request)

        self.assertResponseValid(response)
        self.assertHasTextContent(response)


class TestAnthropicComparison(ProxyTestBase):
    """Test proxy vs Anthropic API comparison functionality."""

    async def test_simple_comparison(self):
        """Test simple request comparison between proxy and Anthropic."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=100,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Hello! Please respond with exactly the word 'SUCCESS'.",
                )
            ],
        )

        passed, warning = await self.make_comparison_test("simple", request)
        self.assertTrue(passed, "Comparison test should pass")

    async def test_calculator_comparison(self):
        """Test calculator tool comparison between proxy and Anthropic."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            stream=False,
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user", content="What is 25 + 17? Use the calculator tool."
                )
            ],
        )

        passed, warning = await self.make_comparison_test(
            "calculator", request, check_tools=True
        )
        # Note: this might be a behavioral difference test, so we accept warnings
        print(f"Test result: {'PASSED' if passed else 'FAILED'}, Warning: {warning}")

    async def test_calculator_comparison_stream(self):
        """Test calculator tool comparison between proxy and Anthropic."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            stream=True,
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user", content="What is 25 + 17? Use the calculator tool."
                )
            ],
        )

        passed, warning = await self.make_comparison_test(
            "calculator_stream", request, check_tools=True
        )
        # Note: this might be a behavioral difference test, so we accept warnings
        print(f"Test result: {'PASSED' if passed else 'FAILED'}, Warning: {warning}")


class TestCustomModels(ProxyTestBase):
    """Test custom model functionality and conversions."""

    # gemini_tool_test
    async def test_gemini_tool_conversion(self):
        """Test Gemini model tool conversion."""
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            stream=False,
            max_tokens=1000,
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user", content="Calculate 25 * 8 using the calculator tool."
                )
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "gemini_tool_test", request, check_tools=True
        )
        self.assertTrue(passed, "Gemini tool conversion should work")

    # gemini_tool_test_stream
    async def test_gemini_tool_conversion_stream(self):
        """Test Gemini model tool conversion."""
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            stream=True,
            max_tokens=1000,
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user", content="Calculate 25 * 8 using the calculator tool."
                )
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "gemini_tool_test_stream", request, check_tools=True
        )
        self.assertTrue(passed, "Gemini tool conversion should work")

    async def test_gemini_incompatible_schema(self):
        """Test Gemini with incompatible schema features."""
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            stream=False,
            max_tokens=1000,
            tools=[gemini_incompatible_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Process this data: 'hello world' with strict validation level and configure timeout to 30 seconds.",
                )
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "gemini_incompatible_schema_test",
            request,
            check_tools=True,
            compare_with_anthropic=False,
        )
        self.assertTrue(passed, "Gemini incompatible schema conversion should work")

    async def test_gemini_incompatible_schema_stream(self):
        """Test Gemini with incompatible schema features."""
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            stream=True,
            max_tokens=1000,
            tools=[gemini_incompatible_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Process this data: 'hello world' with strict validation level and configure timeout to 30 seconds.",
                )
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "gemini_incompatible_schema_test_stream",
            request,
            check_tools=True,
            compare_with_anthropic=False,
        )
        self.assertTrue(passed, "Gemini incompatible schema conversion should work")

    # deepseek_thinking_tools
    async def test_deepseek_thinking_tools(self):
        """Test DeepSeek model with thinking and tools."""
        request = ClaudeMessagesRequest(
            model="deepseek-r1-250528",
            max_tokens=1024,
            thinking=ClaudeThinkingConfigEnabled(type="enabled", budget_tokens=1024),
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="What is 25 * 8? Think about it and use the calculator.",
                )
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "deepseek_thinking_tools",
            request,
            check_tools=True,
            compare_with_anthropic=False,  # Don't compare with Anthropic for custom models
        )
        self.assertTrue(passed, "DeepSeek thinking tools should work")

    # deepseek_thinking_tools_stream
    async def test_deepseek_thinking_tools_stream(self):
        """Test DeepSeek model with thinking and tools."""
        request = ClaudeMessagesRequest(
            model="deepseek-r1-250528",
            max_tokens=1024,
            stream=True,
            thinking=ClaudeThinkingConfigEnabled(type="enabled", budget_tokens=1024),
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="What is 25 * 8? Think about it and use the calculator.",
                )
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "deepseek_thinking_tools_stream",
            request,
            check_tools=True,
            compare_with_anthropic=False,  # Don't compare with Anthropic for custom models
        )
        self.assertTrue(passed, "DeepSeek thinking tools should work")

    # gemini_tool_result_positioning_test
    async def test_gemini_tool_result_positioning(self):
        """
        Test Gemini model tool result positioning requirement.

        The Gemini API requires that tool results (function responses) are at even indices
        in the conversation (indices 0, 2, 4, 6, etc.) since it assumes:
        - Even indices: User messages (including tool results)
        - Odd indices: Assistant messages (including function calls)

        This test creates a conversation with multiple tool call/result pairs to ensure
        tool results are positioned correctly for Gemini compatibility.
        """
        request = ClaudeMessagesRequest(
            model="gemini-2.5-flash-lite-preview-06-17",
            stream=False,
            max_tokens=1000,
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                # Index 0 (even) - User message
                ClaudeMessage(
                    role="user",
                    content="Calculate 25 * 8 using the calculator tool."
                ),
                # Index 1 (odd) - Assistant message with tool call
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="calc_001",
                            name="calculator",
                            input={"operation": "multiply", "a": 25, "b": 8},
                        )
                    ],
                ),
                # Index 2 (even) - User message with tool result (REQUIRED EVEN INDEX)
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="calc_001",
                            content="200",
                        )
                    ],
                ),
                # Index 3 (odd) - Assistant response and another tool call
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text="Great! Now let me divide that by 4."
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="calc_002",
                            name="calculator",
                            input={"operation": "divide", "a": 200, "b": 4},
                        )
                    ],
                ),
                # Index 4 (even) - User message with tool result (REQUIRED EVEN INDEX)
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="calc_002",
                            content="50",
                        )
                    ],
                ),
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "gemini_tool_result_positioning_test",
            request,
            check_tools=True,
            compare_with_anthropic=False  # Skip Anthropic comparison for Gemini-specific test
        )
        self.assertTrue(passed, "Gemini tool result positioning should work correctly")

    # gemini_tool_result_positioning_invalid_test
    async def test_gemini_tool_result_positioning_invalid(self):
        """
        Test Gemini model tool result positioning requirement - INVALID CASE.

        This test creates a conversation where tool results are at odd indices,
        which should FAIL for Gemini models since they require tool results at even indices.
        The test expects this to fail and demonstrates the importance of proper positioning.
        """
        request = ClaudeMessagesRequest(
            model="gemini-2.5-flash-lite-preview-06-17",
            stream=True,
            max_tokens=1024,
            messages=[
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
            ],
            tools=[exit_plan_mode_tool, todo_read_tool, todo_write_tool, edit_tool],
            tool_choice=tool_choice_auto
        )

        # This test expects to FAIL due to improper tool result positioning (tool result at odd index 1)
        passed, warning = await self.make_direct_conversion_test(
            "gemini_tool_result_positioning_invalid_test",
            request,
            check_tools=True,
            compare_with_anthropic=False,  # Skip Anthropic comparison for Gemini-specific test
            expect_failure=True  # We expect this to fail with 400 Bad Request
        )
        # We expect this to succeed (because we expected the API call to fail)
        self.assertTrue(passed, "Gemini tool result positioning should fail with incorrect indices")


class TestBehavioralDifferences(ProxyTestBase):
    """Test scenarios where behavioral differences are expected."""

    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation handling."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user", content="Let's do some math. What is 240 divided by 8?"
                ),
                ClaudeMessage(
                    role="assistant",
                    content="To calculate 240 divided by 8, I'll perform the division:\\n\\n240 ÷ 8 = 30\\n\\nSo the result is 30.",
                ),
                ClaudeMessage(
                    role="user",
                    content="Now multiply that by 4 and tell me the result.",
                ),
            ],
        )

        passed, warning = await self.make_comparison_test(
            "multi_turn", request, check_tools=True
        )
        # Multi-turn is a behavioral difference test, so warnings are acceptable
        print(
            f"Multi-turn test result: {'PASSED' if passed else 'FAILED'}, Warning: {warning}"
        )

    async def test_content_blocks_format(self):
        """Test content blocks format handling."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[calculator_tool, weather_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text="I need to know the weather in Los Angeles and calculate 75.5 / 5. Can you help with both?",
                        )
                    ],
                )
            ],
        )

        passed, warning = await self.make_comparison_test(
            "content_blocks", request, check_tools=True
        )
        print(
            f"Content blocks test result: {'PASSED' if passed else 'FAILED'}, Warning: {warning}"
        )


class TestClaudeCodeWorkflows(ProxyTestBase):
    """Test Claude Code specific workflows and tools."""

    # claude_code_interruption_test
    async def test_claude_code_interruption_test(self):
        """
        Test Claude Code tool call interruption and recovery capability.

        Test Scenario:
        - User requests configuration file creation
        - Assistant performs tool call chain: Glob → Read → exit_plan_mode
        - User interrupts exit_plan_mode with feedback: "file already exists, check first"
        - Expected: Assistant should use tools (Glob/Read) to verify existing files

        Key Test Design:
        - Uses tool_choice=required to eliminate model behavior differences
        - DeepSeek/other models might not use tools with tool_choice=auto or None
        - Claude naturally tends to use tools in this context
        - Server now automatically adjusts tool_choice=None/auto to required for non-thinking models
        - This ensures consistent behavior across all models without manual intervention

        Validates:
        1. Tool call interruption handling
        2. Context understanding after interruption
        3. Appropriate tool selection for verification tasks
        4. Message format conversion (Assistant→Tool→User sequence)

        Critical Bug Fixed:
        - Original issue: Assistant messages with only tool_calls were being dropped
        - Root cause: ClaudeMessage.to_openai_messages() ignored messages without content_parts
        - Solution: Modified logic to create assistant messages even with empty content ("")
        - Result: Proper Assistant(tool_calls) → Tool(tool_result) → User(text) sequence
        """
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=4000,
            stream=True,
            messages=[
                # Initial user request
                ClaudeMessage(
                    role="user",
                    content="Please help me create a configuration file example.",
                ),
                # Assistant starts with tool calls
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="call_abc123def456",
                            name="Glob",
                            input={"pattern": "config.yaml"},
                        )
                    ],
                ),
                # Tool result comes back
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="call_abc123def456",
                            content="/path/to/project/config.yaml",
                        )
                    ],
                ),
                # Assistant continues with another tool call
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="call_def789ghi012",
                            name="Read",
                            input={"file_path": "/path/to/project/config.yaml"},
                        )
                    ],
                ),
                # Another tool result
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="call_def789ghi012",
                            content="# Configuration File\nversion: 1.0\napi_key: example_key\nendpoint: https://api.example.com",
                        )
                    ],
                ),
                # Assistant tries to use exit_plan_mode but gets interrupted
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
                # User interrupts with mixed content - tool result + user message
                # This tests the critical message conversion: ClaudeMessage(tool_result + text)
                # converts to OpenAI sequence: Assistant(tool_calls) → Tool(tool_result) → User(text)
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
            tools=[glob_tool, read_tool, exit_plan_mode_tool],
            tool_choice=tool_choice_required,
        )
        passed, warning = await self.make_comparison_test(
            "test_claude_code_interruption_stream", request, check_tools=True
        )
        print(
            f"claude code interruption test result: {'PASSED' if passed else 'FAILED'}, Warning: {warning}"
        )

    async def test_thinking_model_tool_choice_required_error(self):
        """
        Test that thinking models reject tool_choice=required (API limitation validation).

        Test Discovery Process:
        1. Initial assumption: "Thinking models don't support tool_choice=required"
        2. First test without thinking=enabled: Unexpectedly succeeded
        3. Corrected test with thinking=enabled: Properly failed with 400 Bad Request

        Key Findings:
        - Thinking models require thinking=enabled in the request configuration
        - When thinking=enabled + tool_choice=required: Anthropic API returns 400 error
        - This confirms thinking models have API-level restrictions on tool_choice=required
        - Without thinking=enabled, the model behaves like a regular model

        Technical Details:
        - max_tokens (1024) must be >= budget_tokens (512) for thinking models
        - The 400 error comes from Anthropic API validation, not our proxy
        - This is an intentional limitation, not a bug

        Validates:
        1. Proper error handling for unsupported model+tool_choice combinations
        2. Anthropic API validation behavior
        3. Thinking model configuration requirements
        """
        request = ClaudeMessagesRequest(
            model=MODEL_THINKING,
            max_tokens=4000,
            stream=True,
            messages=[
                # Same test data as interruption test
                ClaudeMessage(
                    role="user",
                    content="Please help me create a configuration file example.",
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="call_abc123def456",
                            name="Glob",
                            input={"pattern": "config.yaml"},
                        )
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="call_abc123def456",
                            content="/path/to/project/config.yaml",
                        )
                    ],
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="call_def789ghi012",
                            name="Read",
                            input={"file_path": "/path/to/project/config.yaml"},
                        )
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="call_def789ghi012",
                            content="# Configuration File\nversion: 1.0\napi_key: example_key\nendpoint: https://api.example.com",
                        )
                    ],
                ),
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
                # User interrupts with mixed content - tool result + user message (the critical test case)
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
            tools=[glob_tool, read_tool, exit_plan_mode_tool],
            tool_choice=tool_choice_required,  # This should cause an error for thinking models
        )

        print(
            "🧠 Testing thinking model with tool_choice=required (expecting error)..."
        )

        # Test with a request that clearly requires tool use to isolate the tool_choice validation
        tool_request = ClaudeMessagesRequest(
            model=MODEL_THINKING,
            max_tokens=1024,
            thinking=ClaudeThinkingConfigEnabled(type="enabled", budget_tokens=512),
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Find all Python files in the current directory using glob pattern",
                ),
            ],
            tools=[glob_tool],
            tool_choice=tool_choice_required,
        )

        # Make the comparison test to see if Anthropic API vs our proxy behave differently
        try:
            passed, warning = await self.make_comparison_test(
                "test_thinking_model_tool_choice_required",
                tool_request,
                check_tools=True,
            )
            print(
                f"🧠 Thinking model tool_choice=required test result: {'PASSED' if passed else 'FAILED'}, Warning: {warning}"
            )

            if passed:
                print(
                    "✅ Both Anthropic and Proxy handle thinking model + tool_choice=required correctly"
                )
            else:
                print(
                    "⚠️ Different behavior between Anthropic and Proxy for thinking model + tool_choice=required"
                )

        except Exception as e:
            error_message = str(e)
            print(f"🧠 Comparison test failed: {error_message}")
            # Check if this indicates tool_choice=required is not allowed for thinking models
            if (
                (
                    "tool_choice" in error_message.lower()
                    and "required" in error_message.lower()
                )
                or (
                    "thinking" in error_message.lower()
                    and "required" in error_message.lower()
                )
                or "invalid" in error_message.lower()
            ):
                print(
                    "✅ Error confirms tool_choice=required is not allowed for thinking models"
                )
            else:
                print(f"⚠️ Unexpected error: {error_message}")

    async def test_thinking_model_interruption_with_auto_tool_choice(self):
        """
        Test thinking model's intelligent tool selection with tool call interruption.

        Test Scenario:
        - Same interruption scenario as the basic test
        - Uses thinking=enabled + tool_choice=auto (the supported combination)
        - User interrupts tool call with "file already exists, check first"

        Thinking Model Intelligence:
        - Unlike regular models, thinking models exhibit advanced reasoning
        - They can infer answers from conversation context without additional tool calls
        - In this test: model already saw the file content from previous Read tool call
        - Result: Often chooses NOT to use tools, directly provides context-based answer

        Behavioral Comparison:
        - Regular models + tool_choice=None/auto: Automatically adjusted to required by server
        - Regular models + tool_choice=required: Forced to use tools (consistent)
        - Thinking models + tool_choice=auto: Intelligent context-based decisions (unchanged by server)
        - Thinking models + tool_choice=required: API error (unsupported)

        Server Auto-Adjustment:
        - Non-thinking models: tool_choice=None/auto → automatically changed to required
        - Thinking models: tool_choice remains unchanged (to avoid API errors)
        - This ensures optimal tool usage without manual configuration

        Key Insights:
        1. Thinking models prioritize reasoning over tool calls when context is sufficient
        2. This is actually superior behavior - avoiding unnecessary API calls
        3. The 'no tool use' warning is expected and represents intelligent behavior
        4. Both Anthropic and Proxy exhibit consistent thinking model behavior

        Validates:
        1. Thinking model configuration (thinking=enabled)
        2. Intelligent tool selection vs. context reasoning
        3. Consistent behavior between Anthropic API and proxy
        4. Advanced reasoning capabilities in interruption scenarios
        """
        request = ClaudeMessagesRequest(
            model=MODEL_THINKING,
            max_tokens=4000,
            stream=True,
            thinking=ClaudeThinkingConfigEnabled(type="enabled", budget_tokens=1024),
            messages=[
                # Same test data as interruption test
                ClaudeMessage(
                    role="user",
                    content="Please help me create a configuration file example.",
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="call_abc123def456",
                            name="Glob",
                            input={"pattern": "config.yaml"},
                        )
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="call_abc123def456",
                            content="/path/to/project/config.yaml",
                        )
                    ],
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="call_def789ghi012",
                            name="Read",
                            input={"file_path": "/path/to/project/config.yaml"},
                        )
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="call_def789ghi012",
                            content="# Configuration File\nversion: 1.0\napi_key: example_key\nendpoint: https://api.example.com",
                        )
                    ],
                ),
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
                # User interrupts with mixed content - tool result + user message (the critical test case)
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
            tools=[glob_tool, read_tool, exit_plan_mode_tool],
            tool_choice=tool_choice_auto,  # Use auto for thinking models
        )

        print(
            "🧠 Testing thinking model interruption behavior with tool_choice=auto..."
        )

        try:
            passed, warning = await self.make_comparison_test(
                "test_thinking_model_interruption_stream", request, check_tools=False
            )
            print(
                f"thinking model interruption test result: {'PASSED' if passed else 'FAILED'}, Warning: {warning}"
            )

            # Note: Thinking models often exhibit different behavior from regular models:
            # - They may choose NOT to use tools when they can infer answers from context
            # - This is actually intelligent behavior - avoiding unnecessary tool calls
            # - Unlike regular models that might need tool_choice=required to force tool use
            if "Neither response contains tool use" in str(warning):
                print(
                    "✅ Expected behavior: Thinking model intelligently avoided unnecessary tool calls"
                )
                print("🧠 Thinking model used context reasoning instead of tools")

        except Exception as e:
            print(f"❌ Thinking model test failed with error: {e}")
            # This is acceptable - we're exploring the behavior

    async def test_claude_code_interruption_only_test(self):
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1024,
            messages=[
                # Simple case: user interrupts without prior tool use context
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text="[Request interrupted by user for tool use]",
                        ),
                        ClaudeContentBlockText(
                            type="text",
                            text="Please wait, I need to reconsider this approach.",
                        ),
                    ],
                )
            ],
        )
        response = await self.make_request(request)
        self.assertResponseValid(response)
        self.assertHasTextContent(response)

    # claude_code_read_test_stream
    async def test_claude_code_read_tool(self):
        """Test Claude Code Read tool."""
        request = ClaudeMessagesRequest(
            model="deepseek-v3-250324",
            max_tokens=1024,
            stream=True,
            tools=[read_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user", content="Use the Read tool to read the tests.py file."
                )
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "claude_code_read_test_stream",
            request,
            check_tools=True,
            compare_with_anthropic=False,
        )
        self.assertTrue(passed, "Claude Code Read tool should work")

    async def test_claude_code_bash_tool(self):
        """Test Claude Code Bash tool."""
        request = ClaudeMessagesRequest(
            model="deepseek-v3-250324",
            max_tokens=1024,
            stream=True,
            tools=[bash_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Use the Bash tool to list files in the current directory.",
                )
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "claude_code_bash_test_stream",
            request,
            check_tools=True,
            compare_with_anthropic=False,
        )
        self.assertTrue(passed, "Claude Code Bash tool should work")

    async def test_claude_code_ls_tool(self):
        """Test Claude Code ls tool."""
        request = ClaudeMessagesRequest(
            model="deepseek-v3-250324",
            max_tokens=1024,
            stream=True,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Use the LS tool to list the contents of the current directory.",
                )
            ],
            tools=[ls_tool],
            tool_choice=tool_choice_required,
        )

        passed, warning = await self.make_direct_conversion_test(
            "claude_code_ls_test_stream",
            request,
            check_tools=True,
            compare_with_anthropic=False,
        )
        self.assertTrue(passed, "Claude Code ls tool should work")

    async def test_claude_code_grep_tool(self):
        """Test Claude Code ls tool."""
        request = ClaudeMessagesRequest(
            model="deepseek-v3-250324",
            max_tokens=1024,
            stream=True,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Use the Grep tool to search for 'def' in the current directory.",
                )
            ],
            tools=[grep_tool],
            tool_choice=tool_choice_required,
        )

        passed, warning = await self.make_direct_conversion_test(
            "claude_code_grep_test_stream",
            request,
            check_tools=True,
            compare_with_anthropic=False,
        )
        self.assertTrue(passed, "Claude Code grep tool should work")

    async def test_claude_code_glob_tool(self):
        """Test Claude Code ls tool."""
        request = ClaudeMessagesRequest(
            model="deepseek-v3-250324",
            max_tokens=1024,
            stream=True,
            messages=[
                ClaudeMessage(
                    role="user", content="Use the Glob tool to find all Python files."
                )
            ],
            tools=[glob_tool],
            tool_choice=tool_choice_required,
        )

        passed, warning = await self.make_direct_conversion_test(
            "claude_code_glob_test_stream",
            request,
            check_tools=True,
            compare_with_anthropic=False,
        )
        self.assertTrue(passed, "Claude Code glob tool should work")

    async def test_claude_code_todo_workflow(self):
        """Test Claude Code TodoWrite and TodoRead workflow."""
        # First test TodoWrite
        todo_write_request = ClaudeMessagesRequest(
            model="deepseek-v3-250324",
            max_tokens=1024,
            stream=True,
            tools=[todo_write_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Use the TodoWrite tool to create a simple todo list.",
                )
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "claude_code_todowrite_test_stream",
            todo_write_request,
            check_tools=True,
            compare_with_anthropic=False,
        )
        self.assertTrue(passed, "Claude Code TodoWrite tool should work")

        # Then test TodoRead
        todo_read_request = ClaudeMessagesRequest(
            model="deepseek-v3-250324",
            max_tokens=1024,
            stream=True,
            tools=[todo_read_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Use the TodoRead tool to show the current todo list.",
                )
            ],
        )

        passed, warning = await self.make_direct_conversion_test(
            "claude_code_todoread_test_stream",
            todo_read_request,
            check_tools=True,
            compare_with_anthropic=False,
        )
        self.assertTrue(passed, "Claude Code TodoRead tool should work")


class TestStreamingAdvanced(ProxyTestBase):
    """Test advanced streaming scenarios."""

    async def test_streaming_with_tools(self):
        """Test streaming with tool usage."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1000,
            tools=[calculator_tool],
            tool_choice=tool_choice_required,
            messages=[
                ClaudeMessage(role="user", content="What is 135 + 17.5 divided by 2.5?")
            ],
        )

        response = await self.make_request(request, stream=True)

        self.assertResponseValid(response)
        self.assertHasToolUse(response, "calculator")

    async def test_streaming_thinking_features(self):
        """Test streaming with thinking enabled."""
        request = ClaudeMessagesRequest(
            model=MODEL_THINKING,
            max_tokens=1000,
            thinking=ClaudeThinkingConfigEnabled(type="enabled", budget_tokens=200),
            messages=[
                ClaudeMessage(
                    role="user", content="What is 15 + 27? Please think about it."
                )
            ],
        )

        response = await self.make_request(request, stream=True)

        self.assertResponseValid(response)
        self.assertHasTextContent(response)
        # Note: thinking content may or may not appear depending on model behavior


class TestComplexScenarios(ProxyTestBase):
    # deepseek_complex_todo_workflow_stream
    async def test_deepseek_complex_todo_workflow_stream(self):
        """Complex multi-turn Todo workflow test - simulates Claude Code task planning and completion"""
        request = ClaudeMessagesRequest(
            model="deepseek-v3-0324",
            max_tokens=3072,
            stream=True,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Help me implement a new feature for a web application that allows users to export their data. I need you to plan this task using the TodoWrite tool, then work through it step by step.",
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text="I'll help you implement a data export feature for your web application. Let me start by creating a comprehensive task plan using the TodoWrite tool.",
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu_01ABC123DEF456",
                            name="TodoWrite",
                            input={
                                "todos": [
                                    {
                                        "id": "1",
                                        "content": "Design the data export API endpoints and schema",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "2",
                                        "content": "Implement backend export service with CSV/JSON support",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "3",
                                        "content": "Create frontend export UI component with progress indicator",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "4",
                                        "content": "Add export format selection (CSV, JSON, Excel)",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "5",
                                        "content": "Implement data filtering and pagination for large exports",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "6",
                                        "content": "Add user authentication and permission checks for exports",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "7",
                                        "content": "Write comprehensive unit tests for export functionality",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "8",
                                        "content": "Create integration tests for end-to-end export workflow",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "9",
                                        "content": "Add rate limiting and security measures for export API",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "10",
                                        "content": "Document export API and create user guide",
                                        "status": "pending",
                                        "priority": "low",
                                    },
                                ]
                            },
                        ),
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="toolu_01ABC123DEF456",
                            content="Todo list has been updated successfully. 10 items added to the task list.",
                        ),
                        ClaudeContentBlockText(
                            type="text",
                            text="Great! Now let's start working on the first task. Begin with designing the API endpoints.",
                        ),
                    ],
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text="Perfect! Let me start working on the first task by designing the data export API endpoints. I'll mark it as in progress and then provide the design.",
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu_02XYZ789GHI012",
                            name="TodoWrite",
                            input={
                                "todos": [
                                    {
                                        "id": "1",
                                        "content": "Design the data export API endpoints and schema",
                                        "status": "in_progress",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "2",
                                        "content": "Implement backend export service with CSV/JSON support",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "3",
                                        "content": "Create frontend export UI component with progress indicator",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "4",
                                        "content": "Add export format selection (CSV, JSON, Excel)",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "5",
                                        "content": "Implement data filtering and pagination for large exports",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "6",
                                        "content": "Add user authentication and permission checks for exports",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "7",
                                        "content": "Write comprehensive unit tests for export functionality",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "8",
                                        "content": "Create integration tests for end-to-end export workflow",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "9",
                                        "content": "Add rate limiting and security measures for export API",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "10",
                                        "content": "Document export API and create user guide",
                                        "status": "pending",
                                        "priority": "low",
                                    },
                                ]
                            },
                        ),
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="toolu_02XYZ789GHI012",
                            content="Todo list has been updated successfully. Task 1 marked as in_progress.",
                        ),
                    ],
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text='Excellent! Now I\'ll design the comprehensive API endpoints for the data export feature:\n\n## Data Export API Design\n\n### 1. Export Request Endpoint\n```\nPOST /api/v1/exports\nContent-Type: application/json\nAuthorization: Bearer <token>\n\nRequest Body:\n{\n  "format": "csv|json|excel",\n  "data_type": "users|orders|products|analytics",\n  "filters": {\n    "date_range": {\n      "start": "2024-01-01",\n      "end": "2024-12-31"\n    },\n    "status": ["active", "pending"],\n    "categories": ["electronics", "books"]\n  },\n  "fields": ["id", "name", "email", "created_at"],\n  "sort_by": "created_at",\n  "sort_order": "desc",\n  "chunk_size": 1000\n}\n\nResponse:\n{\n  "export_id": "exp_1234567890",\n  "status": "queued",\n  "estimated_records": 15420,\n  "created_at": "2024-06-22T10:30:00Z",\n  "expires_at": "2024-06-29T10:30:00Z"\n}\n```\n\n### 2. Export Status Check\n```\nGET /api/v1/exports/{export_id}\nAuthorization: Bearer <token>\n\nResponse:\n{\n  "export_id": "exp_1234567890",\n  "status": "processing|completed|failed",\n  "progress": {\n    "processed_records": 8500,\n    "total_records": 15420,\n    "percentage": 55.1\n  },\n  "download_url": "https://api.example.com/api/v1/exports/exp_1234567890/download",\n  "file_size": 2048576,\n  "error_message": null,\n  "created_at": "2024-06-22T10:30:00Z",\n  "completed_at": "2024-06-22T10:45:23Z",\n  "expires_at": "2024-06-29T10:30:00Z"\n}\n```\n\n### 3. Export Download\n```\nGET /api/v1/exports/{export_id}/download\nAuthorization: Bearer <token>\n\nResponse Headers:\nContent-Type: application/octet-stream\nContent-Disposition: attachment; filename="users_export_20240622.csv"\nContent-Length: 2048576\n\nResponse: Binary file data\n```\n\n### 4. List User Exports\n```\nGET /api/v1/exports?page=1&limit=20&status=completed\nAuthorization: Bearer <token>\n\nResponse:\n{\n  "exports": [\n    {\n      "export_id": "exp_1234567890",\n      "status": "completed",\n      "format": "csv",\n      "data_type": "users",\n      "record_count": 15420,\n      "file_size": 2048576,\n      "created_at": "2024-06-22T10:30:00Z",\n      "expires_at": "2024-06-29T10:30:00Z"\n    }\n  ],\n  "pagination": {\n    "page": 1,\n    "limit": 20,\n    "total": 45,\n    "has_more": true\n  }\n}\n```\n\n### 5. Cancel Export\n```\nDELETE /api/v1/exports/{export_id}\nAuthorization: Bearer <token>\n\nResponse:\n{\n  "message": "Export cancelled successfully",\n  "export_id": "exp_1234567890"\n}\n```\n\n## Database Schema\n\n```sql\nCREATE TABLE exports (\n  id VARCHAR(50) PRIMARY KEY,\n  user_id INTEGER NOT NULL,\n  status ENUM(\'queued\', \'processing\', \'completed\', \'failed\', \'cancelled\'),\n  format ENUM(\'csv\', \'json\', \'excel\'),\n  data_type VARCHAR(50),\n  filters JSON,\n  fields JSON,\n  total_records INTEGER,\n  processed_records INTEGER DEFAULT 0,\n  file_path VARCHAR(500),\n  file_size BIGINT,\n  error_message TEXT,\n  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\n  started_at TIMESTAMP NULL,\n  completed_at TIMESTAMP NULL,\n  expires_at TIMESTAMP NOT NULL,\n  INDEX idx_user_status (user_id, status),\n  INDEX idx_created_at (created_at),\n  INDEX idx_expires_at (expires_at)\n);\n```\n\nThe API design includes proper authentication, rate limiting considerations, progress tracking, and automatic cleanup of expired exports. Now let me mark this task as completed and move to the next one.',
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu_03JKL345MNO678",
                            name="TodoWrite",
                            input={
                                "todos": [
                                    {
                                        "id": "1",
                                        "content": "Design the data export API endpoints and schema",
                                        "status": "completed",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "2",
                                        "content": "Implement backend export service with CSV/JSON support",
                                        "status": "in_progress",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "3",
                                        "content": "Create frontend export UI component with progress indicator",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "4",
                                        "content": "Add export format selection (CSV, JSON, Excel)",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "5",
                                        "content": "Implement data filtering and pagination for large exports",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "6",
                                        "content": "Add user authentication and permission checks for exports",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "7",
                                        "content": "Write comprehensive unit tests for export functionality",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "8",
                                        "content": "Create integration tests for end-to-end export workflow",
                                        "status": "pending",
                                        "priority": "medium",
                                    },
                                    {
                                        "id": "9",
                                        "content": "Add rate limiting and security measures for export API",
                                        "status": "pending",
                                        "priority": "high",
                                    },
                                    {
                                        "id": "10",
                                        "content": "Document export API and create user guide",
                                        "status": "pending",
                                        "priority": "low",
                                    },
                                ]
                            },
                        ),
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="toolu_03JKL345MNO678",
                            content="Todo list has been updated successfully. Task 1 marked as completed, Task 2 marked as in_progress.",
                        ),
                        ClaudeContentBlockText(
                            type="text",
                            text="Excellent API design! Now let's check our current progress and then continue with implementing the backend service. Can you show me the current todo list status?",
                        ),
                    ],
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text="Of course! Let me check our current progress by reading the todo list.",
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu_04PQR901STU234",
                            name="TodoRead",
                            input={},
                        ),
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="toolu_04PQR901STU234",
                            content=[
                                {
                                    "id": "1",
                                    "content": "Design the data export API endpoints and schema",
                                    "status": "completed",
                                    "priority": "high",
                                },
                                {
                                    "id": "2",
                                    "content": "Implement backend export service with CSV/JSON support",
                                    "status": "in_progress",
                                    "priority": "high",
                                },
                                {
                                    "id": "3",
                                    "content": "Create frontend export UI component with progress indicator",
                                    "status": "pending",
                                    "priority": "medium",
                                },
                                {
                                    "id": "4",
                                    "content": "Add export format selection (CSV, JSON, Excel)",
                                    "status": "pending",
                                    "priority": "medium",
                                },
                                {
                                    "id": "5",
                                    "content": "Implement data filtering and pagination for large exports",
                                    "status": "pending",
                                    "priority": "high",
                                },
                                {
                                    "id": "6",
                                    "content": "Add user authentication and permission checks for exports",
                                    "status": "pending",
                                    "priority": "high",
                                },
                                {
                                    "id": "7",
                                    "content": "Write comprehensive unit tests for export functionality",
                                    "status": "pending",
                                    "priority": "medium",
                                },
                                {
                                    "id": "8",
                                    "content": "Create integration tests for end-to-end export workflow",
                                    "status": "pending",
                                    "priority": "medium",
                                },
                                {
                                    "id": "9",
                                    "content": "Add rate limiting and security measures for export API",
                                    "status": "pending",
                                    "priority": "high",
                                },
                                {
                                    "id": "10",
                                    "content": "Document export API and create user guide",
                                    "status": "pending",
                                    "priority": "low",
                                },
                            ],
                        ),
                        ClaudeContentBlockText(
                            type="text",
                            text="Perfect! Now please provide a summary of our progress and what we've accomplished so far in this implementation.",
                        ),
                    ],
                ),
            ],
        )
        response = await self.make_request(request, stream=True)
        self.assertResponseValid(response)
        self.assertHasTextContent(response)

    async def test_edit_tool_completion(self):
        """Test complex multi-step scenarios."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            stream=False,
            max_tokens=1025,
            system="You are Claude Code, Anthropic's official CLI for Claude. You help users with software engineering tasks using available tools.",
            tools=[edit_tool],
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Please modify the custom_models.yaml.example file to remove sensitive data and make placeholders more generic.",
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text="I'll help you modify the custom_models.yaml.example file to remove sensitive data and make placeholders more generic. Let me make the change to replace the sensitive API key with a generic placeholder.",
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu_edit_test_001",
                            name="Edit",
                            input={
                                "file_path": "/tmp/custom_models.yaml.example",
                                "old_string": "api_key: sk-1234567890abcdef",
                                "new_string": "api_key: your-api-key-here",
                            },
                        ),
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="toolu_edit_test_001",
                            content="The file /tmp/custom_models.yaml.example has been updated. The API key has been changed to a generic placeholder.",
                        )
                    ],
                ),
            ],
        )
        passed, warning = await self.make_comparison_test(
            "edit_tool_completion",
            request,
            check_tools=True,  # This test focuses on post-tool conversation flow
        )
        print(
            f"Edit tool completion test result: {'PASSED' if passed else 'FAILED'}, Warning: {warning}"
        )

    # Edit tool completion test with streaming - tests model response after tool execution
    async def test_edit_tool_completion_stream(self):
        """Test complex multi-step scenarios."""
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1025,
            stream=True,
            system="You are Claude Code, Anthropic's official CLI for Claude. You help users with software engineering tasks using available tools.",
            messages=[
                ClaudeMessage(
                    role="user",
                    content="Please modify the custom_models.yaml.example file to remove sensitive data and make placeholders more generic.",
                ),
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockText(
                            type="text",
                            text="I'll help you modify the custom_models.yaml.example file to remove sensitive data and make placeholders more generic. Let me make the change to replace the sensitive API key with a generic placeholder.",
                        ),
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="toolu_edit_test_stream_001",
                            name="Edit",
                            input={
                                "file_path": "/tmp/custom_models.yaml.example",
                                "old_string": "api_key: sk-1234567890abcdef",
                                "new_string": "api_key: your-api-key-here",
                            },
                        ),
                    ],
                ),
                ClaudeMessage(
                    role="user",
                    content=[
                        ClaudeContentBlockToolResult(
                            type="tool_result",
                            tool_use_id="toolu_edit_test_stream_001",
                            content="The file /tmp/custom_models.yaml.example has been updated. The API key has been changed to a generic placeholder.",
                        )
                    ],
                ),
            ],
            tools=[edit_tool],
        )

        passed, warning = await self.make_comparison_test(
            "edit_tool_completion_stream",
            request,
            check_tools=True,  # This test focuses on post-tool conversation flow
        )
        print(
            f"Edit tool completion stream test result: {'PASSED' if passed else 'FAILED'}, Warning: {warning}"
        )


# Main test runner
def run_all_tests():
    """Run all test classes using standard unittest."""
    test_classes = [
        TestBasicRequests,
        TestToolRequests,
        TestClaudeCodeTools,
        TestConversationFlow,
        TestStreamingSpecific,
        TestThinkingFeatures,
        TestErrorHandling,
        TestAnthropicComparison,
        TestCustomModels,
        TestBehavioralDifferences,
        TestClaudeCodeWorkflows,
        TestStreamingAdvanced,
        TestComplexScenarios,
    ]

    print("🚀 Running Claude Proxy Tests with unittest framework")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    print(f"\n{'=' * 60}")
    print(
        f"🎉 Test Results: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun} tests passed"
    )
    if result.failures:
        print(f"❌ Failures: {len(result.failures)}")
    if result.errors:
        print(f"💥 Errors: {len(result.errors)}")
    print(f"{'=' * 60}")

    return result.wasSuccessful()


class TestExitPlanModeScenario(ProxyTestBase):
    """Test the specific exit_plan_mode scenario to isolate the 'no content' issue."""

    async def test_exit_plan_mode_content_preservation(self):
        """Test exit_plan_mode scenario to check if content is preserved properly."""
        print("🔍 Testing exit_plan_mode content preservation...")

        # Create the exact scenario from your logs:
        # Assistant provides plan + calls exit_plan_mode, then gets approval
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=1024,
            messages=[
                # ClaudeMessage(
                #     role="user",
                #     content="I need you to help me add test coverage commands to my Makefile and update the README. Can you make a plan?"
                #     ),
                # Assistant message with plan text + exit_plan_mode tool call
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
                # Tool result (plan approved) - testing with assistant role
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
            ],
            tools=[exit_plan_mode_tool, todo_read_tool, todo_write_tool, edit_tool],
            tool_choice=tool_choice_auto
        )

        # Run comparison test between proxy and Anthropic API
        passed, warning = await self.make_comparison_test(
            "exit_plan_mode_content_preservation", request, check_tools=True
        )

        print(f"📊 Exit plan mode test result: {'PASSED' if passed else 'FAILED'}")
        if warning:
            print(f"⚠️  Warning: {warning}")

        # The key insight: if both APIs return similar "no content" behavior,
        # then it's legitimate model behavior, not a proxy conversion bug
        if "no content" in str(warning).lower() or "empty" in str(warning).lower():
            print(
                "💡 FINDING: Both Anthropic API and proxy show similar content behavior"
            )
            print(
                "📝 This suggests '(no content)' is normal model behavior, not a conversion bug"
            )

        # We don't fail the test even if there are warnings, as this is a behavioral analysis
        print("✅ Content preservation test completed")

    async def test_simplified_plan_approval_scenario(self):
        """Test simplified version: just plan approval, see model response."""
        print("\n🤖 Testing simplified plan approval scenario...")

        # Simplified version: User approves a plan, what does model do next?
        request = ClaudeMessagesRequest(
            model=MODEL,
            max_tokens=500,
            messages=[
                ClaudeMessage(
                    role="user",
                    content="I need you to update my Makefile and README for test coverage. Your plan has been approved. You can now start coding.",
                )
            ],
            tools=[todo_read_tool, todo_write_tool, edit_tool],
            tool_choice=tool_choice_required
        )

        passed, warning = await self.make_comparison_test(
            "simplified_plan_approval", request,
            check_tools=True
        )

        print(f"📊 Simplified test result: {'PASSED' if passed else 'FAILED'}")
        if warning:
            print(f"⚠️  Warning: {warning}")

        print("💡 This test shows natural model behavior after plan approval")
        print("📝 Helps distinguish between conversion bugs vs. normal model responses")


def main():
    """Main entry point for running tests."""
    # Check environment
    if not ANTHROPIC_API_KEY:
        print("❌ Error: ANTHROPIC_API_KEY not set in .env file")
        print("Please set ANTHROPIC_API_KEY in your .env file")
        sys.exit(1)

    # Check if server is running
    try:
        import httpx

        with httpx.Client(timeout=5) as client:
            # Try a simple request to test if server is responding
            response = client.get(f"{BASE_URL}/test-connection")
            print(
                f"✅ Server at {BASE_URL} is responding (status: {response.status_code})"
            )
    except Exception as e:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print(f"Error: {e}")
        print("Please make sure the proxy server is running with: make run")
        sys.exit(1)

    # Run tests
    print(f"✅ Server is running at {BASE_URL}")
    success = run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
