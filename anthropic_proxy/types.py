"""
Pydantic models and type definitions for Claude proxy API.
This module contains all data models, constants, and type definitions.
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from typing import Any, Literal

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)


class ModelDefaults:
    """Default values and limits for model configurations"""

    # Token limits (based on Claude API specifications)
    DEFAULT_MAX_TOKENS = 8192  # Default maximum tokens to generate
    DEFAULT_MAX_INPUT_TOKENS = 128000  # Default maximum input tokens
    MAX_TOKENS_LIMIT = 16384  # Maximum tokens limit for responses
    LONG_CONTEXT_THRESHOLD = 128000  # Threshold for long-context models

    # Thinking configuration
    THINKING_SIGNATURE_LENGTH = 16  # Length of thinking block signature hash

    # Default server settings for proxy
    DEFAULT_HOST = "0.0.0.0"  # Default server host IPv4 (127.0.0.1)
    DEFAULT_PORT = 8082  # Default server port
    DEFAULT_LOG_LEVEL = "WARNING"  # Default logging level
    DEFAULT_MAX_RETRIES = 2  # Default maximum retry attempts


class Constants:
    """Constants for better maintainability"""

    # Message roles (Claude API)
    ROLE_USER = "user"  # User message role
    ROLE_ASSISTANT = "assistant"  # Assistant message role
    ROLE_SYSTEM = "system"  # System prompt role
    ROLE_TOOL = "tool"  # Tool result role (OpenAI compatibility)

    # Content block types (Claude API)
    CONTENT_TEXT = "text"  # Text content block
    CONTENT_IMAGE = "image"  # Image content block
    CONTENT_TOOL_USE = "tool_use"  # Tool invocation block
    CONTENT_TOOL_RESULT = "tool_result"  # Tool result block

    # Tool types (OpenAI compatibility)
    TOOL_FUNCTION = "function"  # Function tool type

    # Stop reasons (Claude API response)
    STOP_END_TURN = "end_turn"  # Natural stopping point
    STOP_MAX_TOKENS = "max_tokens"  # Reached token limit
    STOP_TOOL_USE = "tool_use"  # Tool invocation triggered
    STOP_ERROR = "error"  # Error occurred

    # Streaming event types (Claude API)
    EVENT_MESSAGE_START = "message_start"  # Start of message
    EVENT_MESSAGE_STOP = "message_stop"  # End of message
    EVENT_MESSAGE_DELTA = "message_delta"  # Message content delta
    EVENT_CONTENT_BLOCK_START = "content_block_start"  # Start of content block
    EVENT_CONTENT_BLOCK_STOP = "content_block_stop"  # End of content block
    EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"  # Content block delta
    EVENT_PING = "ping"  # Keepalive ping

    # Delta types for streaming
    DELTA_TEXT = "text_delta"  # Text content delta
    DELTA_INPUT_JSON = "input_json_delta"  # Tool input JSON delta


def generate_unique_id(prefix: str) -> str:
    """
    Generate a unique ID with specified prefix, timestamp and random suffix.
    Format: <prefix>_<timestamp_ms>_<random_hex>
    This ensures uniqueness across all instances.
    """
    import time

    timestamp_ms = int(time.time() * 1000)
    random_suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp_ms}_{random_suffix}"


# === Tool Choice Classes ===
class ClaudeToolChoiceAuto(BaseModel):
    """
    Auto tool choice - the model will automatically decide whether to use tools.

    The model can choose to use any available tool or no tools at all.
    This is the default behavior when tools are provided.
    """

    type: Literal["auto"] = "auto"  # Tool choice type identifier
    disable_parallel_tool_use: bool | None = (
        None  # Whether to disable parallel tool use (default: False)
    )

    def to_openai(self) -> ChatCompletionToolChoiceOptionParam:
        return "auto"


class ClaudeToolChoiceAny(BaseModel):
    """
    Any tool choice - the model will use any available tools.

    The model is required to use at least one tool from the available tools.
    If disable_parallel_tool_use is true, the model will use exactly one tool.
    """

    type: Literal["any"] = "any"  # Tool choice type identifier
    disable_parallel_tool_use: bool | None = (
        None  # Whether to disable parallel tool use (default: False)
    )

    def to_openai(self) -> ChatCompletionToolChoiceOptionParam:
        return "required"


class ClaudeToolChoiceTool(BaseModel):
    """
    Specific tool choice - the model will use the specified tool.

    Forces the model to use a specific tool by name.
    If disable_parallel_tool_use is true, the model will use exactly one tool.
    """

    type: Literal["tool"] = "tool"  # Tool choice type identifier
    name: str  # The name of the tool to use
    disable_parallel_tool_use: bool | None = (
        None  # Whether to disable parallel tool use (default: False)
    )

    def to_openai(self) -> ChatCompletionNamedToolChoiceParam:
        return {"type": "function", "function": {"name": self.name}}


class ClaudeToolChoiceNone(BaseModel):
    """
    None tool choice - the model will not be allowed to use tools.

    Prevents the model from using any tools, even if tools are provided.
    The model will only generate text responses.
    """

    type: Literal["none"] = "none"  # Tool choice type identifier

    def to_openai(self) -> ChatCompletionToolChoiceOptionParam:
        return "none"


# Union type for all tool choice options
# How the model should use the provided tools
ClaudeToolChoice = (
    ClaudeToolChoiceAuto  # Let model decide whether to use tools
    | ClaudeToolChoiceAny  # Require model to use at least one tool
    | ClaudeToolChoiceTool  # Force model to use a specific tool
    | ClaudeToolChoiceNone  # Prevent model from using any tools
)


# === Content Block Classes ===
class ClaudeContentBlockText(BaseModel):
    """
    Text content block.

    Regular text content for messages. This is the most common content type
    and can be used for both user inputs and assistant responses.

    Example: {"type": "text", "text": "Hello, Claude!"}
    """

    model_config = ConfigDict(
        validate_assignment=False, str_strip_whitespace=False, extra="ignore"
    )

    type: Literal["text"]  # Content block type identifier
    text: str  # The text content (minimum length: 1)

    def to_openai(self) -> ChatCompletionContentPartTextParam:
        """Convert Claude text block to OpenAI text format."""
        return {"type": "text", "text": self.text}


class ClaudeContentBlockImageBase64Source(BaseModel):
    """Base64-encoded image source for image content blocks."""

    type: Literal["base64"]  # Source type identifier
    media_type: Literal[
        "image/jpeg", "image/png", "image/gif", "image/webp"
    ]  # Supported image formats
    data: str  # Base64-encoded image data


class ClaudeContentBlockImageURLSource(BaseModel):
    """URL-based image source for image content blocks."""

    type: Literal["url"]  # Source type identifier
    url: str  # URL to the image resource


class ClaudeContentBlockImage(BaseModel):
    """
    Image content block.

    Image content specified directly as base64 data or as a reference via a URL.
    Starting with Claude 3 models, you can send image content blocks.
    Supports JPEG, PNG, GIF, and WebP formats.

    Example: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}
    """

    model_config = ConfigDict(
        validate_assignment=False, str_strip_whitespace=False, extra="ignore"
    )

    type: Literal["image"]  # Content block type identifier
    source: (  # Image source specification
        ClaudeContentBlockImageBase64Source  # Base64-encoded image data
        | ClaudeContentBlockImageURLSource  # URL reference to image
        | dict[str, Any]  # Flexible dict for compatibility
    )

    def to_openai(self) -> ChatCompletionContentPartImageParam | None:
        """Convert Claude image block to OpenAI image_url format."""
        if isinstance(self.source, ClaudeContentBlockImageBase64Source):
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{self.source.media_type};base64,{self.source.data}"
                },
            }
        elif isinstance(self.source, ClaudeContentBlockImageURLSource):
            return {
                "type": "image_url",
                "image_url": {"url": f"{self.source.url}"},
            }
        return None


class ClaudeContentBlockToolUse(BaseModel):
    """
    Tool use content block.

    A block indicating a tool use by the model. When the model invokes tools,
    it returns tool_use content blocks that represent the model's use of those tools.
    You can then run those tools using the tool input generated by the model
    and return results back to the model using tool_result content blocks.
    """

    model_config = ConfigDict(
        validate_assignment=False, str_strip_whitespace=False, extra="ignore"
    )

    type: Literal["tool_use"]  # Content block type identifier
    id: str  # Unique identifier for this tool use
    name: str  # Name of the tool being invoked
    input: dict[str, Any]  # Parameters/arguments for the tool call

    def to_openai(self) -> ChatCompletionMessageToolCallParam:
        """Convert Claude tool_use to OpenAI tool_call format."""
        try:
            arguments_str = json.dumps(
                self.input, ensure_ascii=False, separators=(",", ":")
            )
        except (TypeError, ValueError):
            arguments_str = "{}"

        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": arguments_str},
        }


class ClaudeContentBlockToolResult(BaseModel):
    """
    Tool result content block.

    A block specifying the results of a tool use by the model.
    You provide this in a user message after the model has requested
    tool use, containing the output/result from running the requested tool.
    """

    model_config = ConfigDict(
        validate_assignment=False, str_strip_whitespace=False, extra="ignore"
    )

    type: Literal["tool_result"]  # Content block type identifier
    tool_use_id: str  # ID of the tool_use this result corresponds to
    content: (
        str | list[dict[str, Any]]
    )  # Tool execution result (string or array of content blocks)

    def process_content(self) -> str | list:
        """Process Claude tool_result content into a string format."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            content_parts = []
            for item in self.content:
                # Handle content blocks (text and image blocks according to Claude API spec)
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        # Standard text block: {"type": "text", "text": "content"}
                        content_parts.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image":
                        # Image block - pass through as is
                        content_parts.append(item)
                    elif "text" in item:
                        # Text block without explicit type
                        content_parts.append({"type": "text", "text": item["text"]})
                    else:
                        # Fallback: convert unknown dict structure to text
                        content_parts.append({"type": "text", "text": str(item)})
                else:
                    # Non-dict items should not exist in tool_result content per spec
                    # but handle gracefully by converting to text
                    content_parts.append({"type": "text", "text": str(item)})
            return content_parts
        else:
            # This case should not occur per API spec, but handle gracefully
            return str(self.content)

    def to_openai_message(self, tool_name: str = None) -> ChatCompletionToolMessageParam:
        """Convert Claude tool_result to OpenAI tool role message format."""
        result = {
            "role": "tool",
            "tool_call_id": self.tool_use_id,
            "content": self.process_content(),
        }

        # Add name field if provided (required by some providers like Groq)
        if tool_name:
            result["name"] = tool_name

        return result


class ClaudeContentBlockThinking(BaseModel):
    """
    Thinking content block.

    A block specifying internal thinking by the model. When extended thinking
    is enabled, responses include thinking content blocks showing Claude's
    thinking process before the final answer. Requires a minimum budget of 1,024 tokens.
    """

    type: Literal["thinking"]  # Content block type identifier
    thinking: str  # The internal reasoning/thinking text
    signature: str | None = None  # Optional signature for the thinking block

    def to_openai(self) -> ChatCompletionContentPartTextParam:
        """Thinking blocks should be transformed to assistant text message"""
        return {"type": "text", "text": self.thinking}


class ClaudeSystemContent(BaseModel):
    """System prompt content block for structured system prompts."""

    type: Literal["text"]  # Content block type identifier
    text: str  # The system prompt text content


class ClaudeTool(BaseModel):
    """
    Tool definition for Claude API.

    Definitions of tools that the model may use. Each tool definition includes
    a name, description, and JSON schema for the tool input shape.
    Tools can be used for workflows that include running client-side tools
    and functions, or whenever you want the model to produce a particular
    JSON structure of output.

    See: https://docs.anthropic.com/en/docs/tool-use
    """

    name: str  # Name of the tool (max length: 128, pattern: ^[a-zA-Z0-9_-]{1,128}$)
    description: str | None = (
        None  # Description of what this tool does (strongly recommended)
    )
    input_schema: dict[
        str, Any
    ]  # JSON schema for tool input shape (type must be "object")


class ClaudeThinkingConfigEnabled(BaseModel):
    """
    Enabled thinking configuration.

    Configuration for enabling Claude's extended thinking. When enabled,
    responses include thinking content blocks showing Claude's thinking process
    before the final answer. Requires a minimum budget of 1,024 tokens and
    counts towards your max_tokens limit.
    """

    type: Literal["enabled"] = "enabled"  # Configuration type identifier
    budget_tokens: int | None = (
        None  # Token budget for thinking (minimum: 1024, must be < max_tokens)
    )


class ClaudeThinkingConfigDisabled(BaseModel):
    """
    Disabled thinking configuration.

    Configuration for disabling Claude's extended thinking.
    The model will respond directly without showing internal reasoning.
    """

    type: Literal["disabled"] = "disabled"  # Configuration type identifier


def parse_thinking_config(v):
    """Parse thinking configuration from dict to typed config objects.

    Used by field validators in ClaudeTokenCountRequest and ClaudeMessagesRequest.
    """
    if not isinstance(v, dict):
        return v

    # Check for enabled: true/false format
    if v.get("enabled") is True:
        return ClaudeThinkingConfigEnabled(
            type="enabled", budget_tokens=v.get("budget_tokens")
        )
    if v.get("enabled") is False:
        return ClaudeThinkingConfigDisabled(type="disabled")

    # Check for type: enabled/disabled format
    if v.get("type") == "enabled":
        return ClaudeThinkingConfigEnabled(
            type="enabled", budget_tokens=v.get("budget_tokens")
        )
    if v.get("type") == "disabled":
        return ClaudeThinkingConfigDisabled(type="disabled")

    return v


# === Usage and Token Classes ===
class CompletionTokensDetails(BaseModel):
    """Detailed breakdown of completion token usage."""

    reasoning_tokens: int | None = None  # Tokens used for reasoning/thinking
    accepted_prediction_tokens: int | None = None  # Tokens from accepted predictions
    rejected_prediction_tokens: int | None = None  # Tokens from rejected predictions


class PromptTokensDetails(BaseModel):
    """Detailed breakdown of prompt token usage."""

    cached_tokens: int | None = None  # Number of tokens read from cache


class CacheCreation(BaseModel):
    """Breakdown of cached tokens by TTL (time-to-live)."""

    ephemeral_1h_input_tokens: int = 0  # Input tokens used to create 1-hour cache entry
    ephemeral_5m_input_tokens: int = (
        0  # Input tokens used to create 5-minute cache entry
    )


class ServerToolUse(BaseModel):
    """Server tool usage tracking."""

    web_search_requests: int = 0  # Number of web search tool requests


class ClaudeUsage(BaseModel):
    """
    Billing and rate-limit usage information.

    Anthropic's API bills and rate-limits by token counts, as tokens represent
    the underlying cost to our systems. The token counts may not match one-to-one
    with the exact visible content due to internal transformations and parsing.

    Total input tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens
    """

    # Core Claude fields (required)
    input_tokens: int  # Number of input tokens used
    output_tokens: int  # Number of output tokens generated
    cache_creation_input_tokens: int | None = 0  # Tokens used to create cache entry
    cache_read_input_tokens: int | None = 0  # Tokens read from cache

    # OpenAI/Deepseek additional fields for compatibility
    prompt_tokens: int | None = None  # Alternative name for input_tokens
    completion_tokens: int | None = None  # Alternative name for output_tokens
    total_tokens: int | None = None  # Sum of input and output tokens
    prompt_cache_hit_tokens: int | None = None  # Cache hit tokens (prompt)
    prompt_cache_miss_tokens: int | None = None  # Cache miss tokens (prompt)

    # Detailed breakdown objects
    completion_tokens_details: CompletionTokensDetails | None = (
        None  # Detailed completion token breakdown
    )
    prompt_tokens_details: PromptTokensDetails | None = (
        None  # Detailed prompt token breakdown
    )
    cache_creation: CacheCreation | None = None  # Cache creation token breakdown by TTL
    server_tool_use: ServerToolUse | None = None  # Server tool usage statistics
    service_tier: str | None = None  # Service tier used (standard, priority, batch)


class GlobalUsageStats(BaseModel):
    """Thread-safe global usage statistics tracking for the proxy session."""

    # Session metadata
    session_start_time: datetime = datetime.now()
    total_requests: int = 0

    # Core token counters
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # Cache-related counters
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_hit_tokens: int = 0
    total_cache_miss_tokens: int = 0

    # Reasoning tokens (from thinking models)
    total_reasoning_tokens: int = 0

    # Model usage breakdown
    model_usage_count: dict[str, int] = {}

    def __init__(self, **data):
        super().__init__(**data)
        self._lock = threading.Lock()

    def update_usage(self, usage: ClaudeUsage, model: str = "unknown"):
        """Thread-safe method to update usage statistics."""
        with self._lock:
            self.total_requests += 1

            # Core tokens
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens

            # Calculate total tokens
            if usage.total_tokens is not None:
                self.total_tokens += usage.total_tokens
            else:
                self.total_tokens += usage.input_tokens + usage.output_tokens

            # Cache tokens
            if usage.cache_creation_input_tokens:
                self.total_cache_creation_tokens += usage.cache_creation_input_tokens
            if usage.cache_read_input_tokens:
                self.total_cache_read_tokens += usage.cache_read_input_tokens
            if usage.prompt_cache_hit_tokens:
                self.total_cache_hit_tokens += usage.prompt_cache_hit_tokens
            if usage.prompt_cache_miss_tokens:
                self.total_cache_miss_tokens += usage.prompt_cache_miss_tokens

            # Reasoning tokens
            if (
                usage.completion_tokens_details
                and usage.completion_tokens_details.reasoning_tokens
            ):
                self.total_reasoning_tokens += (
                    usage.completion_tokens_details.reasoning_tokens
                )

            # Model usage tracking
            if model not in self.model_usage_count:
                self.model_usage_count[model] = 0
            self.model_usage_count[model] += 1

    def get_session_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the session statistics."""
        with self._lock:
            session_duration = datetime.now() - self.session_start_time
            return {
                "session_duration_seconds": int(session_duration.total_seconds()),
                "total_requests": self.total_requests,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_tokens,
                "total_cache_creation_tokens": self.total_cache_creation_tokens,
                "total_cache_read_tokens": self.total_cache_read_tokens,
                "total_cache_hit_tokens": self.total_cache_hit_tokens,
                "total_cache_miss_tokens": self.total_cache_miss_tokens,
                "total_reasoning_tokens": self.total_reasoning_tokens,
                "model_usage_count": dict(self.model_usage_count),
                "session_start_time": self.session_start_time.isoformat(),
            }

    def reset_stats(self):
        """Reset all statistics and start a new session."""
        with self._lock:
            self.session_start_time = datetime.now()
            self.total_requests = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_tokens = 0
            self.total_cache_creation_tokens = 0
            self.total_cache_read_tokens = 0
            self.total_cache_hit_tokens = 0
            self.total_cache_miss_tokens = 0
            self.total_reasoning_tokens = 0
            self.model_usage_count.clear()


# Global instance for tracking session usage
global_usage_stats = GlobalUsageStats()


class ClaudeTokenCountRequest(BaseModel):
    """
    Request model for token counting endpoint.

    Used to estimate token usage for a given request without actually
    sending it to the model.
    """

    model_config = ConfigDict(
        validate_assignment=False,
        str_strip_whitespace=False,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    model: str  # Model identifier for token counting
    messages: list["ClaudeMessage"]  # Messages to count tokens for
    system: str | list[ClaudeSystemContent] | None = None  # System prompt (optional)
    tools: list[ClaudeTool] | None = None  # Tools definition (optional)
    thinking: (  # Thinking configuration (optional)
        ClaudeThinkingConfigEnabled | ClaudeThinkingConfigDisabled | dict | None
    ) = None
    tool_choice: dict[str, Any] | None = None  # Tool choice configuration (optional)

    @field_validator("thinking")
    @classmethod
    def validate_thinking_field(cls, v):
        return parse_thinking_config(v)

    def calculate_tokens(self) -> int:
        from .utils import count_tokens_in_messages, count_tokens_in_payload

        total_tokens = count_tokens_in_messages(self.messages, self.model)
        total_tokens += count_tokens_in_payload(self.system)
        total_tokens += count_tokens_in_payload(self.tools)
        total_tokens += count_tokens_in_payload(self.thinking)
        total_tokens += count_tokens_in_payload(self.tool_choice)
        return total_tokens


class ClaudeTokenCountResponse(BaseModel):
    """Response model for token counting endpoint."""

    input_tokens: int  # Estimated number of input tokens for the request


class ClaudeMessage(BaseModel):
    """
    Input message for Claude API.

    Messages represent conversational turns with either user or assistant roles.
    Our models are trained to operate on alternating user and assistant turns.
    Each message must have a role and content, where content can be a simple string
    or an array of content blocks for multimodal inputs.

    See: https://docs.anthropic.com/en/api/messages.md
    """

    model_config = ConfigDict(
        # Optimize performance for message processing
        validate_assignment=False,
        str_strip_whitespace=False,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    role: Literal["user", "assistant"]  # The role of the message sender
    content: (  # Message content - either simple string or array of content blocks
        str  # Simple text content
        | list[  # Array of structured content blocks
            ClaudeContentBlockText  # Text content
            | ClaudeContentBlockImage  # Image content (base64 or URL)
            | ClaudeContentBlockToolUse  # Tool invocation by assistant
            | ClaudeContentBlockToolResult  # Tool result from user
            | ClaudeContentBlockThinking  # Internal reasoning (thinking)
        ]
    )

    def process_interrupted_content(self) -> str:
        """Process Claude Code interrupted messages."""
        if not isinstance(self.content, str):
            return ""

        content = self.content
        if content.startswith("[Request interrupted by user for tool use]"):
            # Split the interrupted message
            interrupted_prefix = "[Request interrupted by user for tool use]"
            remaining_content = content[len(interrupted_prefix) :].strip()
            return remaining_content if remaining_content else content

        return content

    def to_openai_messages(self, tool_name_mapping: dict[str, str] = None) -> list[ChatCompletionMessageParam]:
        """
        Convert Claude message (user/assistant) to OpenAI message format (user/assistant/tool).
        Handles complex logic including tool_result splitting, content block ordering, etc.
        Returns a list of OpenAI messages (can be multiple due to tool_result splitting).
        
        Args:
            tool_name_mapping: Optional mapping from tool_use_id to tool_name for tool_result messages
        """
        openai_messages = []

        # Handle simple string content
        if isinstance(self.content, str):
            openai_messages.append({"role": self.role, "content": self.content})
            return openai_messages

        # Process content blocks in order, maintaining structure
        openai_parts = []
        merged_text = ""

        if self.role == "assistant":
            # assistant
            tool_calls: list[ChatCompletionMessageToolCallParam] = []
            assistant_msg: ChatCompletionAssistantMessageParam = {"role": "assistant"}
            has_non_text_content = False

            for block in self.content:
                if isinstance(block, ClaudeContentBlockThinking):
                    merged_text += block.thinking
                elif isinstance(block, ClaudeContentBlockText):
                    merged_text += block.text
                elif isinstance(block, ClaudeContentBlockImage):
                    has_non_text_content = True
                    part = block.to_openai()
                    if part:
                        if len(merged_text) > 0:
                            openai_parts.append({"type": "text", "text": merged_text})
                            merged_text = ""
                        openai_parts.append(part)
                elif isinstance(block, ClaudeContentBlockToolUse):
                    if len(merged_text) > 0:
                        openai_parts.append({"type": "text", "text": merged_text})
                        merged_text = ""
                    tool_calls.append(block.to_openai())

            # Handle remaining text content
            if len(merged_text) > 0:
                openai_parts.append({"type": "text", "text": merged_text})

            # Collect all text content into a single string
            text_content = ""
            for part in openai_parts:
                if part.get("type") == "text":
                    text_content += part.get("text", "") + "\n"

            # Trim the text content
            trimmed_text_content = text_content.strip()

            # Set content - use text content if available
            if trimmed_text_content:
                assistant_msg["content"] = trimmed_text_content
            else:
                assistant_msg["content"] = None

            # Add tool_calls if there are any
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls

            # Only add the message if it has content or tool calls
            if assistant_msg.get("content") or assistant_msg.get("tool_calls"):
                openai_messages.append(assistant_msg)
        else:
            # user
            pending_tool_result_msgs: list[ChatCompletionToolMessageParam] = []
            user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": ""}
            has_non_text_content = False

            for block in self.content:
                if isinstance(block, ClaudeContentBlockText):
                    merged_text += block.text
                elif isinstance(block, ClaudeContentBlockImage):
                    has_non_text_content = True
                    part = block.to_openai()
                    if part:
                        if len(merged_text) > 0:
                            openai_parts.append({"type": "text", "text": merged_text})
                            merged_text = ""
                        openai_parts.append(part)
                elif isinstance(block, ClaudeContentBlockToolResult):
                    # split user message -> tool result message + user message
                    if len(merged_text) > 0:
                        openai_parts.append({"type": "text", "text": merged_text})
                        merged_text = ""
                    # Get tool name from mapping if available
                    tool_name = None
                    if tool_name_mapping:
                        tool_name = tool_name_mapping.get(block.tool_use_id)
                    pending_tool_result_msgs.append(block.to_openai_message(tool_name))

            # Handle remaining text content
            if len(merged_text) > 0:
                openai_parts.append({"type": "text", "text": merged_text})

            # Set content: use string for text-only, structured for mixed content
            if has_non_text_content or len(openai_parts) > 1:
                user_msg["content"] = openai_parts
            elif len(openai_parts) == 1 and openai_parts[0]["type"] == "text":
                user_msg["content"] = openai_parts[0]["text"]
            else:
                user_msg["content"] = ""

            # Tool results should come before user messages for proper OpenAI sequencing
            openai_messages.extend(pending_tool_result_msgs)

            # Only add user message if it has actual content (not empty)
            if user_msg["content"] and user_msg["content"] != "":
                openai_messages.append(user_msg)

        return openai_messages


class ClaudeMessagesRequest(BaseModel):
    """
    Claude Messages API request model.

    Send a structured list of input messages with text and/or image content,
    and the model will generate the next message in the conversation.
    The Messages API can be used for either single queries or stateless multi-turn conversations.

    See: https://docs.anthropic.com/en/api/messages.md
    """

    model_config = ConfigDict(
        # Optimize performance for high-throughput proxy server
        validate_assignment=False,  # Skip validation on assignment for speed
        str_strip_whitespace=False,  # Skip string stripping for performance
        use_enum_values=True,  # Use enum values directly
        arbitrary_types_allowed=True,  # Allow arbitrary types for flexibility
        extra="ignore",  # Ignore extra fields instead of validation
    )

    # Required fields
    model: str  # The model that will complete your prompt (e.g., "claude-sonnet-4-20250514")
    max_tokens: int  # Maximum number of tokens to generate before stopping (minimum: 1)
    messages: list[
        ClaudeMessage
    ]  # Input messages (limit: 100,000 messages per request)

    # Optional fields
    system: str | list[ClaudeSystemContent] | None = (
        None  # System prompt for context and instructions
    )
    stop_sequences: list[str] | None = (
        None  # Custom text sequences that will cause the model to stop generating
    )
    stream: bool | None = (
        False  # Whether to incrementally stream the response using server-sent events
    )
    temperature: float | None = (
        1.0  # Amount of randomness injected (0.0-1.0, default: 1.0)
    )
    top_p: float | None = (
        None  # Use nucleus sampling (0.0-1.0, alternative to temperature)
    )
    top_k: int | None = (
        None  # Only sample from top K options (minimum: 0, for advanced use cases)
    )
    metadata: dict[str, Any] | None = (
        None  # Metadata about the request (e.g., user_id for abuse detection)
    )
    tools: list[ClaudeTool] | None = None  # Definitions of tools that the model may use
    tool_choice: ClaudeToolChoice | None = (
        None  # How the model should use the provided tools
    )
    thinking: ClaudeThinkingConfigEnabled | ClaudeThinkingConfigDisabled | None = (
        None  # Extended thinking configuration
    )

    @field_validator("thinking")
    @classmethod
    def validate_thinking_field(cls, v):
        return parse_thinking_config(v)

    def extract_system_content(self) -> str:
        """Extract system content from various formats."""
        if not self.system:
            return ""

        if isinstance(self.system, str):
            return self.system
        elif isinstance(self.system, list):
            system_content = ""
            for block in self.system:
                if hasattr(block, "text"):
                    system_content += block.text
                elif isinstance(block, dict) and "text" in block:
                    system_content += block["text"]
            return system_content
        return ""

    def to_openai_request(self) -> dict[str, Any]:
        """Convert Anthropic API request to OpenAI API format using OpenAI SDK types for validation."""
        logger.debug(
            f"🔄 Converting Claude request to OpenAI format for model: {self.model}"
        )
        logger.debug(f"🔄 Input messages count: {len(self.messages)}")

        # Log message types for debugging
        for i, msg in enumerate(self.messages):
            has_tool_use = False
            has_tool_result = False
            if isinstance(msg.content, list):
                for block in msg.content:
                    if block.type == Constants.CONTENT_TOOL_USE:
                        has_tool_use = True
                    elif block.type == Constants.CONTENT_TOOL_RESULT:
                        has_tool_result = True
            logger.debug(
                f"🔄 Message {i} ({msg.role}): tool_use={has_tool_use}, tool_result={has_tool_result}"
            )

        # Build OpenAI messages with type validation
        openai_messages: list[ChatCompletionMessageParam] = []

        # Convert system message if present
        if self.system:
            system_content = self.extract_system_content()
            if system_content:
                system_msg: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": system_content,
                }
                openai_messages.append(system_msg)

        # Build tool_name_mapping by scanning all messages for tool_use blocks
        tool_name_mapping: dict[str, str] = {}
        for msg in self.messages:
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ClaudeContentBlockToolUse):
                        tool_name_mapping[block.id] = block.name

        # Convert Claude messages to OpenAI messages
        for msg in self.messages:
            if msg.role == Constants.ROLE_USER:
                user_messages = msg.to_openai_messages(tool_name_mapping)
                openai_messages.extend(user_messages)
            elif msg.role == Constants.ROLE_ASSISTANT:
                assistant_messages = msg.to_openai_messages(tool_name_mapping)
                openai_messages.extend(assistant_messages)

        # Claude request tools -> OpenAI request tools
        openai_tools = []
        if self.tools:
            logger.debug(f"🔧 TOOL_DEBUG: Converting {len(self.tools)} Claude tools to OpenAI format")
            openai_tools: list[ChatCompletionToolParam] = []
            for i, tool in enumerate(self.tools):
                logger.debug(f"🔧 TOOL_DEBUG: Tool {i}: name={tool.name}, description={tool.description}")
                tool_params: ChatCompletionToolParam = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "strict": False,
                    },
                }
                if hasattr(tool, "input_schema"):
                    from .converter import (
                        clean_gemini_schema,
                        validate_gemini_function_schema,
                    )

                    logger.debug(f"🔧 TOOL_DEBUG: Tool {i} input_schema: {tool.input_schema}")
                    cleaned_schema = clean_gemini_schema(tool.input_schema)
                    logger.debug(f"🔧 TOOL_DEBUG: Tool {i} cleaned_schema: {cleaned_schema}")
                    tool_params["function"]["parameters"] = cleaned_schema
                    is_valid, errors = validate_gemini_function_schema(tool_params)
                    logger.debug(f"🔧 TOOL_DEBUG: Tool {i} is valid: {is_valid} errors: {errors}")
                else:
                    # If no input_schema, add empty parameters object
                    logger.debug(f"🔧 TOOL_DEBUG: Tool {i} has no input_schema, adding empty parameters")
                    tool_params["function"]["parameters"] = {}

                openai_tools.append(tool_params)
                logger.debug(f"🔧 TOOL_DEBUG: Tool {i} converted successfully")
            logger.debug(f"🔧 TOOL_DEBUG: Final OpenAI tools count: {len(openai_tools)}")

        # Handle tool_choice with type validation
        tool_choice = None
        if self.tool_choice:
            tool_choice = self.tool_choice.to_openai()
            logger.debug(
                f"openai tool choice param: {tool_choice} <- {self.tool_choice}"
            )

        request_params = {
            "model": self.model,
            "messages": openai_messages,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if openai_tools:
            request_params["tools"] = openai_tools
            logger.debug(f"🔧 TOOL_DEBUG: Added {len(openai_tools)} tools to OpenAI request")
        if tool_choice:
            request_params["tool_choice"] = tool_choice
            logger.debug(f"🔧 TOOL_DEBUG: Set tool_choice to: {tool_choice}")
        else:
            logger.debug("🔧 TOOL_DEBUG: No tool_choice specified")

        logger.debug(f"🔄 Output messages count: {len(openai_messages)}")

        # DEBUG: Validate and debug OpenAI message sequence
        from .utils import _compare_request_data, _debug_openai_message_sequence

        _debug_openai_message_sequence(openai_messages, "claude_to_openai_conversion")
        # Compare request data and log any mismatches
        _compare_request_data(self, request_params)

        return request_params

class ClaudeMessagesResponse(BaseModel):
    """
    Claude Messages API response model.

    The response from Claude after processing a message request.
    Contains the generated content, metadata, and usage information.

    See: https://docs.anthropic.com/en/api/messages.md
    """

    id: str  # Unique object identifier (format may change over time)
    type: Literal["message"] = (
        "message"  # Object type - always "message" for Messages API
    )
    role: Literal["assistant"] = (
        "assistant"  # Conversational role - always "assistant" for responses
    )
    model: str  # The model that handled the request
    content: list[  # Content generated by the model (array of content blocks)
        ClaudeContentBlockText | ClaudeContentBlockToolUse | ClaudeContentBlockThinking
    ]
    stop_reason: (  # The reason that generation stopped
        Literal[
            "end_turn",  # Model reached a natural stopping point
            "max_tokens",  # Exceeded requested max_tokens or model's maximum
            "stop_sequence",  # One of your custom stop_sequences was generated
            "tool_use",  # Model invoked one or more tools
            "pause_turn",  # Paused a long-running turn (can be continued)
            "refusal",  # Streaming classifiers intervened for policy violations
            "error",  # An error occurred during generation
        ]
        | None
    ) = None
    stop_sequence: str | None = None  # Which custom stop sequence was generated, if any
    usage: ClaudeUsage  # Billing and rate-limit usage information
