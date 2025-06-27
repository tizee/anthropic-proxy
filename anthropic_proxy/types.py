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
    ChatCompletionAssistantMessageParam,
)
from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)


class ModelDefaults:
    """Default values and limits for model configurations"""

    # Token limits
    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_MAX_INPUT_TOKENS = 128000
    MAX_TOKENS_LIMIT = 16384
    LONG_CONTEXT_THRESHOLD = 128000

    # Pricing defaults
    DEFAULT_INPUT_COST_PER_TOKEN = 0.000001
    DEFAULT_OUTPUT_COST_PER_TOKEN = 0.000002

    # Hash signature length
    THINKING_SIGNATURE_LENGTH = 16

    # Default server settings
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = 8082
    DEFAULT_LOG_LEVEL = "WARNING"
    DEFAULT_MAX_RETRIES = 2


class Constants:
    """Constants for better maintainability"""

    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_TOOL = "tool"

    CONTENT_TEXT = "text"
    CONTENT_IMAGE = "image"
    CONTENT_TOOL_USE = "tool_use"
    CONTENT_TOOL_RESULT = "tool_result"

    TOOL_FUNCTION = "function"

    STOP_END_TURN = "end_turn"
    STOP_MAX_TOKENS = "max_tokens"
    STOP_TOOL_USE = "tool_use"
    STOP_ERROR = "error"

    EVENT_MESSAGE_START = "message_start"
    EVENT_MESSAGE_STOP = "message_stop"
    EVENT_MESSAGE_DELTA = "message_delta"
    EVENT_CONTENT_BLOCK_START = "content_block_start"
    EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
    EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
    EVENT_PING = "ping"

    DELTA_TEXT = "text_delta"
    DELTA_INPUT_JSON = "input_json_delta"


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
    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: bool | None = None

    def to_openai(self) -> ChatCompletionToolChoiceOptionParam:
        return "auto"


class ClaudeToolChoiceAny(BaseModel):
    type: Literal["any"] = "any"
    disable_parallel_tool_use: bool | None = None

    def to_openai(self) -> ChatCompletionToolChoiceOptionParam:
        return "required"


class ClaudeToolChoiceTool(BaseModel):
    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: bool | None = None

    def to_openai(self) -> ChatCompletionNamedToolChoiceParam:
        return {"type": "function", "function": {"name": self.name}}


class ClaudeToolChoiceNone(BaseModel):
    type: Literal["none"] = "none"

    def to_openai(self) -> ChatCompletionToolChoiceOptionParam:
        return "none"


# Union type for all tool choice options
ClaudeToolChoice = (
    ClaudeToolChoiceAuto
    | ClaudeToolChoiceAny
    | ClaudeToolChoiceTool
    | ClaudeToolChoiceNone
)


# === Content Block Classes ===
class ClaudeContentBlockText(BaseModel):
    model_config = ConfigDict(
        validate_assignment=False, str_strip_whitespace=False, extra="ignore"
    )

    type: Literal["text"]
    text: str

    def to_openai(self) -> ChatCompletionContentPartTextParam:
        """Convert Claude text block to OpenAI text format."""
        return {"type": "text", "text": self.text}


class ClaudeContentBlockImageBase64Source(BaseModel):
    type: Literal["base64"]
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    data: str


class ClaudeContentBlockImageURLSource(BaseModel):
    type: Literal["url"]
    url: str


class ClaudeContentBlockImage(BaseModel):
    model_config = ConfigDict(
        validate_assignment=False, str_strip_whitespace=False, extra="ignore"
    )

    type: Literal["image"]
    source: (
        ClaudeContentBlockImageBase64Source
        | ClaudeContentBlockImageURLSource
        | dict[str, Any]
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
    model_config = ConfigDict(
        validate_assignment=False, str_strip_whitespace=False, extra="ignore"
    )

    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]

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
    model_config = ConfigDict(
        validate_assignment=False, str_strip_whitespace=False, extra="ignore"
    )

    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict[str, Any]] | dict[str, Any]

    def process_content(self) -> str | list:
        """Process Claude tool_result content into a string format."""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            content_parts = []
            for item in self.content:
                content_parts.append({"type": "text", "text": item["content"]})
            return content_parts
        else:
            try:
                return json.dumps(
                    self.content, ensure_ascii=False, separators=(",", ":")
                )
            except (TypeError, ValueError):
                return "{}"

    def to_openai_message(self) -> ChatCompletionToolMessageParam:
        """Convert Claude tool_result to OpenAI tool role message format."""
        return {
            "role": "tool",
            "tool_call_id": self.tool_use_id,
            "content": self.process_content(),
        }


class ClaudeContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None

    def to_openai(self) -> ChatCompletionContentPartTextParam:
        """Thinking blocks should be transformed to assistant text message"""
        return {"type": "text", "text": self.thinking}


class ClaudeSystemContent(BaseModel):
    type: Literal["text"]
    text: str


class ClaudeTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class ClaudeThinkingConfigEnabled(BaseModel):
    type: Literal["enabled"] = "enabled"
    budget_tokens: int | None = None


class ClaudeThinkingConfigDisabled(BaseModel):
    type: Literal["disabled"] = "disabled"


# === Usage and Token Classes ===
class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int | None = None
    accepted_prediction_tokens: int | None = None
    rejected_prediction_tokens: int | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int | None = None


class CacheCreation(BaseModel):
    ephemeral_1h_input_tokens: int = 0
    ephemeral_5m_input_tokens: int = 0


class ServerToolUse(BaseModel):
    web_search_requests: int = 0


class ClaudeUsage(BaseModel):
    # Core Claude fields (existing)
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = 0
    cache_read_input_tokens: int | None = 0

    # OpenAI/Deepseek additional fields for compatibility
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None

    # Detailed breakdown objects
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None
    cache_creation: CacheCreation | None = None
    server_tool_use: ServerToolUse | None = None
    service_tier: str | None = None


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
    model_config = ConfigDict(
        validate_assignment=False,
        str_strip_whitespace=False,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    model: str
    messages: list["ClaudeMessage"]
    system: str | list[ClaudeSystemContent] | None = None
    tools: list[ClaudeTool] | None = None
    thinking: (
        ClaudeThinkingConfigEnabled | ClaudeThinkingConfigDisabled | dict | None
    ) = None
    tool_choice: dict[str, Any] | None = None

    @field_validator("thinking")
    @classmethod
    def validate_thinking_field(cls, v):
        if isinstance(v, dict):
            if v.get("enabled") is True:
                return ClaudeThinkingConfigEnabled(
                    type="enabled", budget_tokens=v.get("budget_tokens")
                )
            elif v.get("enabled") is False:
                return ClaudeThinkingConfigDisabled(type="disabled")
            elif v.get("type") == "enabled":
                return ClaudeThinkingConfigEnabled(
                    type="enabled", budget_tokens=v.get("budget_tokens")
                )
            elif v.get("type") == "disabled":
                return ClaudeThinkingConfigDisabled(type="disabled")
        return v

    def calculate_tokens(self) -> int:
        from .utils import count_tokens_in_messages

        return count_tokens_in_messages(self.messages, self.model)


class ClaudeTokenCountResponse(BaseModel):
    input_tokens: int


class ClaudeMessage(BaseModel):
    model_config = ConfigDict(
        # Optimize performance for message processing
        validate_assignment=False,
        str_strip_whitespace=False,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="ignore",
    )

    role: Literal["user", "assistant"]
    content: (
        str
        | list[
            ClaudeContentBlockText
            | ClaudeContentBlockImage
            | ClaudeContentBlockToolUse
            | ClaudeContentBlockToolResult
            | ClaudeContentBlockThinking
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

    def to_openai_messages(self) -> list[ChatCompletionMessageParam]:
        """
        Convert Claude message (user/assistant) to OpenAI message format (user/assistant/tool).
        Handles complex logic including tool_result splitting, content block ordering, etc.
        Returns a list of OpenAI messages (can be multiple due to tool_result splitting).
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
                    pending_tool_result_msgs.append(block.to_openai_message())

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
    model_config = ConfigDict(
        # Optimize performance for high-throughput proxy server
        validate_assignment=False,  # Skip validation on assignment for speed
        str_strip_whitespace=False,  # Skip string stripping for performance
        use_enum_values=True,  # Use enum values directly
        arbitrary_types_allowed=True,  # Allow arbitrary types for flexibility
        extra="ignore",  # Ignore extra fields instead of validation
    )

    model: str
    max_tokens: int
    messages: list[ClaudeMessage]
    system: str | list[ClaudeSystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    temperature: float | None = 1.0
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[ClaudeTool] | None = None
    tool_choice: ClaudeToolChoice | None = None
    thinking: ClaudeThinkingConfigEnabled | ClaudeThinkingConfigDisabled | None = None

    @field_validator("thinking")
    @classmethod
    def validate_thinking_field(cls, v):
        if isinstance(v, dict):
            if v.get("enabled") is True:
                return ClaudeThinkingConfigEnabled(
                    type="enabled", budget_tokens=v.get("budget_tokens")
                )
            elif v.get("enabled") is False:
                return ClaudeThinkingConfigDisabled(type="disabled")
            elif v.get("type") == "enabled":
                return ClaudeThinkingConfigEnabled(
                    type="enabled", budget_tokens=v.get("budget_tokens")
                )
            elif v.get("type") == "disabled":
                return ClaudeThinkingConfigDisabled(type="disabled")
        return v

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

        # Convert Claude messages to OpenAI messages
        for msg in self.messages:
            if msg.role == Constants.ROLE_USER:
                user_messages = msg.to_openai_messages()
                openai_messages.extend(user_messages)
            elif msg.role == Constants.ROLE_ASSISTANT:
                assistant_messages = msg.to_openai_messages()
                openai_messages.extend(assistant_messages)

        # Claude request tools -> OpenAI request tools
        openai_tools = []
        if self.tools:
            openai_tools: list[ChatCompletionToolParam] = []
            for tool in self.tools:
                tool_params: ChatCompletionToolParam = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                    },
                }
                if hasattr(tool, "input_schema"):
                    from .converter import clean_gemini_schema

                    tool_params["function"]["parameters"] = clean_gemini_schema(
                        tool.input_schema
                    )
                    openai_tools.append(tool_params)

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
        if tool_choice:
            request_params["tool_choice"] = tool_choice

        logger.debug(f"🔄 Output messages count: {len(openai_messages)}")

        # DEBUG: Validate and debug OpenAI message sequence
        from .utils import _debug_openai_message_sequence, _compare_request_data

        _debug_openai_message_sequence(openai_messages, "claude_to_openai_conversion")
        # Compare request data and log any mismatches
        _compare_request_data(self, request_params)

        return request_params

    def calculate_tokens(self) -> int:
        from .utils import count_tokens_in_messages

        return count_tokens_in_messages(self.messages, self.model)


class ClaudeMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[
        ClaudeContentBlockText | ClaudeContentBlockToolUse | ClaudeContentBlockThinking
    ]
    stop_reason: (
        Literal[
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
            "error",
        ]
        | None
    ) = None
    stop_sequence: str | None = None
    usage: ClaudeUsage
