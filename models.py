"""
Pydantic models for Claude proxy API requests and responses.
This module contains only the model definitions without any server startup code.
"""

import hashlib
import json
import logging
import re
import threading
import uuid
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal, Union

import tiktoken
from openai import AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from pydantic import BaseModel, field_validator


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


logger = logging.getLogger(__name__)

try:
    enc = tiktoken.get_encoding("cl100k_base")
    logger.debug("✅ TikToken encoder initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize TikToken encoder: {e}")
    enc = None

# Constants for better maintainability


class Constants:
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


def generate_thinking_signature(thinking_content: str) -> str:
    """Generate a signature for thinking content using SHA-256 hash."""
    if not thinking_content:
        return ""

    # Create SHA-256 hash of the thinking content
    hash_object = hashlib.sha256(thinking_content.encode("utf-8"))
    signature = hash_object.hexdigest()[: ModelDefaults.THINKING_SIGNATURE_LENGTH]
    return f"thinking_{signature}"


def parse_function_calls_from_thinking(thinking_content: str) -> tuple[str, list]:
    """Parse function calls from thinking content with custom markers.

    Returns:
        tuple: (cleaned_thinking_content, list_of_tool_calls)
    """
    import json

    # Pattern to match function call blocks
    pattern = r"<\|FunctionCallBegin\|>\[(.*?)\]<\|FunctionCallEnd\|>"

    tool_calls = []
    cleaned_content = thinking_content

    matches = re.findall(pattern, thinking_content, re.DOTALL)

    for match in matches:
        try:
            # Parse the JSON array of function calls
            function_call_data = json.loads(f"[{match}]")

            for call_data in function_call_data:
                if (
                    isinstance(call_data, dict)
                    and "name" in call_data
                    and "parameters" in call_data
                ):
                    # Create a tool call in OpenAI format
                    tool_call = {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": call_data["name"],
                            "arguments": json.dumps(call_data["parameters"]),
                        },
                    }
                    tool_calls.append(tool_call)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse function call from thinking content: {e}")
            continue

    # Remove the function call markers from thinking content
    cleaned_content = re.sub(pattern, "", thinking_content, flags=re.DOTALL).strip()

    return cleaned_content, tool_calls


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
ClaudeToolChoice = Union[
    ClaudeToolChoiceAuto,
    ClaudeToolChoiceAny,
    ClaudeToolChoiceTool,
    ClaudeToolChoiceNone,
]


class ClaudeContentBlockText(BaseModel):
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
    type: Literal["image"]
    source: ClaudeContentBlockImageBase64Source | ClaudeContentBlockImageURLSource | dict[str, Any]

    # only user message contains image content
    #
    def to_openai(self) -> ChatCompletionContentPartImageParam | None:
        """
        Convert Claude image block to OpenAI image_url format.

        Claude format:
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAB..."
            }
        }

        OpenAI format:
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAB..."
            }
        }
        """
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
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]

    # only assistant message contains tool_calls
    def to_openai(self) -> ChatCompletionMessageToolCallParam:
        """
        Convert Claude tool_use to OpenAI tool_call format.

        Claude format:
        {
            "type": "tool_use",
            "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "name": "get_stock_price",
            "input": { "ticker": "^GSPC" }
        }

        OpenAI format:
        {
            "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "arguments": "{\"ticker\":\"^GSPC\"}"
            }
        }
        """
        import json

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
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict[str, Any]]

    def process_content(
        self,
    ) -> str | Iterable[ChatCompletionContentPartTextParam]:
        """
        Process Claude tool_result content into a string format.

        Claude supports various content formats:
        - Simple string: "259.75 USD"
        - List with text blocks: [{"type": "text", "text": "result"}]
        - List with image block (OpenAI models only support text content parts)
        """

        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # Handle list content by extracting all text
            content_parts: list[ChatCompletionContentPartTextParam] = []
            for item in self.content:
                content_parts.append({"type": "text", "text": item["content"]})
            return content_parts
        else:
            # Fallback: serialize anything else
            return ""

    # Claude Tool Result content block -> OpenAI Tool Role Message
    def to_openai_message(self) -> ChatCompletionToolMessageParam:
        """
        Convert Claude tool_result to OpenAI tool role message format.

        Claude format:
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "content": "259.75 USD"
        }

        OpenAI format:
        {
            "role": "tool",
            "tool_call_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "content": "259.75 USD"
        }
        """
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


class ClaudeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ClaudeContentBlockText | ClaudeContentBlockImage | ClaudeContentBlockToolUse | ClaudeContentBlockToolResult | ClaudeContentBlockThinking]

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

        Note:
            For Claude user message, there is no tool call and thinking content.
            tool call content block and thinking content is for assistant message only
        """
        openai_messages = []

        # Handle simple string content
        if isinstance(self.content, str):
            # processed_content = self.process_interrupted_content()
            openai_messages.append({"role": self.role, "content": self.content})
            return openai_messages

        # Process content blocks in order, maintaining structure
        content_parts: list[
            ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam
        ] = []
        tool_calls: list[ChatCompletionMessageToolCallParam] = []

        # Claude user message content blocks -> OpenAI user message content
        # parts
        # https://docs.anthropic.com/en/api/messages#body-messages-content
        # https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages

        # correct order: assistant message with tool_calls should always
        # be before than assistant message with tool result.
        for block in self.content:
            if not isinstance(block, ClaudeContentBlockToolResult) and not isinstance(
                block, ClaudeContentBlockToolUse
            ):
                openai_part = block.to_openai()
                if openai_part:
                    content_parts.append(openai_part)
            elif isinstance(block, ClaudeContentBlockToolUse):
                tool_calls.append(block.to_openai())
            elif isinstance(block, ClaudeContentBlockToolResult):
                # CRITICAL: Split message when tool_result is encountered
                if content_parts:
                    current_message: dict[str, Any] = {"role": self.role}
                    if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                        current_message["content"] = content_parts[0]["text"]
                        # openai_messages.append(
                        #     {
                        #         "role": self.role,
                        #         "content": content_parts[0]["text"],
                        #         "tool_calls": tool_calls}
                        # )
                    else:
                        current_message["content"] = content_parts
                        # openai_messages.append(
                        #     {"role": self.role, "content": content_parts}
                        # )
                    if self.role == "assistant" and len(tool_calls) > 0:
                        current_message["tool_calls"] = tool_calls
                    openai_messages.append(current_message)
                    content_parts.clear()
                    tool_calls.clear()
                # Add tool result immediately to maintain chronological order
                tool_message = block.to_openai_message()
                openai_messages.append(tool_message)

        # Process any remaining content
        if content_parts or (self.role == "assistant" and len(tool_calls) > 0):
            current_message: dict[str, Any] = {"role": self.role}
            if content_parts:
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    current_message["content"] = content_parts[0]["text"]
                else:
                    current_message["content"] = content_parts
            else:
                # Assistant message with only tool_calls, no content
                current_message["content"] = ""

            if self.role == "assistant" and len(tool_calls) > 0:
                current_message["tool_calls"] = tool_calls
            openai_messages.append(current_message)

        return openai_messages


class ClaudeTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class ClaudeThinkingConfigEnabled(BaseModel):
    type: Literal["enabled"] = "enabled"
    budget_tokens: int | None = None


class ClaudeThinkingConfigDisabled(BaseModel):
    type: Literal["disabled"] = "disabled"


class ClaudeMessagesRequest(BaseModel):
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

    # see https://platform.openai.com/docs/api-reference/chat/create
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

        # --- Claude Messages -> OpenAI messages ---
        # Convert system message if present
        if self.system:
            system_content = self.extract_system_content()
            if system_content:
                system_msg: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": system_content,
                }
                openai_messages.append(system_msg)

        # Note:
        # Claude only has user and assistant these
        # two roles. So the Claude assistant's tool use
        # content blocks should be added to the OpenAI
        # assistant's tool_calls.
        # For tool result content blocks, we need to create
        # tool role messages.
        # Besides, tool use should always before tool result
        for msg in self.messages:
            if msg.role == Constants.ROLE_USER:
                user_messages = msg.to_openai_messages()
                openai_messages.extend(user_messages)
            elif msg.role == Constants.ROLE_ASSISTANT:
                assistant_messages = msg.to_openai_messages()
                openai_messages.extend(assistant_messages)

        # Claude request tools -> OpenAI request tools
        # https://docs.anthropic.com/en/api/messages#body-tools
        # https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
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
                    if "gemini" in self.model.lower():
                        tool_params["function"]["parameters"] = clean_gemini_schema(
                            tool.input_schema
                        )
                    else:
                        tool_params["function"]["parameters"] = tool.input_schema
                    openai_tools.append(tool_params)
                    logger.debug(f"openai_tools append: {tool_params}")

        # Handle tool_choice with type validation
        # Claude tool_choice -> OpenAI tool_choice
        # https://docs.anthropic.com/en/api/messages#body-tool-choice
        # https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
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

        raw_json = self.model_dump()
        logger.debug(f"🔄 Output messages count: {len(openai_messages)}")
        logger.debug(f"🔄 Original request: {raw_json}")
        logger.debug(f"🔄 OpenAI request: {request_params}")

        # Note: OpenAI API requires that messages with role 'tool' must be a response
        # to a preceding message with 'tool_calls'. The current message conversion
        # logic naturally produces the correct sequence: Assistant(tool_calls) → Tool → User

        # Compare request data and log any mismatches
        _compare_request_data(self, request_params)

        # Return the request with validated components
        # Individual components (messages, tools) are already validated using OpenAI SDK types
        return request_params

    def calculate_tokens(self) -> int:
        # TODO
        return 0


class ClaudeTokenCountRequest(BaseModel):
    model: str
    messages: list[ClaudeMessage]
    system: str | list[ClaudeSystemContent] | None = None
    tools: list[ClaudeTool] | None = None
    thinking: ClaudeThinkingConfigEnabled | ClaudeThinkingConfigDisabled | dict | None = None
    tool_choice: dict[str, Any] | None = None

    @field_validator("thinking")
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
        return 0


class ClaudeTokenCountResponse(BaseModel):
    input_tokens: int


# Supporting models for detailed token breakdown
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


def extract_usage_from_openai_response(openai_response) -> ClaudeUsage:
    """Extract usage data from OpenAI API response and convert to ClaudeUsage format."""
    usage = (
        openai_response.usage
        if hasattr(openai_response, "usage") and openai_response.usage
        else None
    )

    if not usage:
        return ClaudeUsage(input_tokens=0, output_tokens=0)

    # Core fields mapping
    input_tokens = getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)

    # Extract completion_tokens_details if present
    completion_details = None
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        details = usage.completion_tokens_details
        completion_details = CompletionTokensDetails(
            reasoning_tokens=getattr(details, "reasoning_tokens", None),
            accepted_prediction_tokens=getattr(
                details, "accepted_prediction_tokens", None
            ),
            rejected_prediction_tokens=getattr(
                details, "rejected_prediction_tokens", None
            ),
        )

    # Extract prompt_tokens_details if present
    prompt_details = None
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        details = usage.prompt_tokens_details
        prompt_details = PromptTokensDetails(
            cached_tokens=getattr(details, "cached_tokens", None)
        )

    # Handle Deepseek-specific fields
    prompt_cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", None)
    prompt_cache_miss_tokens = getattr(usage, "prompt_cache_miss_tokens", None)

    return ClaudeUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
        completion_tokens_details=completion_details,
        prompt_tokens_details=prompt_details,
    )


def extract_usage_from_claude_response(
    claude_usage_dict: dict[str, Any],
) -> ClaudeUsage:
    """Extract usage data from Claude API response format and convert to enhanced ClaudeUsage."""
    if not claude_usage_dict:
        return ClaudeUsage(input_tokens=0, output_tokens=0)

    # Core fields
    input_tokens = claude_usage_dict.get("input_tokens", 0)
    output_tokens = claude_usage_dict.get("output_tokens", 0)
    cache_creation_input_tokens = claude_usage_dict.get(
        "cache_creation_input_tokens", 0
    )
    cache_read_input_tokens = claude_usage_dict.get("cache_read_input_tokens", 0)

    # Handle cache_creation object
    cache_creation = None
    if "cache_creation" in claude_usage_dict and claude_usage_dict["cache_creation"]:
        cache_data = claude_usage_dict["cache_creation"]
        cache_creation = CacheCreation(
            ephemeral_1h_input_tokens=cache_data.get("ephemeral_1h_input_tokens", 0),
            ephemeral_5m_input_tokens=cache_data.get("ephemeral_5m_input_tokens", 0),
        )

    # Handle server_tool_use object
    server_tool_use = None
    if "server_tool_use" in claude_usage_dict and claude_usage_dict["server_tool_use"]:
        tool_data = claude_usage_dict["server_tool_use"]
        server_tool_use = ServerToolUse(
            web_search_requests=tool_data.get("web_search_requests", 0)
        )

    service_tier = claude_usage_dict.get("service_tier")

    return ClaudeUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_creation=cache_creation,
        server_tool_use=server_tool_use,
        service_tier=service_tier,
    )


def update_global_usage_stats(usage: ClaudeUsage, model: str, context: str = ""):
    """Update global usage statistics and log the usage information."""
    global global_usage_stats

    # Update the global stats
    global_usage_stats.update_usage(usage, model)

    # Log current usage
    logger.info(
        f"📊 USAGE UPDATE [{context}]: Model={model}, Input={usage.input_tokens}t, Output={usage.output_tokens}t"
    )

    # Log cache-related tokens if present
    cache_info = []
    if usage.cache_read_input_tokens and usage.cache_read_input_tokens > 0:
        cache_info.append(f"CacheRead={usage.cache_read_input_tokens}t")
    if usage.cache_creation_input_tokens and usage.cache_creation_input_tokens > 0:
        cache_info.append(f"CacheCreate={usage.cache_creation_input_tokens}t")
    if usage.prompt_cache_hit_tokens and usage.prompt_cache_hit_tokens > 0:
        cache_info.append(f"CacheHit={usage.prompt_cache_hit_tokens}t")
    if usage.prompt_cache_miss_tokens and usage.prompt_cache_miss_tokens > 0:
        cache_info.append(f"CacheMiss={usage.prompt_cache_miss_tokens}t")

    if cache_info:
        logger.info(f"💾 CACHE USAGE: {', '.join(cache_info)}")

    # Log reasoning tokens if present
    if (
        usage.completion_tokens_details
        and usage.completion_tokens_details.reasoning_tokens
    ):
        logger.info(
            f"🧠 REASONING TOKENS: {usage.completion_tokens_details.reasoning_tokens}t"
        )

    # Log session totals
    summary = global_usage_stats.get_session_summary()
    logger.info(
        f"📈 SESSION TOTALS: Requests={summary['total_requests']}, Input={summary['total_input_tokens']}t, Output={summary['total_output_tokens']}t, Total={summary['total_tokens']}t"
    )

    # Log reasoning and cache totals if significant
    if summary["total_reasoning_tokens"] > 0:
        logger.info(f"🧠 SESSION REASONING: {summary['total_reasoning_tokens']}t")

    total_cache = (
        summary["total_cache_hit_tokens"]
        + summary["total_cache_miss_tokens"]
        + summary["total_cache_read_tokens"]
    )
    if total_cache > 0:
        logger.info(
            f"💾 SESSION CACHE: Hit={summary['total_cache_hit_tokens']}t, Miss={summary['total_cache_miss_tokens']}t, Read={summary['total_cache_read_tokens']}t"
        )


# see https://docs.anthropic.com/en/api/messages#response-id
class ClaudeMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[
        ClaudeContentBlockText | ClaudeContentBlockToolUse | ClaudeContentBlockThinking
    ]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal", "error"] | None = None
    stop_sequence: str | None = None
    usage: ClaudeUsage


# openai response -> claude response
def convert_openai_response_to_anthropic(
    openai_response: ChatCompletion, original_request: ClaudeMessagesRequest
) -> ClaudeMessagesResponse:
    """Convert OpenAI response back to Anthropic API format using OpenAI SDK type validation."""
    try:
        # Validate and extract response data using OpenAI SDK types
        response_id = f"msg_{uuid.uuid4()}"
        content_text = ""
        tool_calls = None
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        thinking_content = ""

        logger.debug(f"Converting OpenAI response: {type(openai_response)}")

        choice = openai_response.choices[0]
        # Extract message content
        message = choice.message
        content_text = message.content or ""
        raw_message = message.model_dump()

        # Extract reasoning_content for thinking models (OpenAI format)
        if "reasoning_content" in raw_message:
            thinking_content = raw_message["reasoning_content"]
            logger.debug(
                f"Extracted reasoning_content: {len(thinking_content)} characters"
            )

        # Extract tool calls if present
        if message.tool_calls:
            tool_calls = message.tool_calls
        # Extract finish reason
        finish_reason = choice.finish_reason

        # Extract usage information
        usage = openai_response.usage
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

        logger.debug(f"Raw content extracted: {len(content_text)} characters")
        logger.debug(f"Tool calls from response: {tool_calls}")

        # Enhanced debugging for Claude Code tool testing
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug("=== ENHANCED DEBUG INFO ===")
            if thinking_content:
                logger.debug(f"Thinking content preview: {thinking_content[:500]}...")
            if content_text:
                logger.debug(f"Raw content text: {repr(content_text)}")
            logger.debug("=== END DEBUG INFO ===")

        # Build content blocks
        content_blocks = []

        # Add thinking content first if present (for Claude Code display)
        if thinking_content:
            thinking_signature = generate_thinking_signature(thinking_content)
            content_blocks.append(
                ClaudeContentBlockThinking(
                    type="thinking",
                    thinking=thinking_content,
                    signature=thinking_signature,
                )
            )
            logger.debug(
                f"Added thinking content block: {len(thinking_content)} characters, signature: {thinking_signature}"
            )

        # Add text content if present
        if content_text:
            content_blocks.append(
                ClaudeContentBlockText(type=Constants.CONTENT_TEXT, text=content_text)
            )

        # Process tool calls
        if tool_calls:
            for tool_call in tool_calls:
                try:
                    arguments_dict = _parse_tool_arguments(tool_call.function.arguments)
                    content_blocks.append(
                        ClaudeContentBlockToolUse(
                            type=Constants.CONTENT_TOOL_USE,
                            id=tool_call.id,
                            name=tool_call.function.name,
                            input=arguments_dict,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error processing tool call: {e}")
                    continue

        # Only add empty content block if there are no tool calls (to avoid Claude Code loops)
        # if not content_blocks and not tool_calls:
        #     content_blocks.append(
        #         ClaudeContentBlockText(type=Constants.CONTENT_TEXT, text="")
        #     )

        # Map finish reason to Anthropic format
        if finish_reason == "length":
            stop_reason = Constants.STOP_MAX_TOKENS
        elif finish_reason == "tool_calls" or finish_reason is None and tool_calls:
            stop_reason = Constants.STOP_TOOL_USE
        else:
            stop_reason = Constants.STOP_END_TURN

        # Final debug info for Claude message creation
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug(
                f"Creating Claude response with {len(content_blocks)} content blocks:"
            )
            for i, block in enumerate(content_blocks):
                block_type = getattr(block, "type", "unknown")
                logger.debug(f"  Block {i}: {block_type}")
                if block_type == "thinking":
                    thinking_text = getattr(block, "thinking", "")
                    logger.debug(f"    Thinking length: {len(thinking_text)}")
                elif block_type == "text":
                    text = getattr(block, "text", "")
                    logger.debug(f"    Text: {repr(text[:200])}...")
                elif block_type == "tool_use":
                    name = getattr(block, "name", "unknown")
                    logger.debug(f"    Tool: {name}")

        # Extract comprehensive usage data from OpenAI response
        enhanced_usage = extract_usage_from_openai_response(openai_response)

        # Create Claude response
        claude_response = ClaudeMessagesResponse(
            id=response_id,
            model=original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=content_blocks,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=enhanced_usage,
        )

        # Compare response data and log any mismatches
        _compare_response_data(openai_response, claude_response)

        return claude_response

    except Exception as e:
        logger.error(f"Error converting response: {e}")
        return ClaudeMessagesResponse(
            id=f"msg_error_{uuid.uuid4()}",
            model=original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=[
                ClaudeContentBlockText(
                    type=Constants.CONTENT_TEXT, text="Response conversion error"
                )
            ],
            stop_reason=Constants.STOP_ERROR,
            usage=ClaudeUsage(input_tokens=0, output_tokens=0),
        )


# Helper function to clean schema for Gemini
def _remove_gemini_incompatible_uri_format(schema: dict) -> dict:
    """Fix only the specific URI format issue that causes MALFORMED_FUNCTION_CALL in Gemini."""
    # Create a copy to avoid modifying the original
    fixed_schema = {}

    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            # Fix properties recursively
            fixed_properties = {}
            for prop_name, prop_schema in value.items():
                if isinstance(prop_schema, dict):
                    fixed_prop = _remove_gemini_incompatible_uri_format(prop_schema)
                    fixed_properties[prop_name] = fixed_prop
                else:
                    fixed_properties[prop_name] = prop_schema
            fixed_schema[key] = fixed_properties
        elif key == "format" and value == "uri":
            # Skip the format field if it's 'uri' - this is the main issue
            logger.debug("Removing 'uri' format for Gemini compatibility")
            continue
        elif isinstance(value, dict):
            # Recursively fix nested objects
            fixed_schema[key] = _remove_gemini_incompatible_uri_format(value)
        elif isinstance(value, list):
            # Handle arrays
            fixed_list = []
            for item in value:
                if isinstance(item, dict):
                    fixed_list.append(_remove_gemini_incompatible_uri_format(item))
                else:
                    fixed_list.append(item)
            fixed_schema[key] = fixed_list
        else:
            # Keep everything else as-is
            fixed_schema[key] = value

    return fixed_schema


def clean_gemini_schema(schema: Any) -> Any:
    """
    Recursively clean and validate JSON schema for Gemini OpenAI API compatibility.

    Gemini requires a strict subset of OpenAPI schema format:
    - Function names: descriptive, no spaces/special chars (underscores/camelCase OK)
    - Parameters: must be object type with properties structure
    - Required: array of mandatory parameter names
    - Enum: use arrays for fixed value sets
    """
    if isinstance(schema, dict):
        # Create a copy to avoid modifying the original
        cleaned_schema = {}

        for key, value in schema.items():
            # Skip problematic fields that cause MALFORMED_FUNCTION_CALL
            if key in [
                "additionalProperties",  # Gemini doesn't support this
                "default",  # Can cause validation issues
                "examples",  # Not part of core OpenAPI subset
                "title",  # Not needed for function parameters
                "$schema",  # Schema metadata not needed
                "$id",  # Schema metadata not needed
                "definitions",  # Complex references not supported
                "$ref",  # References not supported
                "allOf",
                "anyOf",
                "oneOf",  # Complex schema combinations
                "not",  # Negation schemas not supported
                "patternProperties",  # Advanced property patterns
                "dependencies",  # Complex dependency validation
                "const",  # Use enum instead
                "contains",  # Array contains validation
                "propertyNames",  # Property name validation
                "if",
                "then",
                "else",  # Conditional schemas
                "readOnly",
                "writeOnly",  # OpenAPI 3.0 specific fields
                "deprecated",  # OpenAPI metadata
                "external_docs",  # OpenAPI documentation
                "xml",  # XML-specific metadata
                "example",  # Use examples array instead
            ]:
                logger.debug(
                    f"Removing unsupported field '{key}' for Gemini compatibility"
                )
                continue

            # Handle format field - only allow very basic formats
            if key == "format":
                # Gemini only supports very limited formats
                allowed_formats = {"date-time", "email", "uri"}  # Minimal set
                if value not in allowed_formats:
                    logger.debug(
                        f"Removing unsupported format '{value}' for Gemini compatibility"
                    )
                    continue
                # Even for allowed formats, be conservative and remove format
                # as it can cause validation issues
                logger.debug(
                    f"Removing format '{value}' for maximum Gemini compatibility"
                )
                continue

            # Handle string length constraints - be more lenient
            if key in ["minLength", "maxLength"] and isinstance(value, int):
                # Keep reasonable constraints but avoid extremes
                if key == "minLength" and value < 0:
                    continue
                if key == "maxLength" and value > 1000000:  # 1M chars max
                    cleaned_schema[key] = 1000000
                    continue

            # Handle numeric constraints
            if key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]:
                # Keep basic numeric constraints
                if isinstance(value, (int, float)):
                    cleaned_schema[key] = value
                continue

            # Handle pattern - remove complex regex patterns
            if key == "pattern":
                # Gemini may not support complex regex patterns
                logger.debug(f"Removing pattern '{value}' for Gemini compatibility")
                continue

            # Handle multipleOf - keep simple ones
            if key == "multipleOf" and isinstance(value, (int, float)) and value > 0:
                cleaned_schema[key] = value
                continue

            # Recursively clean nested schemas
            if isinstance(value, dict):
                cleaned_value = clean_gemini_schema(value)
                if cleaned_value:  # Only add if not empty
                    cleaned_schema[key] = cleaned_value
            elif isinstance(value, list):
                cleaned_value = [clean_gemini_schema(item) for item in value]
                # Filter out empty items
                cleaned_value = [
                    item for item in cleaned_value if item is not None and item != {}
                ]
                if cleaned_value:  # Only add if not empty
                    cleaned_schema[key] = cleaned_value
            else:
                # Keep primitive values
                cleaned_schema[key] = value

        return cleaned_schema
    elif isinstance(schema, list):
        # Recursively clean items in a list
        cleaned_list = [clean_gemini_schema(item) for item in schema]
        # Filter out empty items
        return [item for item in cleaned_list if item is not None and item != {}]

    return schema


def validate_gemini_function_schema(tool_def: dict) -> tuple[bool, str]:
    """
    Validate a function definition for Gemini OpenAI API compatibility.

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check basic structure
        if not isinstance(tool_def, dict):
            return False, "Tool definition must be a dictionary"

        if tool_def.get("type") != "function":
            return False, "Tool type must be 'function'"

        function_def = tool_def.get("function")
        if not isinstance(function_def, dict):
            return False, "Tool must have a 'function' object"

        # Validate function name
        name = function_def.get("name")
        if not name or not isinstance(name, str):
            return False, "Function must have a non-empty string name"

        # Check name format - no spaces, only alphanumeric, underscores, hyphens
        import re

        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", name):
            return (
                False,
                f"Function name '{name}' contains invalid characters. Use alphanumeric, underscores, or hyphens only",
            )

        # Validate description
        description = function_def.get("description", "")
        if not isinstance(description, str):
            return False, "Function description must be a string"

        if len(description.strip()) == 0:
            return False, "Function description cannot be empty (required for Gemini)"

        # Validate parameters schema
        parameters = function_def.get("parameters")
        if parameters is None:
            # Empty parameters are allowed
            return True, ""

        if not isinstance(parameters, dict):
            return False, "Parameters must be a dictionary"

        # For Gemini, parameters must be object type
        param_type = parameters.get("type")
        if param_type is None:
            return False, "Parameters must have a 'type' field"

        if param_type != "object":
            return False, f"Parameters type must be 'object', got '{param_type}'"

        # Validate properties if present
        properties = parameters.get("properties")
        if properties is not None:
            if not isinstance(properties, dict):
                return False, "Properties must be a dictionary"

            for prop_name, prop_schema in properties.items():
                if not isinstance(prop_name, str) or not prop_name.strip():
                    return (
                        False,
                        f"Property name must be a non-empty string, got '{prop_name}'",
                    )

                if not isinstance(prop_schema, dict):
                    return False, f"Property '{prop_name}' schema must be a dictionary"

                # Each property must have a type
                prop_type = prop_schema.get("type")
                if prop_type is None:
                    return False, f"Property '{prop_name}' must have a 'type' field"

                # Validate allowed types for Gemini
                allowed_types = {
                    "string",
                    "number",
                    "integer",
                    "boolean",
                    "array",
                    "object",
                }
                if prop_type not in allowed_types:
                    return (
                        False,
                        f"Property '{prop_name}' has unsupported type '{prop_type}'. Allowed: {allowed_types}",
                    )

                # Validate array items
                if prop_type == "array":
                    items = prop_schema.get("items")
                    if items is None:
                        return (
                            False,
                            f"Array property '{prop_name}' must have 'items' definition",
                        )

                    if isinstance(items, dict):
                        items_type = items.get("type")
                        if items_type is None:
                            return (
                                False,
                                f"Array property '{prop_name}' items must have a 'type' field",
                            )

                        if items_type not in allowed_types:
                            return (
                                False,
                                f"Array property '{prop_name}' items has unsupported type '{items_type}'",
                            )

        # Validate required array if present
        required = parameters.get("required")
        if required is not None:
            if not isinstance(required, list):
                return False, "Required must be an array"

            for req_field in required:
                if not isinstance(req_field, str) or not req_field.strip():
                    return (
                        False,
                        f"Required field must be a non-empty string, got '{req_field}'",
                    )

                # Check that required fields exist in properties
                if properties and req_field not in properties:
                    return (
                        False,
                        f"Required field '{req_field}' not found in properties",
                    )

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def _extract_tool_call_data(tool_call) -> tuple:
    """Extract tool call data from different formats."""
    if isinstance(tool_call, dict):
        tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
        function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
        name = function_data.get("name", "")
        arguments_str = function_data.get("arguments", "{}")
    elif hasattr(tool_call, "id") and hasattr(tool_call, Constants.TOOL_FUNCTION):
        tool_id = tool_call.id
        name = tool_call.function.name
        arguments_str = tool_call.function.arguments
    else:
        return None, None, None

    return tool_id, name, arguments_str


def _parse_tool_arguments(arguments_str: str) -> dict:
    """Parse tool arguments safely."""
    try:
        arguments_dict = json.loads(arguments_str)
        if not isinstance(arguments_dict, dict):
            arguments_dict = {"input": arguments_dict}
        return arguments_dict
    except json.JSONDecodeError:
        return {"raw_arguments": arguments_str}


def _parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"


def _send_message_start_event(message_id: str, model: str):
    """Send message_start event."""
    message_data = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }
    event_str = f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
    logger.debug(
        f"STREAMING_EVENT: message_start - message_id: {message_id}, model: {model}"
    )
    return event_str


def _send_content_block_start_event(index: int, block_type: str, **kwargs):
    """Send content_block_start event."""
    content_block = {"type": block_type, **kwargs}
    if block_type == "text":
        content_block["text"] = ""
    elif block_type == "tool_use":
        # Ensure tool_use blocks have required fields
        if "id" not in content_block:
            content_block["id"] = f"toolu_{uuid.uuid4().hex[:24]}"
        if "name" not in content_block:
            content_block["name"] = ""
        if "input" not in content_block:
            content_block["input"] = {}
    event_data = {
        "type": "content_block_start",
        "index": index,
        "content_block": content_block,
    }
    event_str = f"event: content_block_start\ndata: {json.dumps(event_data)}\n\n"
    logger.debug(
        f"STREAMING_EVENT: content_block_start - index: {index}, block_type: {block_type}, kwargs: {kwargs}"
    )
    return event_str


# see https://docs.anthropic.com/en/docs/build-with-claude/streaming#event-types
def _send_content_block_delta_event(index: int, delta_type: str, content: str):
    """Send content_block_delta event."""
    delta = {"type": delta_type}
    if delta_type == "text_delta":
        delta["text"] = content
    elif delta_type == "input_json_delta":
        delta["partial_json"] = content
    elif delta_type == "thinking_delta":
        delta["thinking"] = content
    elif delta_type == "signature_delta":
        delta["signature"] = content
    event_data = {"type": "content_block_delta", "index": index, "delta": delta}
    event_str = f"event: content_block_delta\ndata: {json.dumps(event_data)}\n\n"
    logger.debug(
        f"STREAMING_EVENT: content_block_delta - index: {index}, delta_type: {delta_type}, content_len: {len(content)}"
    )
    return event_str


def _send_content_block_stop_event(index: int):
    """Send content_block_stop event."""
    event_data = {"type": "content_block_stop", "index": index}
    event_str = f"event: content_block_stop\ndata: {json.dumps(event_data)}\n\n"
    logger.debug(f"STREAMING_EVENT: content_block_stop - index: {index}")
    return event_str


def _send_tool_use_delta_events(
    index: int, tool_id: str, tool_name: str, arguments: str
):
    """Send tool use delta events for a complete tool call."""
    # Send input JSON delta
    logger.debug(
        f"STREAMING_EVENT: tool_use deltas sent - index: {index}, tool: {tool_name}"
    )
    return _send_content_block_delta_event(index, "input_json_delta", arguments)


def _process_image_content_block(block, image_parts: list[dict]) -> None:
    """Process an image content block by adding it to image_parts."""
    if (
        isinstance(block.source, dict)
        and block.source.get("type") == "base64"
        and "media_type" in block.source
        and "data" in block.source
    ):
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                },
            }
        )


def _normalize_block(block):
    if isinstance(block, BaseModel):
        return block.model_dump(exclude_none=True)  # pydantic ≥v2
    return block


def _send_message_delta_event(
    stop_reason: str, output_tokens: int, content_blocks=None
):
    """Send message_delta event."""
    usage = {"output_tokens": output_tokens}
    delta = {"stop_reason": stop_reason, "stop_sequence": None}

    # Include content blocks in the delta if provided (required for proper Anthropic format)
    if content_blocks is not None:
        # Convert dict content blocks to proper Pydantic models
        converted_blocks = [_normalize_block(b) for b in content_blocks]
        delta["content"] = converted_blocks

    event_data = {"type": "message_delta", "delta": delta, "usage": usage}
    event_str = f"event: message_delta\ndata: {json.dumps(event_data)}\n\n"
    logger.debug(
        f"STREAMING_EVENT: message_delta - stop_reason: {stop_reason}, output_tokens: {output_tokens}, content_blocks_count: {len(content_blocks) if content_blocks else 0}"
    )
    return event_str


def _send_message_stop_event():
    """Send message_stop event."""
    event_data = {"type": "message_stop"}
    event_str = f"event: message_stop\ndata: {json.dumps(event_data)}\n\n"
    logger.debug("STREAMING_EVENT: message_stop")
    return event_str


def _send_ping_event():
    """Send ping event."""
    event_data = {"type": "ping"}
    event_str = f"event: ping\ndata: {json.dumps(event_data)}\n\n"
    logger.debug("STREAMING_EVENT: ping")
    return event_str


def _send_done_event():
    """Send [DONE] marker to terminate stream."""
    event_str = "data: [DONE]\n\n"
    logger.debug("STREAMING_EVENT: [DONE]")
    return event_str


def _map_finish_reason_to_stop_reason(finish_reason: str) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    if finish_reason == "length":
        return "max_tokens"
    elif finish_reason == "tool_calls":
        return "tool_use"
    else:
        return "end_turn"


# openai SSE -> claude SSE
async def convert_openai_streaming_response_to_anthropic(
    response_generator: AsyncStream[ChatCompletionChunk],
    original_request: ClaudeMessagesRequest,
    routed_model: str = None,
):
    """Handle streaming responses from OpenAI SDK and convert to Anthropic format."""
    has_sent_stop_reason = False
    is_tool_use = False
    input_tokens = 0
    output_tokens = 0
    current_content_blocks = []
    accumulated_text = ""
    accumulated_thinking = ""
    try:
        # Send initial events
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield _send_message_start_event(message_id, original_request.model)
        yield _send_ping_event()

        # State tracking variables
        content_block_index = 0
        text_block_started = False
        text_block_closed = False

        # Tool call state
        tool_json = ""
        current_tool_id = None
        current_tool_name = None

        # Thinking/reasoning state
        thinking_block_started = False
        thinking_block_closed = False

        # Streaming comparison tracking
        openai_chunks_received = 0

        logger.debug(f"Starting streaming for model: {original_request.model}")

        # Process each chunk
        async for chunk in response_generator:
            try:
                # Count OpenAI chunks received
                openai_chunks_received += 1

                # Detailed chunk logging
                chunk_data = {
                    "chunk_id": chunk.id,
                    "has_choices": len(chunk.choices) > 0,
                    "has_usage": chunk.usage,
                }

                if chunk_data["has_choices"]:
                    choice = chunk.choices[0]
                    chunk_data["choice_data"] = {
                        "has_delta": choice.delta is not None,
                        "finish_reason": choice.finish_reason,
                    }
                    if chunk_data["choice_data"]["has_delta"]:
                        delta = choice.delta
                        chunk_data["delta_data"] = {
                            "content": getattr(delta, "content", None),
                            "role": getattr(delta, "role", None),
                            "tool_calls": getattr(delta, "tool_calls", None)
                            is not None,
                        }
                        if chunk_data["delta_data"]["content"]:
                            chunk_data["delta_data"]["content_length"] = len(
                                chunk_data["delta_data"]["content"]
                            )

                if chunk_data["has_usage"]:
                    chunk_data["usage"] = {
                        "prompt_tokens": getattr(chunk.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(
                            chunk.usage, "completion_tokens", None
                        ),
                        "total_tokens": getattr(chunk.usage, "total_tokens", None),
                    }

                logger.debug(
                    f"STREAMING_CHUNK #{openai_chunks_received}: {json.dumps(chunk_data, default=str)}"
                )

                # Raw chunk debugging - print original chunk data when DEBUG_RAW_CHUNKS is set
                logger.debug(f"=== RAW CHUNK #{openai_chunks_received} ===")
                logger.debug(f"Raw chunk type: {type(chunk)}")
                logger.debug(
                    f"Raw chunk dict: {chunk.model_dump() if hasattr(chunk, 'model_dump') else str(chunk)}"
                )
                logger.debug("=== END RAW CHUNK ===")

                # Also print raw json representation if possible
                try:
                    if hasattr(chunk, "model_dump"):
                        raw_json = json.dumps(chunk.model_dump(), indent=2, default=str)
                        logger.debug(f"Raw chunk JSON:\n{raw_json}")
                except Exception as e:
                    logger.info(f"Could not serialize chunk to JSON: {e}")

                # Check if this is the end of the response with usage data
                if chunk.usage is not None:
                    # Extract input tokens from usage if available (for cost calculation)
                    reported_input_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                    reported_output_tokens = getattr(
                        chunk.usage, "completion_tokens", 0
                    )
                    logger.debug(
                        f"Reported usage - Input: {reported_input_tokens}, Output: {reported_output_tokens}"
                    )
                    # Update session statistics for streaming (using calculated tokens)
                    logger.debug(
                        f"Session stats - Input: {input_tokens}, Output: {output_tokens}"
                    )
                    logger.debug(
                        f"Content lengths - Text: {len(accumulated_text)}, Thinking: {len(accumulated_thinking)}"
                    )

                # Handle content from choices
                if len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    # Get the delta from the choice
                    delta = choice.delta

                    # Check for finish_reason to know when we're done
                    finish_reason = choice.finish_reason

                    # Handle tool calls first
                    delta_tool_calls = delta.tool_calls

                    if delta_tool_calls:
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]

                        for tool_call in delta_tool_calls:
                            # If we haven't started a tool yet, we need to handle any accumulated text first
                            if not is_tool_use:
                                # If we have accumulated text, send it first
                                if accumulated_text and not text_block_started:
                                    text_block = {"type": "text", "text": ""}
                                    current_content_blocks.append(text_block)
                                    yield _send_content_block_start_event(
                                        content_block_index, "text"
                                    )
                                    text_block_started = True

                                    # Send the accumulated text
                                    yield _send_content_block_delta_event(
                                        content_block_index,
                                        "text_delta",
                                        accumulated_text,
                                    )

                                    # Update the content block
                                    current_content_blocks[content_block_index][
                                        "text"
                                    ] = accumulated_text

                                    # Close the text block
                                    yield _send_content_block_stop_event(
                                        content_block_index
                                    )
                                    text_block_closed = True
                                    content_block_index += 1
                                elif text_block_started and not text_block_closed:
                                    # Close any open text block
                                    yield _send_content_block_stop_event(
                                        content_block_index
                                    )
                                    text_block_closed = True
                                    content_block_index += 1
                                elif not text_block_started and not text_block_closed:
                                    # Even if no text, we might need to close the implicit text block
                                    text_block = {"type": "text", "text": ""}
                                    current_content_blocks.append(text_block)
                                    yield _send_content_block_start_event(
                                        content_block_index, "text"
                                    )
                                    yield _send_content_block_stop_event(
                                        content_block_index
                                    )
                                    text_block_closed = True
                                    content_block_index += 1

                                # Now start the tool use block
                                is_tool_use = True

                                # Extract tool info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get("function", {})
                                    current_tool_name = (
                                        function.get("name", "")
                                        if isinstance(function, dict)
                                        else ""
                                    )
                                    current_tool_id = tool_call.get(
                                        "id", f"toolu_{uuid.uuid4().hex[:24]}"
                                    )
                                else:
                                    function = getattr(tool_call, "function", None)
                                    current_tool_name = (
                                        getattr(function, "name", "")
                                        if function
                                        else ""
                                    )
                                    current_tool_id = getattr(
                                        tool_call,
                                        "id",
                                        f"toolu_{uuid.uuid4().hex[:24]}",
                                    )

                                # Create tool use block
                                tool_block = {
                                    "type": "tool_use",
                                    "id": current_tool_id,
                                    "name": current_tool_name,
                                    "input": {},
                                }
                                current_content_blocks.append(tool_block)

                                yield _send_content_block_start_event(
                                    content_block_index,
                                    "tool_use",
                                    id=current_tool_id,
                                    name=current_tool_name,
                                )
                                tool_json = ""

                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and "function" in tool_call:
                                function = tool_call.get("function", {})
                                arguments = (
                                    function.get("arguments", "")
                                    if isinstance(function, dict)
                                    else ""
                                )
                            elif hasattr(tool_call, "function"):
                                function = getattr(tool_call, "function", None)
                                arguments = (
                                    getattr(function, "arguments", "")
                                    if function
                                    else ""
                                )

                            # If we have arguments, send them as a delta
                            if arguments:
                                tool_json += arguments

                                # Try to parse JSON to update the content block
                                try:
                                    parsed_json = json.loads(tool_json)
                                    current_content_blocks[content_block_index][
                                        "input"
                                    ] = parsed_json
                                except json.JSONDecodeError:
                                    # JSON not yet complete, continue accumulating
                                    pass

                                # Send the delta
                                yield _send_content_block_delta_event(
                                    content_block_index, "input_json_delta", arguments
                                )

                    # Handle reasoning/thinking content first (from deepseek reasoning models)
                    delta_reasoning = None
                    raw_delta = delta.model_dump()
                    if isinstance(raw_delta, dict) and "reasoning_content" in raw_delta:
                        delta_reasoning = raw_delta["reasoning_content"]

                    if delta_reasoning is not None and delta_reasoning != "":
                        accumulated_thinking += delta_reasoning
                        logger.debug(
                            f"Added thinking content: +{len(delta_reasoning)} chars, total: {len(accumulated_thinking)} chars"
                        )

                        # Start thinking block if not started
                        if not thinking_block_started:
                            thinking_block = {"type": "thinking", "thinking": ""}
                            current_content_blocks.append(thinking_block)
                            yield _send_content_block_start_event(
                                content_block_index, "thinking"
                            )
                            thinking_block_started = True

                        # Send thinking delta
                        yield _send_content_block_delta_event(
                            content_block_index, "thinking_delta", delta_reasoning
                        )

                        # Update content block
                        if content_block_index < len(current_content_blocks):
                            current_content_blocks[content_block_index]["thinking"] = (
                                accumulated_thinking
                            )

                    # Check if reasoning is complete (no more reasoning deltas and we have normal content)
                    elif thinking_block_started and not thinking_block_closed:
                        # If we have normal content coming and thinking was active, close thinking block
                        delta_content = None
                        if hasattr(delta, "content"):
                            delta_content = delta.content
                        elif isinstance(delta, dict) and "content" in delta:
                            delta_content = delta["content"]

                        if delta_content is not None and delta_content != "":
                            # Parse function calls from thinking content before closing
                            cleaned_thinking, function_calls = (
                                parse_function_calls_from_thinking(accumulated_thinking)
                            )

                            # Update thinking content with cleaned version (function calls removed)
                            if content_block_index < len(current_content_blocks):
                                current_content_blocks[content_block_index][
                                    "thinking"
                                ] = cleaned_thinking

                                # Generate signature for cleaned thinking content
                                thinking_signature = generate_thinking_signature(
                                    cleaned_thinking
                                )
                                current_content_blocks[content_block_index][
                                    "signature"
                                ] = thinking_signature

                                # Send signature delta before closing thinking block
                                yield _send_content_block_delta_event(
                                    content_block_index,
                                    "signature_delta",
                                    thinking_signature,
                                )

                            yield _send_content_block_stop_event(content_block_index)
                            thinking_block_closed = True
                            content_block_index += 1

                            # Add function call blocks if any were found
                            for tool_call in function_calls:
                                logger.debug(
                                    f"Adding tool call block: {tool_call['function']['name']}"
                                )

                                # Create tool use content block
                                tool_block = {
                                    "type": "tool_use",
                                    "id": tool_call["id"],
                                    "name": tool_call["function"]["name"],
                                    "input": json.loads(
                                        tool_call["function"]["arguments"]
                                    ),
                                }
                                current_content_blocks.append(tool_block)

                                # Send tool call events
                                yield _send_content_block_start_event(
                                    content_block_index,
                                    "tool_use",
                                    id=tool_call["id"],
                                    name=tool_call["function"]["name"],
                                )
                                yield _send_tool_use_delta_events(
                                    content_block_index,
                                    tool_call["id"],
                                    tool_call["function"]["name"],
                                    tool_call["function"]["arguments"],
                                )
                                yield _send_content_block_stop_event(
                                    content_block_index
                                )
                                content_block_index += 1

                    # Handle text content - check for text content first
                    delta_content = delta.content

                    # If we have text content and we're currently in tool use mode, end the tool use first
                    if (
                        delta_content is not None
                        and delta_content != ""
                        and is_tool_use
                    ):
                        # End current tool call block
                        yield _send_content_block_stop_event(content_block_index)
                        content_block_index += 1
                        is_tool_use = False
                        # Reset tool state
                        tool_json = ""
                        current_tool_id = None
                        current_tool_name = None

                    # Handle text content
                    if (
                        delta_content is not None
                        and delta_content != ""
                        and not is_tool_use
                    ):
                        accumulated_text += delta_content
                        # Calculate current output tokens in real-time using tiktoken
                        current_output_tokens = count_tokens_in_response(
                            response_content=accumulated_text,
                            thinking_content=accumulated_thinking,
                            tool_calls=[],
                        )
                        logger.debug(
                            f"Added text content: +{len(delta_content)} chars, total: {len(accumulated_text)} chars, tokens: {current_output_tokens}"
                        )

                        # Start text block if not started
                        if not text_block_started:
                            text_block = {"type": "text", "text": ""}
                            current_content_blocks.append(text_block)
                            yield _send_content_block_start_event(
                                content_block_index, "text"
                            )
                            text_block_started = True

                        # Send text delta
                        yield _send_content_block_delta_event(
                            content_block_index, "text_delta", delta_content
                        )

                        # Update content block
                        if content_block_index < len(current_content_blocks):
                            current_content_blocks[content_block_index]["text"] = (
                                accumulated_text
                            )

                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        logger.debug(f"Processing finish_reason: {finish_reason}")
                        has_sent_stop_reason = True

                        # Close thinking block if it was started
                        if thinking_block_started and not thinking_block_closed:
                            # Parse function calls from thinking content before closing
                            cleaned_thinking, function_calls = (
                                parse_function_calls_from_thinking(accumulated_thinking)
                            )

                            # Update thinking content with cleaned version (function calls removed)
                            if content_block_index < len(current_content_blocks):
                                current_content_blocks[content_block_index][
                                    "thinking"
                                ] = cleaned_thinking

                                # Generate signature for cleaned thinking content
                                thinking_signature = generate_thinking_signature(
                                    cleaned_thinking
                                )
                                current_content_blocks[content_block_index][
                                    "signature"
                                ] = thinking_signature

                                # Send signature delta before closing thinking block
                                yield _send_content_block_delta_event(
                                    content_block_index,
                                    "signature_delta",
                                    thinking_signature,
                                )

                            yield _send_content_block_stop_event(content_block_index)
                            thinking_block_closed = True
                            content_block_index += 1

                            # Add function call blocks if any were found
                            for tool_call in function_calls:
                                logger.debug(
                                    f"Adding tool call block: {tool_call['function']['name']}"
                                )

                                # Create tool use content block
                                tool_block = {
                                    "type": "tool_use",
                                    "id": tool_call["id"],
                                    "name": tool_call["function"]["name"],
                                    "input": json.loads(
                                        tool_call["function"]["arguments"]
                                    ),
                                }
                                current_content_blocks.append(tool_block)

                                # Send tool call events
                                yield _send_content_block_start_event(
                                    content_block_index,
                                    "tool_use",
                                    id=tool_call["id"],
                                    name=tool_call["function"]["name"],
                                )
                                yield _send_tool_use_delta_events(
                                    content_block_index,
                                    tool_call["id"],
                                    tool_call["function"]["name"],
                                    tool_call["function"]["arguments"],
                                )
                                yield _send_content_block_stop_event(
                                    content_block_index
                                )
                                content_block_index += 1

                        # If we haven't started any blocks yet, start and immediately close a text block
                        if (
                            not text_block_started
                            and not is_tool_use
                            and not thinking_block_started
                        ):
                            text_block = {"type": "text", "text": ""}
                            current_content_blocks.append(text_block)
                            yield _send_content_block_start_event(
                                content_block_index, "text"
                            )
                            yield _send_content_block_stop_event(content_block_index)
                        elif text_block_started or is_tool_use:
                            # Close the current content block
                            yield _send_content_block_stop_event(content_block_index)

                        # Determine stop reason
                        stop_reason = _map_finish_reason_to_stop_reason(finish_reason)
                        logger.debug(f"Mapped stop_reason: {stop_reason}")

                        # Calculate accurate output tokens from accumulated content
                        final_output_tokens = _calculate_accurate_output_tokens(
                            accumulated_text,
                            accumulated_thinking,
                            output_tokens,
                            "Finish reason received",
                        )

                        # Send message delta with final content and stop reason
                        yield _send_message_delta_event(
                            stop_reason, final_output_tokens, current_content_blocks
                        )

                        # Send message stop
                        yield _send_message_stop_event()
                        yield _send_done_event()
                        logger.debug("Streaming completed successfully")
                        has_sent_stop_reason = True
                        return

            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue

            # Validate content integrity after processing each chunk
            if (
                openai_chunks_received % 10 == 0
            ):  # Every 10 chunks to avoid too much logging
                validation = _validate_streaming_content_integrity(
                    accumulated_text,
                    accumulated_thinking,
                    current_content_blocks,
                    f"After chunk #{openai_chunks_received}",
                )
                if validation["has_issues"]:
                    logger.warning(f"Content integrity issues: {validation}")
                else:
                    logger.debug(f"Content validation passed: {validation}")

        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            logger.debug("No finish_reason received, closing stream manually")

            # Close thinking block if it was started
            if thinking_block_started and not thinking_block_closed:
                # Parse function calls from thinking content before closing
                cleaned_thinking, function_calls = parse_function_calls_from_thinking(
                    accumulated_thinking
                )

                # Update thinking content with cleaned version (function calls removed)
                if content_block_index < len(current_content_blocks):
                    current_content_blocks[content_block_index]["thinking"] = (
                        cleaned_thinking
                    )

                    # Generate signature for cleaned thinking content
                    thinking_signature = generate_thinking_signature(cleaned_thinking)
                    current_content_blocks[content_block_index]["signature"] = (
                        thinking_signature
                    )

                    # Send signature delta before closing thinking block
                    yield _send_content_block_delta_event(
                        content_block_index,
                        "signature_delta",
                        thinking_signature,
                    )

                yield _send_content_block_stop_event(content_block_index)
                thinking_block_closed = True
                content_block_index += 1

                # Add function call blocks if any were found
                for tool_call in function_calls:
                    logger.debug(
                        f"Adding tool call block: {tool_call['function']['name']}"
                    )

                    # Create tool use content block
                    tool_block = {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": json.loads(tool_call["function"]["arguments"]),
                    }
                    current_content_blocks.append(tool_block)

                    # Send tool call events
                    yield _send_content_block_start_event(
                        content_block_index,
                        "tool_use",
                        id=tool_call["id"],
                        name=tool_call["function"]["name"],
                    )
                    yield _send_tool_use_delta_events(
                        content_block_index,
                        tool_call["id"],
                        tool_call["function"]["name"],
                        tool_call["function"]["arguments"],
                    )
                    yield _send_content_block_stop_event(content_block_index)
                    content_block_index += 1

            # If we haven't started any blocks yet, start and immediately close a text block
            if (
                not text_block_started
                and not is_tool_use
                and not thinking_block_started
            ):
                text_block = {"type": "text", "text": ""}
                current_content_blocks.append(text_block)
                yield _send_content_block_start_event(content_block_index, "text")
                yield _send_content_block_stop_event(content_block_index)
            elif text_block_started or is_tool_use:
                # Close the current content block if there is one
                yield _send_content_block_stop_event(content_block_index)

            # Calculate accurate output tokens from accumulated content
            final_output_tokens = _calculate_accurate_output_tokens(
                accumulated_text,
                accumulated_thinking,
                output_tokens,
                "No finish_reason",
            )

            # Send final events
            stop_reason = "tool_use" if is_tool_use else "end_turn"
            yield _send_message_delta_event(
                stop_reason, final_output_tokens, current_content_blocks
            )
            yield _send_message_stop_event()
            yield _send_done_event()
            logger.debug("Streaming completed without finish_reason")

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = (
            f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        )
        logger.error(error_message)

        # Send error events
        logger.error(f"Streaming error: {error_message}")
        yield _send_message_delta_event("error", 0, [])
        yield _send_message_stop_event()
        yield _send_done_event()

    finally:
        if not has_sent_stop_reason:
            # Close thinking block if it was started (fallback in finally block)
            if thinking_block_started and not thinking_block_closed:
                logger.debug("Finally block: closing unclosed thinking block")
                # Parse function calls from thinking content before closing
                cleaned_thinking, function_calls = parse_function_calls_from_thinking(
                    accumulated_thinking
                )

                # Update thinking content with cleaned version (function calls removed)
                if content_block_index < len(current_content_blocks):
                    current_content_blocks[content_block_index]["thinking"] = (
                        cleaned_thinking
                    )

                    # Generate signature for cleaned thinking content
                    thinking_signature = generate_thinking_signature(cleaned_thinking)
                    current_content_blocks[content_block_index]["signature"] = (
                        thinking_signature
                    )

                # Add function call blocks if any were found
                for tool_call in function_calls:
                    logger.debug(
                        f"Finally block: Adding tool call block: {tool_call['function']['name']}"
                    )

                    # Create tool use content block
                    tool_block = {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["function"]["name"],
                        "input": json.loads(tool_call["function"]["arguments"]),
                    }
                    current_content_blocks.append(tool_block)

                thinking_block_closed = True

            # Calculate accurate output tokens from accumulated content
            final_output_tokens = _calculate_accurate_output_tokens(
                accumulated_text, accumulated_thinking, output_tokens, "Finally block"
            )

            stop_reason = "tool_use" if is_tool_use else "end_turn"
            yield _send_message_delta_event(
                stop_reason, final_output_tokens, current_content_blocks
            )
            yield _send_message_stop_event()
            yield _send_done_event()
            logger.debug(
                "Streaming finally: emitted missing message_delta + message_stop"
            )
        # Final content integrity validation
        final_validation = _validate_streaming_content_integrity(
            accumulated_text,
            accumulated_thinking,
            current_content_blocks,
            "Final streaming validation",
        )
        if final_validation["has_issues"]:
            logger.error(f"FINAL CONTENT INTEGRITY ISSUES: {final_validation}")
        else:
            logger.info(f"Final content validation passed: {final_validation}")

        # DEBUG: Rebuild and log complete response for debugging streaming issues
        try:
            final_stop_reason = "tool_use" if is_tool_use else "end_turn"
            final_output_tokens = _calculate_accurate_output_tokens(
                accumulated_text,
                accumulated_thinking,
                output_tokens,
                "Final debug output",
            )

            complete_response = _rebuild_complete_response_from_streaming(
                accumulated_text=accumulated_text,
                accumulated_thinking=accumulated_thinking,
                current_content_blocks=current_content_blocks,
                output_tokens=final_output_tokens,
                model=original_request.model,
                stop_reason=final_stop_reason,
            )

            logger.info("=== STREAMING COMPLETE RESPONSE DEBUG ===")
            logger.info(f"Original request model: {original_request.model}")
            logger.info(f"Accumulated text length: {len(accumulated_text)}")
            logger.info(f"Accumulated thinking length: {len(accumulated_thinking)}")
            logger.info(f"Current content blocks count: {len(current_content_blocks)}")
            logger.info(f"Is tool use: {is_tool_use}")
            logger.info(f"Final stop reason: {final_stop_reason}")
            logger.info("Complete rebuilt response:")
            logger.info(json.dumps(complete_response, indent=2, ensure_ascii=False))
            logger.info("=== END STREAMING RESPONSE DEBUG ===")

        except Exception as debug_error:
            logger.error(f"Error in streaming response debug: {debug_error}")

        # Track usage statistics for streaming responses
        try:
            # Calculate final tokens for usage tracking
            final_output_tokens = _calculate_accurate_output_tokens(
                accumulated_text,
                accumulated_thinking,
                output_tokens,
                "Streaming usage tracking",
            )

            # Create usage object from streaming data
            # Note: For streaming, we typically don't have detailed usage from the API response
            # so we create a basic usage object with calculated tokens
            streaming_usage = ClaudeUsage(
                input_tokens=input_tokens if input_tokens > 0 else 0,
                output_tokens=final_output_tokens,
                prompt_tokens=input_tokens if input_tokens > 0 else 0,
                completion_tokens=final_output_tokens,
                total_tokens=(input_tokens if input_tokens > 0 else 0)
                + final_output_tokens,
            )

            # Add reasoning tokens if we detected thinking content
            if accumulated_thinking:
                if not streaming_usage.completion_tokens_details:
                    streaming_usage.completion_tokens_details = (
                        CompletionTokensDetails()
                    )

                reasoning_tokens = count_tokens_in_response(
                    thinking_content=accumulated_thinking
                )
                streaming_usage.completion_tokens_details.reasoning_tokens = (
                    reasoning_tokens
                )

            # Update global usage statistics
            model_name = routed_model if routed_model else original_request.model
            update_global_usage_stats(streaming_usage, model_name, "Streaming")

        except Exception as usage_error:
            logger.error(f"Error tracking streaming usage: {usage_error}")


def _validate_streaming_content_integrity(
    accumulated_text: str,
    accumulated_thinking: str,
    current_content_blocks: list[dict[str, Any]],
    context: str = "",
) -> dict[str, Any]:
    """
    Validate that streaming content is being properly preserved.
    Returns validation results for logging.
    """
    validation_results = {
        "has_text_content": bool(accumulated_text.strip()),
        "has_thinking_content": bool(accumulated_thinking.strip()),
        "has_tool_calls": any(
            block.get("type") == "tool_use" for block in current_content_blocks
        ),
        "total_content_blocks": len(current_content_blocks),
        "text_length": len(accumulated_text),
        "thinking_length": len(accumulated_thinking),
        "context": context,
    }

    # Check for potential issues
    issues = []
    if (
        not validation_results["has_text_content"]
        and not validation_results["has_tool_calls"]
    ):
        issues.append("No text content or tool calls found")

    if validation_results["total_content_blocks"] == 0:
        issues.append("No content blocks created")

    validation_results["issues"] = issues
    validation_results["has_issues"] = len(issues) > 0

    return validation_results


def _rebuild_complete_response_from_streaming(
    accumulated_text: str,
    accumulated_thinking: str,
    current_content_blocks: list[dict[str, Any]],
    output_tokens: int,
    model: str,
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    """
    Rebuild a complete Anthropic-format response from streaming data.
    This helps debug streaming issues by showing what the final response looks like.
    """
    # Build content blocks array similar to non-streaming responses
    content_blocks = []

    # Add thinking block if present
    if accumulated_thinking.strip():
        content_blocks.append({"type": "thinking", "thinking": accumulated_thinking})

    # Add text block if present
    if accumulated_text.strip():
        content_blocks.append({"type": "text", "text": accumulated_text})

    # Add any tool use blocks from current_content_blocks
    for block in current_content_blocks:
        if block.get("type") == "tool_use":
            content_blocks.append(block)

    # Build complete response structure
    complete_response = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"output_tokens": output_tokens},
    }

    return complete_response


def _calculate_accurate_output_tokens(
    accumulated_text: str,
    accumulated_thinking: str,
    reported_tokens: int,
    context: str = "",
) -> int:
    """Calculate accurate output tokens using tiktoken (always use calculated, not reported)."""
    calculated_tokens = count_tokens_in_response(
        response_content=accumulated_text,
        thinking_content=accumulated_thinking,
        tool_calls=[],  # Tool calls handled separately in streaming
    )

    # Always use calculated tokens since reported tokens are often unreliable (0)
    # for third-party models accessed through proxy APIs
    final_tokens = calculated_tokens

    logger.debug(
        f"{context} - Token comparison - Reported: {reported_tokens}, Calculated: {calculated_tokens}, Using: {final_tokens} (always using calculated)"
    )
    return final_tokens


def count_tokens_in_response(
    response_content: str = "", thinking_content: str = "", tool_calls: list = []
) -> int:
    """Count tokens in response content using tiktoken"""
    if enc is None:
        logger.warning(
            "TikToken encoder not available for response counting, using approximate count"
        )
        return 0

    total_tokens = 0

    try:
        # Count main response content tokens
        if response_content:
            total_tokens += len(enc.encode(response_content))

        # Count thinking content tokens
        if thinking_content:
            total_tokens += len(enc.encode(thinking_content))

        # Count tool calls tokens
        if tool_calls:
            for tool_call in tool_calls:
                name = getattr(tool_call.function, "name", "")
                arguments = getattr(tool_call.function, "arguments", "")
                total_tokens += len(enc.encode(name + arguments))

    except Exception as e:
        logger.error(f"Error counting response tokens: {e}")
        # Fallback estimation
        return 0

    return int(total_tokens)



def _compare_request_data(
    claude_request: ClaudeMessagesRequest, openai_request: dict[str, Any]
) -> None:
    """Compare original Claude request with converted OpenAI request and log differences."""
    try:
        # Count Claude request data
        claude_tools_count = len(claude_request.tools) if claude_request.tools else 0
        claude_messages_count = len(claude_request.messages)
        claude_has_tool_choice = claude_request.tool_choice is not None
        claude_has_system = claude_request.system is not None
        claude_has_tool_result = False
        for msg in claude_request.messages:
            if isinstance(msg.content, list):
                for content in msg.content:
                    if isinstance(content, ClaudeContentBlockToolResult):
                        claude_has_tool_result = True
                        break
            if claude_has_tool_result:
                break

        # Count OpenAI request data
        openai_tools_count = len(openai_request.get("tools", []))
        openai_messages_count = len(openai_request.get("messages", []))
        openai_has_tool_choice = "tool_choice" in openai_request
        openai_has_system = any(
            msg.get("role") == "system" for msg in openai_request.get("messages", [])
        )
        openai_has_tool_msg = any(
            msg.get("role") == "tool" for msg in openai_request.get("messages", [])
        )

        # Log conversion summary
        logger.debug("CONVERSION SUMMARY:")
        logger.debug(
            f"  Tools: {claude_tools_count} -> {openai_tools_count}"
        )
        logger.debug(
            f"  Messages: {claude_messages_count} -> {openai_messages_count}"
        )
        logger.debug(
            f"  Tool Choice: {claude_has_tool_choice} -> {openai_has_tool_choice}"
        )
        logger.debug(
            f"  System Message: {claude_has_system} -> {openai_has_system}"
        )
        logger.debug(
            f"  Tool Results: {claude_has_tool_result} -> {openai_has_tool_msg}"
        )

        # Check for unexpected conversion issues
        warnings = []

        # Tools count should always match exactly
        if claude_tools_count != openai_tools_count:
            warnings.append(
                f"Unexpected tools count difference: {claude_tools_count} -> {openai_tools_count}"
            )

        # Tool choice should always match exactly
        if claude_has_tool_choice != openai_has_tool_choice:
            warnings.append(
                f"Unexpected tool choice difference: {claude_has_tool_choice} -> {openai_has_tool_choice}"
            )

        # Message count differences are expected in certain cases:
        # - System messages get extracted to separate OpenAI system message
        # - Tool results get converted to separate tool role messages
        # Only warn if message count differs in unexpected scenarios
        if (
            claude_messages_count != openai_messages_count
            and not claude_has_system  # System message extraction expected
            and not claude_has_tool_result  # Tool result splitting expected
        ):
            warnings.append(
                f"Unexpected message count difference: {claude_messages_count} -> {openai_messages_count} (no system/tool_result expected)"
            )

        if warnings:
            logger.warning(f"CONVERSION ISSUES: {'; '.join(warnings)}")
        else:
            logger.debug("CONVERSION: All transformations completed successfully ✓")

    except Exception as e:
        logger.error(f"Error during conversion validation: {e}")


def _compare_response_data(
    openai_response, claude_response: ClaudeMessagesResponse
) -> None:
    """Compare OpenAI response with converted Claude response and log differences."""
    try:
        # Count OpenAI response data
        openai_content_blocks = 0
        openai_tool_calls = 0
        openai_finish_reason = None

        if hasattr(openai_response, "choices") and openai_response.choices:
            choice = openai_response.choices[0]
            if hasattr(choice, "message") and choice.message:
                if hasattr(choice.message, "content") and choice.message.content:
                    openai_content_blocks = 1  # OpenAI has single content field
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    openai_tool_calls = len(choice.message.tool_calls)
            if hasattr(choice, "finish_reason"):
                openai_finish_reason = choice.finish_reason

        # Count Claude response data
        claude_content_blocks = (
            len(claude_response.content) if claude_response.content else 0
        )
        claude_tool_use_blocks = (
            sum(
                1
                for block in claude_response.content
                if hasattr(block, "type") and block.type == "tool_use"
            )
            if claude_response.content
            else 0
        )
        claude_stop_reason = claude_response.stop_reason

        # Log comparison
        logger.info("RESPONSE CONVERSION COMPARISON:")
        logger.info(
            f"  OpenAI -> Claude Content Blocks: {openai_content_blocks} -> {claude_content_blocks}"
        )
        logger.info(
            f"  OpenAI -> Claude Tool Calls/Use: {openai_tool_calls} -> {claude_tool_use_blocks}"
        )
        logger.info(
            f"  OpenAI -> Claude Finish/Stop Reason: {openai_finish_reason} -> {claude_stop_reason}"
        )

        # Check for mismatches and log errors
        errors = []
        if openai_tool_calls != claude_tool_use_blocks:
            errors.append(
                f"Tool use count mismatch: OpenAI({openai_tool_calls}) != Claude({claude_tool_use_blocks})"
            )

        if errors:
            logger.error(f"RESPONSE CONVERSION ERRORS: {'; '.join(errors)}")
        else:
            logger.info("RESPONSE CONVERSION: Key counts match ✓")

    except Exception as e:
        logger.error(f"Error in response comparison: {e}")
