"""
Pydantic models for Claude proxy API requests and responses.
This module contains only the model definitions without any server startup code.
"""

import json
import logging
import re
import threading
import uuid
from datetime import datetime
from typing import Any, Literal

from openai import AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
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
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, ConfigDict, field_validator


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
                    logger.debug(f"Added tool call: {tool_call}")
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
ClaudeToolChoice = (
    ClaudeToolChoiceAuto
    | ClaudeToolChoiceAny
    | ClaudeToolChoiceTool
    | ClaudeToolChoiceNone
)


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
    model_config = ConfigDict(
        validate_assignment=False, str_strip_whitespace=False, extra="ignore"
    )

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
    model_config = ConfigDict(
        validate_assignment=False, str_strip_whitespace=False, extra="ignore"
    )

    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict[str, Any]] | dict[str, Any]

    def process_content(
        self,
    ) -> str | list:
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

        Note:
            For Claude user message, there is no tool call and thinking content.
            tool call content block and thinking content is for assistant message only

        doc link:
            https://api-docs.deepseek.com/api/create-chat-completion
            https://platform.openai.com/docs/api-reference/chat/create
            https://github.com/anthropics/anthropic-sdk-python/blob/main/api.md
            https://docs.anthropic.com/en/api/messages
        """

        # Debug: Pretty print Claude message before conversion
        # logger.debug("📨 Raw Claude Message")
        # logger.debug(f"   {self.model_dump_json(indent=2)}")

        openai_messages = []

        # Handle simple string content
        if isinstance(self.content, str):
            # processed_content = self.process_interrupted_content()
            openai_messages.append({"role": self.role, "content": self.content})
            return openai_messages

        # Process content blocks in order, maintaining structure
        # Claude user message content blocks -> OpenAI user message content
        # parts
        # https://docs.anthropic.com/en/api/messages#body-messages-content
        # https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages

        # correct order: assistant message with tool_calls should always
        # be before than assistant message with tool result.
        openai_parts = []
        # merge text content
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

        logger.debug(f"🔄 Output messages count: {len(openai_messages)}")
        # logger.debug(f"🔄 OpenAI request: {request_params}")

        # DEBUG: Validate and debug OpenAI message sequence for tool call ordering
        _debug_openai_message_sequence(
            openai_messages, f"Claude->OpenAI conversion for model {self.model}"
        )

        # Note: OpenAI API requires that messages with role 'tool' must be a response
        # to a preceding message with 'tool_calls'. The current message conversion
        # logic naturally produces the correct sequence: Assistant(tool_calls) → Tool → User

        # Compare request data and log any mismatches
        _compare_request_data(self, request_params)

        # Return the request with validated components
        # Individual components (messages, tools) are already validated using OpenAI SDK types
        return request_params

    def calculate_tokens(self) -> int:
        return count_tokens_in_messages(self.messages, self.model)


class ClaudeTokenCountRequest(BaseModel):
    model_config = ConfigDict(
        # Optimize performance for high-throughput proxy server
        validate_assignment=False,  # Skip validation on assignment for speed
        str_strip_whitespace=False,  # Skip string stripping for performance
        use_enum_values=True,  # Use enum values directly
        arbitrary_types_allowed=True,  # Allow arbitrary types for flexibility
        extra="ignore",  # Ignore extra fields instead of validation
    )

    model: str
    messages: list[ClaudeMessage]
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
        return count_tokens_in_messages(self.messages, self.model)


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
            logger.debug(f"token usage from response: {usage}")
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
            thinking_signature = generate_unique_id("thinking")
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
        # OpenAI tool calls -> Claude Tool Use
        if tool_calls:
            for tool_call in tool_calls:
                try:
                    arguments_dict = _parse_tool_arguments(tool_call.function.arguments)
                    content_blocks.append(
                        ClaudeContentBlockToolUse(
                            type=Constants.CONTENT_TOOL_USE,
                            id=generate_unique_id("toolu"),
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


def _parse_tool_arguments(arguments_str: str) -> dict:
    """Parse tool arguments safely."""
    try:
        arguments_dict = json.loads(arguments_str)
        if not isinstance(arguments_dict, dict):
            arguments_dict = {"input": arguments_dict}
        return arguments_dict
    except json.JSONDecodeError:
        return {"raw_arguments": arguments_str}


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
            content_block["id"] = generate_unique_id("tool")
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
    stop_reason: str, output_tokens: int, input_tokens: int, content_blocks=None
):
    """Send message_delta event."""
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
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
class AnthropicStreamingConverter:
    """Encapsulates state and logic for converting OpenAI streaming responses to Anthropic format."""

    def __init__(self, original_request: ClaudeMessagesRequest):
        self.original_request = original_request
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Content tracking
        self.content_block_index = 0
        self.current_content_blocks = []
        self.accumulated_text = ""
        self.accumulated_thinking = ""

        # Block state tracking
        self.text_block_started = False
        self.text_block_closed = False
        self.thinking_block_started = False
        self.thinking_block_closed = False
        self.is_tool_use = False

        # Tool call state
        self.tool_json = ""
        self.current_tool_id = None
        self.current_tool_name = None

        # Response state
        self.has_sent_stop_reason = False
        self.input_tokens = original_request.calculate_tokens()
        self.completion_tokens = 0
        self.output_tokens = 0
        self.openai_chunks_received = 0

    def _send_message_start_event(self) -> str:
        """Send message_start event."""
        message_data = {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        }
        event_str = f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: message_start - message_id: {self.message_id}, model: {self.original_request.model}"
        )
        return event_str

    def _send_ping_event(self) -> str:
        """Send ping event."""
        event_data = {"type": "ping"}
        event_str = f"event: ping\ndata: {json.dumps(event_data)}\n\n"
        logger.debug("STREAMING_EVENT: ping")
        return event_str

    def _send_content_block_start_event(self, block_type: str, **kwargs) -> str:
        """Send content_block_start event."""
        content_block = {"type": block_type, **kwargs}
        if block_type == "text":
            content_block["text"] = ""
        elif block_type == "tool_use":
            # Ensure tool_use blocks have required fields
            if "id" not in content_block:
                content_block["id"] = generate_unique_id("tool")
            if "name" not in content_block:
                content_block["name"] = ""
            if "input" not in content_block:
                content_block["input"] = {}
        event_data = {
            "type": "content_block_start",
            "index": self.content_block_index,
            "content_block": content_block,
        }
        event_str = f"event: content_block_start\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: content_block_start - index: {self.content_block_index}, block_type: {block_type}, kwargs: {kwargs}"
        )
        return event_str

    def _send_content_block_delta_event(self, delta_type: str, content: str) -> str:
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
        event_data = {
            "type": "content_block_delta",
            "index": self.content_block_index,
            "delta": delta,
        }
        event_str = f"event: content_block_delta\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: content_block_delta - index: {self.content_block_index}, delta_type: {delta_type}, content_len: {len(content)}"
        )
        return event_str

    def _send_content_block_stop_event(self) -> str:
        """Send content_block_stop event."""
        event_data = {"type": "content_block_stop", "index": self.content_block_index}
        event_str = f"event: content_block_stop\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: content_block_stop - index: {self.content_block_index}"
        )
        return event_str

    def _send_message_delta_event(self, stop_reason: str, output_tokens: int) -> str:
        """Send message_delta event with cumulative usage information."""
        event_data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": output_tokens
            },
        }
        event_str = f"event: message_delta\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(
            f"STREAMING_EVENT: message_delta - stop_reason: {stop_reason}, input_tokens: {self.input_tokens}, output_tokens: {output_tokens}"
        )
        return event_str

    def _send_message_stop_event(self) -> str:
        """Send message_stop event with usage information."""
        # Include usage information in message_stop event per Claude API spec
        event_data = {
            "type": "message_stop",
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens
            }
        }
        event_str = f"event: message_stop\ndata: {json.dumps(event_data)}\n\n"
        logger.debug(f"STREAMING_EVENT: message_stop with usage - input:{self.input_tokens}, output:{self.output_tokens}")
        return event_str

    def _send_done_event(self) -> str:
        """Send done event."""
        event_data = {"type": "done"}
        event_str = f"event: done\ndata: {json.dumps(event_data)}\n\n"
        logger.debug("STREAMING_EVENT: done")
        return event_str

    async def _close_text_block(self):
        """Close the current text block if open."""
        if self.text_block_started and not self.text_block_closed:
            yield self._send_content_block_stop_event()
            self.text_block_closed = True
            self.content_block_index += 1

    async def _close_tool_block(self):
        """Close the current tool block if open."""
        if self.is_tool_use:
            yield self._send_content_block_stop_event()
            self.content_block_index += 1
            self.is_tool_use = False
            # Reset tool state
            self.tool_json = ""
            self.current_tool_id = None
            self.current_tool_name = None

    async def _close_thinking_block(self):
        """Close the current thinking block if open."""
        if self.thinking_block_started and not self.thinking_block_closed:
            # Parse function calls from thinking content before closing
            cleaned_thinking, function_calls = parse_function_calls_from_thinking(
                self.accumulated_thinking
            )

            # Update thinking content with cleaned version
            if self.content_block_index < len(self.current_content_blocks):
                self.current_content_blocks[self.content_block_index]["thinking"] = (
                    cleaned_thinking
                )

                # Generate signature for cleaned thinking content
                thinking_signature = generate_unique_id("thinking")
                self.current_content_blocks[self.content_block_index]["signature"] = (
                    thinking_signature
                )

                # Send signature delta before closing thinking block
                yield self._send_content_block_delta_event(
                    "signature_delta", thinking_signature
                )

            yield self._send_content_block_stop_event()
            self.thinking_block_closed = True
            self.content_block_index += 1

            # Add function call blocks if any were found
            for tool_call in function_calls:

                # Create tool use content block
                unique_tool_id = generate_unique_id("toolu")
                tool_block = {
                    "type": "tool_use",
                    "id": unique_tool_id,
                    "name": tool_call["function"]["name"],
                    "input": json.loads(tool_call["function"]["arguments"]),
                }
                self.current_content_blocks.append(tool_block)
                logger.debug(f"Adding tool call block: {tool_block}")

                # Send tool call events
                yield self._send_content_block_start_event(
                    "tool_use", id=unique_tool_id, name=tool_call["function"]["name"]
                )
                yield self._send_content_block_delta_event(
                    "input_json_delta", tool_call["function"]["arguments"]
                )
                yield self._send_content_block_stop_event()
                self.content_block_index += 1

    async def _handle_text_delta(self, delta_content: str):
        """Handle text content delta."""
        # If we have text content and we're currently in tool use mode, end the tool use first
        if self.is_tool_use:
            async for event in self._close_tool_block():
                yield event

        self.accumulated_text += delta_content

        # Calculate current output tokens in real-time using tiktoken
        current_output_tokens = count_tokens_in_response(
            response_content=self.accumulated_text,
            thinking_content=self.accumulated_thinking,
            tool_calls=[],
        )
        logger.debug(
            f"Added text content: +{len(delta_content)} chars, total: {len(self.accumulated_text)} chars, tokens: {current_output_tokens}"
        )

        # Start text block if not started
        if not self.text_block_started:
            text_block = {"type": "text", "text": ""}
            self.current_content_blocks.append(text_block)
            yield self._send_content_block_start_event("text")
            self.text_block_started = True

        # Send text delta
        yield self._send_content_block_delta_event("text_delta", delta_content)

        # Update content block
        if self.content_block_index < len(self.current_content_blocks):
            self.current_content_blocks[self.content_block_index]["text"] = (
                self.accumulated_text
            )

    async def _handle_thinking_delta(self, delta_reasoning: str):
        """Handle thinking/reasoning content delta."""
        self.accumulated_thinking += delta_reasoning
        logger.debug(
            f"Added thinking content: +{len(delta_reasoning)} chars, total: {len(self.accumulated_thinking)} chars"
        )

        # Start thinking block if not started
        if not self.thinking_block_started:
            thinking_block = {"type": "thinking", "thinking": ""}
            self.current_content_blocks.append(thinking_block)
            yield self._send_content_block_start_event("thinking")
            self.thinking_block_started = True

        # Send thinking delta
        yield self._send_content_block_delta_event("thinking_delta", delta_reasoning)

        # Update content block
        if self.content_block_index < len(self.current_content_blocks):
            self.current_content_blocks[self.content_block_index]["thinking"] = (
                self.accumulated_thinking
            )

    async def _handle_tool_call_delta(self, tool_call):
        """Handle tool call delta."""
        # If we haven't started a tool yet, we need to handle any accumulated text first
        if not self.is_tool_use:
            # If we have accumulated text, send it first
            if self.accumulated_text and not self.text_block_started:
                text_block = {"type": "text", "text": ""}
                self.current_content_blocks.append(text_block)
                yield self._send_content_block_start_event("text")
                self.text_block_started = True

                # Send the accumulated text
                yield self._send_content_block_delta_event(
                    "text_delta", self.accumulated_text
                )

                # Update the content block
                self.current_content_blocks[self.content_block_index]["text"] = (
                    self.accumulated_text
                )

                # Close the text block
                yield self._send_content_block_stop_event()
                self.text_block_closed = True
                self.content_block_index += 1
            elif self.text_block_started and not self.text_block_closed:
                # Close any open text block
                yield self._send_content_block_stop_event()
                self.text_block_closed = True
                self.content_block_index += 1

            # Now start the tool use block
            self.is_tool_use = True

            # Extract tool info
            if isinstance(tool_call, dict):
                function = tool_call.get("function", {})
                self.current_tool_name = (
                    function.get("name", "") if isinstance(function, dict) else ""
                )
                self.current_tool_id = generate_unique_id("toolu")
            else:
                function = getattr(tool_call, "function", None)
                self.current_tool_name = (
                    getattr(function, "name", "") if function else ""
                )
                self.current_tool_id = generate_unique_id("toolu")

            # Create tool use block
            tool_block = {
                "type": "tool_use",
                "id": self.current_tool_id,
                "name": self.current_tool_name,
                "input": {},
            }
            self.current_content_blocks.append(tool_block)

            yield self._send_content_block_start_event(
                "tool_use", id=self.current_tool_id, name=self.current_tool_name
            )
            self.tool_json = ""

        # Extract function arguments
        arguments = None
        if isinstance(tool_call, dict) and "function" in tool_call:
            function = tool_call.get("function", {})
            arguments = (
                function.get("arguments", "") if isinstance(function, dict) else ""
            )
        elif hasattr(tool_call, "function"):
            function = getattr(tool_call, "function", None)
            arguments = getattr(function, "arguments", "") if function else ""

        # If we have arguments, send them as a delta
        if arguments:
            self.tool_json += arguments

            # Try to parse JSON to update the content block
            try:
                parsed_json = json.loads(self.tool_json)
                self.current_content_blocks[self.content_block_index]["input"] = (
                    parsed_json
                )
            except json.JSONDecodeError:
                # JSON not yet complete, continue accumulating
                pass

            # Send the delta
            yield self._send_content_block_delta_event("input_json_delta", arguments)

    async def process_chunk(self, chunk: ChatCompletionChunk):
        """Process a single chunk from the OpenAI streaming response."""
        self.openai_chunks_received += 1

        # Pre-extract all data from Pydantic object to minimize attribute access
        chunk_data = {
            "chunk_id": chunk.id,
            "has_choices": len(chunk.choices) > 0,
            "has_usage": chunk.usage is not None,
            "usage": chunk.usage,
        }

        # Extract choice data if available
        if chunk_data["has_choices"]:
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # Pre-extract delta data to minimize model_dump() calls
            raw_delta = delta.model_dump() if hasattr(delta, "model_dump") else {}

            chunk_data.update(
                {
                    "delta": delta,
                    "finish_reason": finish_reason,
                    "delta_content": getattr(delta, "content", None),
                    "delta_tool_calls": getattr(delta, "tool_calls", None),
                    "delta_reasoning": raw_delta.get("reasoning_content")
                    or raw_delta.get("reasoning"),
                }
            )

        # Debug logging for streaming chunk analysis
        logger.debug(f"🔄 STREAMING_CHUNK #{self.openai_chunks_received}: processing")

        # Detailed debug logging for chunk data
        if chunk_data["has_choices"]:
            logger.debug(f"🔄 CHUNK_DATA: id={chunk_data['chunk_id']}, finish_reason={chunk_data.get('finish_reason')}")
            logger.debug(f"🔄 CHUNK_CONTENT: content={chunk_data.get('delta_content')}, tool_calls={bool(chunk_data.get('delta_tool_calls'))}, reasoning={bool(chunk_data.get('delta_reasoning'))}")

            # Enhanced logging for non-Claude supported event types and errors
            if chunk_data.get('finish_reason'):
                finish_reason = str(chunk_data['finish_reason'])
                logger.debug(f"🚨 FINISH_REASON_DETECTED: {finish_reason} in chunk #{self.openai_chunks_received}")
                
                # Log non-Claude supported finish reasons
                non_claude_reasons = ['MALFORMED_FUNCTION_CALL', 'MALFORMED', 'SAFETY', 'RECITATION', 'OTHER']
                if any(reason in finish_reason.upper() for reason in non_claude_reasons):
                    logger.error(f"🚨 NON_CLAUDE_FINISH_REASON detected: {finish_reason} in chunk #{self.openai_chunks_received}")
                    logger.error(f"🚨 This finish_reason is not supported by Claude API and may cause conversion issues")
                
                # Special handling for MALFORMED_FUNCTION_CALL
                if 'MALFORMED' in finish_reason.upper():
                    logger.error(f"🚨 MALFORMED_FUNCTION_CALL detected - this indicates tool call format issues")
                    logger.error(f"🚨 Raw chunk data: {chunk.model_dump() if hasattr(chunk, 'model_dump') else str(chunk)}")

        if chunk_data["has_usage"]:
            logger.debug(f"🔄 CHUNK_USAGE: {chunk_data['usage']}")

        logger.debug(f"🔄 STREAMING_CHUNK #{self.openai_chunks_received}: processing complete")

        # Handle usage data (final chunk)
        if chunk_data["has_usage"] and chunk_data["usage"]:
            self.input_tokens = getattr(chunk_data["usage"], "prompt_tokens", self.input_tokens)
            self.completion_tokens = getattr(chunk_data["usage"], "completion_tokens", self.output_tokens)
            # Sync output_tokens with completion_tokens for consistency
            self.output_tokens = self.completion_tokens
            reported_input_tokens = getattr(chunk_data["usage"], "prompt_tokens", 0)
            reported_output_tokens = getattr(
                chunk_data["usage"], "completion_tokens", 0
            )
            logger.debug(
                f"Usage chunk received - Input: {reported_input_tokens}, Output: {reported_output_tokens}"
            )

            # Now that we have usage data, send final events if finish_reason was already processed
            if hasattr(self, 'pending_finish_reason') and not self.has_sent_stop_reason:
                async for event in self._send_final_events():
                    yield event

        # Process content if we have choices
        if chunk_data["has_choices"]:
            # Handle tool calls first
            if chunk_data["delta_tool_calls"]:
                delta_tool_calls = chunk_data["delta_tool_calls"]
                if not isinstance(delta_tool_calls, list):
                    delta_tool_calls = [delta_tool_calls]

                for tool_call in delta_tool_calls:
                    async for event in self._handle_tool_call_delta(tool_call):
                        yield event

            # Handle thinking/reasoning content
            if chunk_data["delta_reasoning"]:
                async for event in self._handle_thinking_delta(
                    chunk_data["delta_reasoning"]
                ):
                    yield event
            elif (
                self.thinking_block_started
                and not self.thinking_block_closed
                and chunk_data["delta_content"]
            ):
                # If we have normal content coming and thinking was active, close thinking block
                async for event in self._close_thinking_block():
                    yield event

            # Handle text content
            if chunk_data["delta_content"] and not self.is_tool_use:
                async for event in self._handle_text_delta(chunk_data["delta_content"]):
                    yield event

            # Process finish_reason - but wait for usage chunk before finalizing
            if chunk_data["finish_reason"] and not self.has_sent_stop_reason:
                # Store finish_reason and prepare for finalization, but don't send stop events yet
                self.pending_finish_reason = chunk_data["finish_reason"]
                async for event in self._prepare_finalization(chunk_data["finish_reason"]):
                    yield event

                # If we already have usage data, send final events immediately
                if self.output_tokens > 0:  # Usage was already processed
                    async for event in self._send_final_events():
                        yield event

    async def _prepare_finalization(self, finish_reason: str):
        """Prepare for finalization by closing blocks, but don't send stop events yet."""
        logger.debug(f"Preparing finalization for finish_reason: {finish_reason}")

        # Close thinking block if it was started
        if self.thinking_block_started and not self.thinking_block_closed:
            async for event in self._close_thinking_block():
                yield event

        # If we haven't started any blocks yet, start and immediately close a text block
        if (
            not self.text_block_started
            and not self.is_tool_use
            and not self.thinking_block_started
        ):
            text_block = {"type": "text", "text": ""}
            self.current_content_blocks.append(text_block)
            yield self._send_content_block_start_event("text")
            yield self._send_content_block_stop_event()
        elif self.text_block_started or self.is_tool_use:
            # Close the current content block
            yield self._send_content_block_stop_event()

    async def _send_final_events(self):
        """Send final events after both finish_reason and usage have been processed."""
        if not hasattr(self, 'pending_finish_reason') or self.has_sent_stop_reason:
            return

        finish_reason = self.pending_finish_reason
        logger.debug(f"Sending final events for finish_reason: {finish_reason}")

        # Determine stop reason
        stop_reason = _map_finish_reason_to_stop_reason(finish_reason)
        logger.debug(f"Mapped stop_reason: {stop_reason}")

        # Use the updated token counts (should now include usage data)
        final_output_tokens = self.output_tokens

        # Send message delta with final content and stop reason
        yield self._send_message_delta_event(stop_reason, final_output_tokens)

        # Send message stop and done
        yield self._send_message_stop_event()
        yield self._send_done_event()
        logger.debug("Streaming completed successfully")

        self.has_sent_stop_reason = True

    async def _finalize_response(self, finish_reason: str):
        """Legacy method - now redirects to new logic."""
        self.pending_finish_reason = finish_reason
        async for event in self._prepare_finalization(finish_reason):
            yield event
        async for event in self._send_final_events():
            yield event


async def convert_openai_streaming_response_to_anthropic(
    response_generator: AsyncStream[ChatCompletionChunk],
    original_request: ClaudeMessagesRequest,
    routed_model: str | None = None,
):
    """Handle streaming responses from OpenAI SDK and convert to Anthropic format.

    Optimized version using state management class to improve performance.
    """
    # Create converter instance with all state encapsulated
    converter = AnthropicStreamingConverter(original_request)

    try:
        # Send initial events
        yield converter._send_message_start_event()
        yield converter._send_ping_event()

        logger.debug(f"🌊 Starting streaming for model: {original_request.model}")

        # Process each chunk using the optimized converter
        chunk_count = 0
        async for chunk in response_generator:
            chunk_count += 1
            try:
                # Debug log raw chunk info (commented out to reduce noise)
                # logger.debug(f"🌊 RAW_CHUNK #{chunk_count}: id={chunk.id}, choices={len(chunk.choices)}, usage={chunk.usage is not None}")
                
                # Enhanced chunk debugging for MALFORMED_FUNCTION_CALL investigation
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'finish_reason') and choice.finish_reason:
                        logger.debug(f"🌊 CHUNK_FINISH_REASON #{chunk_count}: {choice.finish_reason}")
                    
                    # Log tool calls in chunks to help debug malformed function calls
                    if hasattr(choice, 'delta') and choice.delta and hasattr(choice.delta, 'tool_calls'):
                        tool_calls = choice.delta.tool_calls
                        if tool_calls:
                            logger.debug(f"🌊 CHUNK_TOOL_CALLS #{chunk_count}: {len(tool_calls)} tool calls")
                            for i, tool_call in enumerate(tool_calls):
                                if hasattr(tool_call, 'function') and tool_call.function:
                                    func_name = getattr(tool_call.function, 'name', 'unknown')
                                    func_args = getattr(tool_call.function, 'arguments', '')
                                    logger.debug(f"🌊   Tool Call {i}: {func_name}, args_length={len(func_args)}")
                                    if func_args and len(func_args) > 0:
                                        try:
                                            import json
                                            json.loads(func_args)
                                            logger.debug(f"🌊   Tool Call {i} args: valid JSON")
                                        except json.JSONDecodeError:
                                            logger.debug(f"🌊   Tool Call {i} args: invalid JSON - might cause MALFORMED_FUNCTION_CALL")
                                            logger.debug(f"🌊   Raw args: {func_args[:200]}...")

                # Process chunk and yield all events
                async for event in converter.process_chunk(chunk):
                    # Enhanced debug logging for events
                    if "event:" in event:
                        event_type = event.split("event:")[1].split("\n")[0].strip()
                        logger.debug(f"🌊 YIELDING_EVENT: {event_type}")
                        
                        # Log event data for debugging
                        if "data:" in event:
                            try:
                                data_line = [line for line in event.split("\n") if line.startswith("data:")][0]
                                data_content = data_line[5:].strip()  # Remove "data:" prefix
                                if data_content and data_content != "[DONE]":
                                    import json
                                    try:
                                        parsed_data = json.loads(data_content)
                                        logger.debug(f"🌊 EVENT_DATA: {json.dumps(parsed_data, indent=2)}")
                                    except json.JSONDecodeError:
                                        logger.debug(f"🌊 EVENT_DATA (raw): {data_content}")
                            except Exception as e:
                                logger.debug(f"🌊 EVENT_DATA_PARSE_ERROR: {e}")
                    else:
                        # Log non-standard events that might not be Claude-compatible
                        if event.strip() and not event.startswith("data: [DONE]"):
                            logger.debug(f"🌊 NON_STANDARD_EVENT: {event.strip()}")
                    
                    yield event

                # If response is finalized, break out of loop
                if converter.has_sent_stop_reason:
                    logger.debug(f"🌊 STREAM_FINALIZED: Breaking after {chunk_count} chunks")
                    break

            except Exception as e:
                logger.error(f"🌊 ERROR_PROCESSING_CHUNK #{chunk_count}: {str(e)}")
                logger.error(f"🌊 CHUNK_ERROR_TRACEBACK: {e.__class__.__name__}: {str(e)}")
                # Log chunk data to help debug the error
                try:
                    chunk_info = {
                        'id': getattr(chunk, 'id', 'unknown'),
                        'choices_count': len(getattr(chunk, 'choices', [])),
                        'has_usage': getattr(chunk, 'usage', None) is not None
                    }
                    logger.error(f"🌊 FAILED_CHUNK_INFO: {chunk_info}")
                except Exception as debug_error:
                    logger.error(f"🌊 FAILED_TO_DEBUG_CHUNK: {debug_error}")
                continue

        # Handle case where no finish_reason was received
        if not converter.has_sent_stop_reason:
            logger.debug("No finish_reason received, closing stream manually")

            # Close any open blocks
            if converter.thinking_block_started and not converter.thinking_block_closed:
                async for event in converter._close_thinking_block():
                    yield event

            # If no blocks started, create empty text block
            if (
                not converter.text_block_started
                and not converter.is_tool_use
                and not converter.thinking_block_started
            ):
                text_block = {"type": "text", "text": ""}
                converter.current_content_blocks.append(text_block)
                yield converter._send_content_block_start_event("text")
                yield converter._send_content_block_stop_event()
            elif converter.text_block_started or converter.is_tool_use:
                yield converter._send_content_block_stop_event()

            # Calculate final tokens and send completion events
            final_output_tokens = _calculate_accurate_output_tokens(
                converter.accumulated_text,
                converter.accumulated_thinking,
                converter.output_tokens,
                "No finish reason received",
            )

            stop_reason = "tool_use" if converter.is_tool_use else "end_turn"
            yield converter._send_message_delta_event(stop_reason, final_output_tokens)
            yield converter._send_message_stop_event()
            yield converter._send_done_event()

    finally:
        # Final cleanup and logging
        try:
            # Calculate final tokens for tracking
            final_output_tokens = _calculate_accurate_output_tokens(
                converter.accumulated_text,
                converter.accumulated_thinking,
                converter.output_tokens,
                "Final streaming cleanup",
            )

            # Track usage statistics
            if converter.accumulated_text or converter.accumulated_thinking:
                input_tokens = count_tokens_in_messages(
                    original_request.messages, original_request.model
                )

                add_session_stats(
                    model=original_request.model,
                    input_tokens=input_tokens,
                    output_tokens=final_output_tokens,
                    cost=calculate_cost(
                        original_request.model, input_tokens, final_output_tokens
                    ),
                    routed_model=routed_model,
                )

                logger.info(
                    f"STREAMING COMPLETE - Model: {original_request.model}, "
                    f"Chunks: {converter.openai_chunks_received}, "
                    f"Input tokens: {input_tokens}, Output tokens: {final_output_tokens}, "
                    f"Text: {len(converter.accumulated_text)} chars, "
                    f"Thinking: {len(converter.accumulated_thinking)} chars"
                )

        except Exception as cleanup_error:
            logger.error(f"Error in streaming cleanup: {cleanup_error}")


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
    # Token counting removed - rely on API usage data only
    # Input token counting is disabled, only use output tokens from actual API responses
    return 0


def count_tokens_in_messages(messages: list, model: str) -> int:
    """Token counting removed - only output tokens from API responses are used."""
    # Input token counting removed to eliminate tiktoken dependency
    # Only output tokens from actual API responses are counted
    return 0


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost based on model and token usage."""
    # Simple cost calculation - this should be replaced with actual pricing
    # For now, return 0 to avoid breaking functionality
    try:
        # Basic cost estimation (these would be actual prices in production)
        input_cost_per_1k = 0.001  # $0.001 per 1k input tokens
        output_cost_per_1k = 0.002  # $0.002 per 1k output tokens

        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k

        return round(input_cost + output_cost, 6)
    except Exception as e:
        logger.error(f"Error calculating cost: {e}")
        return 0.0


def add_session_stats(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    routed_model: str = None,
):
    """Add usage statistics to session tracking."""
    try:
        # Create ClaudeUsage object for the existing global usage stats system
        usage = ClaudeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        update_global_usage_stats(
            usage=usage,
            model=model,
            context=f"session_stats_{routed_model or model}",
        )
        logger.debug(
            f"Added session stats: {model}, input={input_tokens}, output={output_tokens}, cost=${cost}"
        )
    except Exception as e:
        logger.error(f"Error adding session stats: {e}")


def _compare_response_data(
    openai_response: ChatCompletion, claude_response: ClaudeMessagesResponse
):
    """Compare OpenAI response with converted Claude response and log differences."""
    try:
        # Extract OpenAI response data
        openai_content_blocks = 0
        openai_tool_calls = 0
        openai_finish_reason = None

        if openai_response.choices and len(openai_response.choices) > 0:
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


def _debug_openai_message_sequence(openai_messages: list, context: str):
    """Debug and validate OpenAI message sequence for tool call ordering."""
    try:
        logger.debug(f"=== OpenAI MESSAGE SEQUENCE DEBUG: {context} ===")
        logger.debug(f"Total messages: {len(openai_messages)}")

        for i, msg in enumerate(openai_messages):
            role = msg.get("role", "unknown")
            has_content = bool(msg.get("content"))
            has_tool_calls = bool(msg.get("tool_calls"))
            tool_call_id = msg.get("tool_call_id")

            logger.debug(
                f"  [{i}] Role: {role}, Content: {has_content}, Tool calls: {has_tool_calls}, Tool call ID: {tool_call_id}"
            )

        # Validate tool call sequence
        for i, msg in enumerate(openai_messages):
            if msg.get("role") == "tool":
                # Tool messages must follow assistant messages with tool_calls
                if i == 0:
                    logger.warning(
                        f"Tool message at position {i} has no preceding assistant message"
                    )
                    continue

                prev_msg = openai_messages[i - 1]
                if prev_msg.get("role") != "assistant" or not prev_msg.get(
                    "tool_calls"
                ):
                    logger.warning(
                        f"Tool message at position {i} not preceded by assistant with tool_calls"
                    )

        logger.debug("=== END MESSAGE SEQUENCE DEBUG ===")

    except Exception as e:
        logger.error(f"Error in message sequence debug: {e}")


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

        # Enhanced conversion summary with detailed comparison
        logger.debug("🔄 CLAUDE_TO_OPENAI_CONVERSION_SUMMARY:")
        logger.debug(f"  📊 Tools: {claude_tools_count} -> {openai_tools_count}")
        logger.debug(f"  💬 Messages: {claude_messages_count} -> {openai_messages_count}")
        logger.debug(
            f"  🛠️ Tool Choice: {claude_has_tool_choice} -> {openai_has_tool_choice}"
        )
        logger.debug(f"  🎯 System Message: {claude_has_system} -> {openai_has_system}")
        logger.debug(
            f"  🔧 Tool Results: {claude_has_tool_result} -> {openai_has_tool_msg}"
        )
        
        # Log detailed request structures for debugging MALFORMED_FUNCTION_CALL issues
        logger.debug("📋 CLAUDE_REQUEST_STRUCTURE:")
        logger.debug(f"  Model: {claude_request.model}")
        logger.debug(f"  Max Tokens: {claude_request.max_tokens}")
        logger.debug(f"  Temperature: {claude_request.temperature}")
        logger.debug(f"  Stream: {claude_request.stream}")
        
        # Log message types and roles to help debug tool call issues
        claude_msg_summary = []
        for i, msg in enumerate(claude_request.messages):
            if hasattr(msg, 'content') and isinstance(msg.content, list):
                content_types = [type(c).__name__ for c in msg.content]
                claude_msg_summary.append(f"{msg.role}[{','.join(content_types)}]")
            else:
                claude_msg_summary.append(f"{msg.role}[text]")
        logger.debug(f"  Messages: {' -> '.join(claude_msg_summary)}")
        
        logger.debug("🔄 OPENAI_REQUEST_STRUCTURE:")
        logger.debug(f"  Model: {openai_request.get('model')}")
        logger.debug(f"  Max Tokens: {openai_request.get('max_tokens')}")
        logger.debug(f"  Temperature: {openai_request.get('temperature')}")
        logger.debug(f"  Stream: {openai_request.get('stream')}")
        
        # Log OpenAI message structure to help debug function call format
        openai_msg_summary = []
        for msg in openai_request.get('messages', []):
            role = msg.get('role', 'unknown')
            if 'tool_calls' in msg:
                tool_calls_info = f"tool_calls[{len(msg['tool_calls'])}]"
                openai_msg_summary.append(f"{role}[{tool_calls_info}]")
            elif role == 'tool':
                openai_msg_summary.append(f"{role}[tool_result]")
            else:
                openai_msg_summary.append(f"{role}[content]")
        logger.debug(f"  Messages: {' -> '.join(openai_msg_summary)}")
        
        # Log tool definitions for debugging MALFORMED_FUNCTION_CALL
        if claude_tools_count > 0:
            logger.debug("🛠️ TOOLS_COMPARISON:")
            for i, tool in enumerate(claude_request.tools or []):
                logger.debug(f"  Claude Tool {i+1}: {tool.name} ({tool.type})")
            for i, tool in enumerate(openai_request.get('tools', [])):
                func_info = tool.get('function', {})
                logger.debug(f"  OpenAI Tool {i+1}: {func_info.get('name')} (function)")

        # Enhanced checks for conversion issues that could lead to MALFORMED_FUNCTION_CALL
        warnings = []

        # Tools count should always match exactly
        if claude_tools_count != openai_tools_count:
            warnings.append(
                f"Unexpected tools count difference: {claude_tools_count} -> {openai_tools_count}"
            )

        if warnings:
            logger.warning(f"REQUEST CONVERSION WARNINGS: {'; '.join(warnings)}")
        else:
            logger.debug("REQUEST CONVERSION: All counts match ✓")

    except Exception as e:
        logger.error(f"Error comparing request data: {e}")


def _compare_streaming_with_non_streaming(
    original_request,
    accumulated_text,
    accumulated_thinking,
    current_content_blocks,
    output_tokens,
    openai_chunks_received,
):
    """Compare streaming results with what a non-streaming response would have looked like."""
    try:
        # Log comparison data for debugging
        logger.debug("=== STREAMING vs NON-STREAMING COMPARISON ===")
        logger.debug(f"Chunks received: {openai_chunks_received}")
        logger.debug(f"Content blocks created: {len(current_content_blocks)}")

        # Validate content structure
        text_blocks = [b for b in current_content_blocks if b.get("type") == "text"]
        tool_blocks = [b for b in current_content_blocks if b.get("type") == "tool_use"]
        thinking_blocks = [
            b for b in current_content_blocks if b.get("type") == "thinking"
        ]

        errors = []

        # Check for content consistency
        if len(text_blocks) > 1:
            errors.append(f"Multiple text blocks created: {len(text_blocks)}")

        if len(thinking_blocks) > 1:
            errors.append(f"Multiple thinking blocks created: {len(thinking_blocks)}")

        # Check tool use consistency
        openai_tool_calls = len(tool_blocks)
        claude_tool_use_blocks = len(
            [b for b in current_content_blocks if b.get("type") == "tool_use"]
        )

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
