"""
Gemini ↔ Anthropic converter.

Converts between Gemini GenerateContent format and Anthropic Messages format.
Re-exports existing conversion logic from gemini_converter.py with the new interface.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from ._gemini_impl import (
    anthropic_to_gemini_request,
    anthropic_to_gemini_sdk_params,
    ensure_tool_ids,
    _clean_malformed_parts,
)
from ._gemini_streaming import (
    convert_gemini_streaming_response_to_anthropic,
    GeminiStreamingConverter as GeminiStreamingConverterImpl,
)
from ..types import (
    ClaudeContentBlockImage,
    ClaudeContentBlockImageBase64Source,
    ClaudeContentBlockImageURLSource,
    ClaudeContentBlockText,
    ClaudeContentBlockThinking,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockToolUse,
    ClaudeMessage,
    ClaudeMessagesRequest,
    ClaudeMessagesResponse,
    ClaudeThinkingConfigDisabled,
    ClaudeThinkingConfigEnabled,
    ClaudeTool,
    ClaudeToolChoiceAny,
    ClaudeToolChoiceAuto,
    ClaudeToolChoiceNone,
    ClaudeToolChoiceTool,
    ClaudeUsage,
    generate_unique_id,
)
from .base import BaseConverter, BaseStreamingConverter

logger = logging.getLogger(__name__)


def _parse_gemini_tool_choice(tool_config: dict[str, Any] | None) -> Any:
    """Convert Gemini toolConfig to Claude tool_choice."""
    if not tool_config:
        return None

    function_calling = tool_config.get("functionCallingConfig", {})
    mode = function_calling.get("mode", "").upper()
    allowed = function_calling.get("allowedFunctionNames", [])

    if mode == "AUTO":
        return ClaudeToolChoiceAuto()
    elif mode == "ANY":
        if allowed and len(allowed) == 1:
            return ClaudeToolChoiceTool(name=allowed[0])
        return ClaudeToolChoiceAny()
    elif mode == "NONE":
        return ClaudeToolChoiceNone()

    return None


def _parse_gemini_tools(tools: list[dict[str, Any]] | None) -> list[ClaudeTool] | None:
    """Convert Gemini tools to Claude tools."""
    if not tools:
        return None

    claude_tools = []
    for tool in tools:
        function_declarations = tool.get("functionDeclarations", [])
        for func in function_declarations:
            claude_tools.append(
                ClaudeTool(
                    name=func.get("name", ""),
                    description=func.get("description"),
                    input_schema=func.get("parameters", {"type": "object", "properties": {}}),
                )
            )
    return claude_tools if claude_tools else None


def _parse_gemini_part(part: dict[str, Any]) -> Any:
    """Convert a Gemini part to Claude content block."""
    if "text" in part:
        # Check for thought/thinking
        if part.get("thought") or part.get("thoughtSignature"):
            return ClaudeContentBlockThinking(
                type="thinking",
                thinking=part["text"],
                signature=part.get("thoughtSignature"),
            )
        return ClaudeContentBlockText(type="text", text=part["text"])

    elif "inlineData" in part:
        inline = part["inlineData"]
        return ClaudeContentBlockImage(
            type="image",
            source=ClaudeContentBlockImageBase64Source(
                type="base64",
                media_type=inline.get("mimeType", "image/png"),
                data=inline.get("data", ""),
            ),
        )

    elif "fileData" in part:
        file_data = part["fileData"]
        return ClaudeContentBlockImage(
            type="image",
            source=ClaudeContentBlockImageURLSource(
                type="url",
                url=file_data.get("fileUri", ""),
            ),
        )

    elif "functionCall" in part:
        fc = part["functionCall"]
        return ClaudeContentBlockToolUse(
            type="tool_use",
            id=fc.get("id", generate_unique_id("toolu")),
            name=fc.get("name", ""),
            input=fc.get("args", {}),
        )

    elif "functionResponse" in part:
        fr = part["functionResponse"]
        response = fr.get("response", {})
        content = response.get("content", "")
        if isinstance(content, dict):
            content = str(content)
        return ClaudeContentBlockToolResult(
            type="tool_result",
            tool_use_id=fr.get("id", ""),
            content=content,
        )

    return None


def _parse_gemini_contents(contents: list[dict[str, Any]]) -> list[ClaudeMessage]:
    """Convert Gemini contents to Claude messages."""
    claude_messages: list[ClaudeMessage] = []

    for content in contents:
        role = content.get("role", "user")
        claude_role = "user" if role == "user" else "assistant"
        parts = content.get("parts", [])

        blocks: list[Any] = []
        for part in parts:
            block = _parse_gemini_part(part)
            if block:
                blocks.append(block)

        if blocks:
            # Check if all blocks are tool_results - they go in user messages
            if all(isinstance(b, ClaudeContentBlockToolResult) for b in blocks):
                claude_messages.append(ClaudeMessage(role="user", content=blocks))
            else:
                claude_messages.append(ClaudeMessage(role=claude_role, content=blocks))
        elif not blocks and parts:
            # Empty content
            claude_messages.append(ClaudeMessage(role=claude_role, content=""))

    return claude_messages


def _parse_gemini_system(system_instruction: dict[str, Any] | None) -> str | None:
    """Extract system prompt from Gemini systemInstruction."""
    if not system_instruction:
        return None

    parts = system_instruction.get("parts", [])
    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(part["text"])
    return "\n".join(texts) if texts else None


def _parse_gemini_thinking_config(generation_config: dict[str, Any] | None) -> Any:
    """Convert Gemini thinkingConfig to Claude thinking."""
    if not generation_config:
        return None

    thinking_config = generation_config.get("thinkingConfig")
    if not thinking_config:
        return None

    include_thoughts = thinking_config.get("includeThoughts", False)
    if not include_thoughts:
        return ClaudeThinkingConfigDisabled()

    budget = thinking_config.get("thinkingBudget")
    return ClaudeThinkingConfigEnabled(budget_tokens=budget)


class GeminiConverter(BaseConverter):
    """
    Converter between Gemini GenerateContent and Anthropic Messages format.
    """

    def __init__(self, *, is_antigravity: bool = False, session_id: str | None = None):
        self.is_antigravity = is_antigravity
        self.session_id = session_id

    def request_to_anthropic(self, payload: dict[str, Any]) -> ClaudeMessagesRequest:
        """Convert Gemini GenerateContent request to Anthropic Messages format."""
        model = payload.get("model", "")
        contents = payload.get("contents", [])
        generation_config = payload.get("generationConfig", {})
        system_instruction = payload.get("systemInstruction")
        tools = payload.get("tools")
        tool_config = payload.get("toolConfig")

        # Parse messages
        claude_messages = _parse_gemini_contents(contents)

        # Parse system
        system_prompt = _parse_gemini_system(system_instruction)

        # Build request
        request_data: dict[str, Any] = {
            "model": model,
            "max_tokens": generation_config.get("maxOutputTokens", 4096),
            "messages": claude_messages,
            "stream": False,  # Gemini streaming is handled differently
        }

        if system_prompt:
            request_data["system"] = system_prompt
        if "temperature" in generation_config:
            request_data["temperature"] = generation_config["temperature"]
        if "topP" in generation_config:
            request_data["top_p"] = generation_config["topP"]
        if "topK" in generation_config:
            request_data["top_k"] = generation_config["topK"]
        if "stopSequences" in generation_config:
            request_data["stop_sequences"] = generation_config["stopSequences"]

        # Convert tools
        claude_tools = _parse_gemini_tools(tools)
        if claude_tools:
            request_data["tools"] = claude_tools

        # Convert tool_choice
        claude_tool_choice = _parse_gemini_tool_choice(tool_config)
        if claude_tool_choice:
            request_data["tool_choice"] = claude_tool_choice

        # Convert thinking config
        thinking = _parse_gemini_thinking_config(generation_config)
        if thinking:
            request_data["thinking"] = thinking

        return ClaudeMessagesRequest.model_validate(request_data)

    def request_from_anthropic(
        self,
        request: ClaudeMessagesRequest,
        *,
        model_id: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert Anthropic Messages request to Gemini format."""
        system_prefix = kwargs.get("system_prefix")
        session_id = kwargs.get("session_id", self.session_id)
        is_antigravity = kwargs.get("is_antigravity", self.is_antigravity)

        return anthropic_to_gemini_request(
            request,
            model_id or request.model,
            is_antigravity=is_antigravity,
            system_prefix=system_prefix,
            session_id=session_id,
        )

    def response_to_anthropic(
        self,
        response: Any,
        original_request: ClaudeMessagesRequest,
    ) -> ClaudeMessagesResponse:
        """Convert Gemini response to Anthropic format."""
        # Handle dict response
        if isinstance(response, dict):
            candidates = response.get("candidates", [])
            if not candidates:
                return ClaudeMessagesResponse(
                    id=generate_unique_id("msg"),
                    model=original_request.model,
                    content=[ClaudeContentBlockText(type="text", text="")],
                    stop_reason="end_turn",
                    usage=ClaudeUsage(input_tokens=0, output_tokens=0),
                )

            candidate = candidates[0]
            content_data = candidate.get("content", {})
            parts = content_data.get("parts", [])

            blocks: list[Any] = []
            for part in parts:
                block = _parse_gemini_part(part)
                if block:
                    blocks.append(block)

            if not blocks:
                blocks = [ClaudeContentBlockText(type="text", text="")]

            # Determine stop reason
            finish_reason = candidate.get("finishReason", "STOP")
            stop_reason = "end_turn"
            if finish_reason == "MAX_TOKENS":
                stop_reason = "max_tokens"
            elif finish_reason in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
                stop_reason = "refusal"
            elif finish_reason == "TOOL_USE" or any(isinstance(b, ClaudeContentBlockToolUse) for b in blocks):
                stop_reason = "tool_use"

            # Extract usage
            usage_metadata = response.get("usageMetadata", {})
            input_tokens = usage_metadata.get("promptTokenCount", 0)
            output_tokens = usage_metadata.get("candidatesTokenCount", 0)

            return ClaudeMessagesResponse(
                id=generate_unique_id("msg"),
                model=original_request.model,
                content=blocks,
                stop_reason=stop_reason,
                usage=ClaudeUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            )

        raise TypeError(f"Cannot convert response of type {type(response)}")

    def response_from_anthropic(
        self,
        response: ClaudeMessagesResponse,
    ) -> dict[str, Any]:
        """Convert Anthropic response to Gemini format."""
        parts: list[dict[str, Any]] = []

        for block in response.content:
            if isinstance(block, ClaudeContentBlockText):
                parts.append({"text": block.text})
            elif isinstance(block, ClaudeContentBlockThinking):
                part: dict[str, Any] = {"text": block.thinking, "thought": True}
                if block.signature:
                    part["thoughtSignature"] = block.signature
                parts.append(part)
            elif isinstance(block, ClaudeContentBlockToolUse):
                parts.append({
                    "functionCall": {
                        "name": block.name,
                        "args": block.input,
                        "id": block.id,
                    }
                })

        # Determine finish reason
        finish_reason = "STOP"
        if response.stop_reason == "max_tokens":
            finish_reason = "MAX_TOKENS"
        elif response.stop_reason == "tool_use":
            finish_reason = "TOOL_USE"
        elif response.stop_reason == "refusal":
            finish_reason = "SAFETY"

        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": parts,
                    },
                    "finishReason": finish_reason,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": response.usage.input_tokens,
                "candidatesTokenCount": response.usage.output_tokens,
                "totalTokenCount": response.usage.input_tokens + response.usage.output_tokens,
            },
        }


class GeminiStreamingConverter(BaseStreamingConverter):
    """
    Streaming converter for Gemini format.
    """

    def __init__(self, *, is_antigravity: bool = False, session_id: str | None = None):
        self.is_antigravity = is_antigravity
        self.session_id = session_id

    async def stream_to_anthropic(
        self,
        stream: AsyncIterator[Any],
        original_request: ClaudeMessagesRequest,
    ) -> AsyncIterator[str]:
        """Convert Gemini streaming response to Anthropic SSE format."""
        from ._gemini_streaming import convert_gemini_streaming_response_to_anthropic

        async for event in convert_gemini_streaming_response_to_anthropic(
            stream,
            original_request,
            session_id=self.session_id,
        ):
            yield event

    async def stream_from_anthropic(
        self,
        stream: AsyncIterator[str],
        model: str = "",
    ) -> AsyncIterator[str]:
        """
        Convert Anthropic SSE stream to Gemini streaming format.

        Anthropic events -> Gemini chunks.
        Gemini streaming uses newline-delimited JSON chunks, each with candidates array.
        """
        import json

        current_text = ""
        current_tool_calls: list[dict[str, Any]] = {}
        thinking_text = ""

        async for event_str in stream:
            if not event_str.strip():
                continue

            # Extract data from SSE format
            data_line = None
            for line in event_str.strip().split("\n"):
                if line.startswith("data: "):
                    data_line = line[6:]
                    break

            if not data_line or data_line == "[DONE]":
                continue

            try:
                event = json.loads(data_line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "content_block_start":
                block = event.get("content_block", {})
                block_type = block.get("type", "")
                block_index = event.get("index", 0)

                if block_type == "tool_use":
                    current_tool_calls[block_index] = {
                        "id": block.get("id"),
                        "name": block.get("name", ""),
                        "args": "",
                    }

            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                delta_type = delta.get("type", "")
                block_index = event.get("index", 0)

                if delta_type == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        current_text += text
                        # Emit Gemini chunk
                        chunk = {
                            "candidates": [{
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": text}],
                                },
                            }],
                        }
                        yield json.dumps(chunk) + "\n"

                elif delta_type == "input_json_delta":
                    partial_json = delta.get("partial_json", "")
                    if partial_json and block_index in current_tool_calls:
                        current_tool_calls[block_index]["args"] += partial_json

                elif delta_type == "thinking_delta":
                    thinking = delta.get("thinking", "")
                    if thinking:
                        thinking_text += thinking
                        # Emit thinking chunk
                        chunk = {
                            "candidates": [{
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": thinking, "thought": True}],
                                },
                            }],
                        }
                        yield json.dumps(chunk) + "\n"

            elif event_type == "content_block_stop":
                block_index = event.get("index", 0)
                if block_index in current_tool_calls:
                    tc = current_tool_calls[block_index]
                    try:
                        args = json.loads(tc["args"]) if tc["args"] else {}
                    except json.JSONDecodeError:
                        args = {}

                    chunk = {
                        "candidates": [{
                            "content": {
                                "role": "model",
                                "parts": [{
                                    "functionCall": {
                                        "name": tc["name"],
                                        "args": args,
                                        "id": tc["id"],
                                    }
                                }],
                            },
                        }],
                    }
                    yield json.dumps(chunk) + "\n"
                    del current_tool_calls[block_index]

            elif event_type == "message_delta":
                delta = event.get("delta", {})
                stop_reason = delta.get("stop_reason", "")
                usage = event.get("usage", {})

                finish_reason = "STOP"
                if stop_reason == "max_tokens":
                    finish_reason = "MAX_TOKENS"
                elif stop_reason == "tool_use":
                    finish_reason = "TOOL_USE"
                elif stop_reason == "refusal":
                    finish_reason = "SAFETY"

                # Emit final chunk with finish reason and usage
                chunk: dict[str, Any] = {
                    "candidates": [{
                        "content": {"role": "model", "parts": []},
                        "finishReason": finish_reason,
                    }],
                }
                if usage:
                    chunk["usageMetadata"] = {
                        "promptTokenCount": usage.get("input_tokens", 0),
                        "candidatesTokenCount": usage.get("output_tokens", 0),
                        "totalTokenCount": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    }
                yield json.dumps(chunk) + "\n"
