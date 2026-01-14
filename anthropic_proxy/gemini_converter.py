"""
Gemini ↔ Anthropic conversion helpers.

This module converts Anthropic Messages requests to Gemini GenerateContent
requests (Gemini-style contents/parts). It also provides utilities for
function/tool mapping and thinking configuration.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .converter import clean_gemini_schema
from .types import (
    ClaudeContentBlockImage,
    ClaudeContentBlockImageBase64Source,
    ClaudeContentBlockImageURLSource,
    ClaudeContentBlockText,
    ClaudeContentBlockThinking,
    ClaudeContentBlockToolResult,
    ClaudeContentBlockToolUse,
    ClaudeMessagesRequest,
)

logger = logging.getLogger(__name__)


def _sanitize_tool_name(name: str) -> str:
    """Sanitize tool name to comply with Gemini/Antigravity constraints."""
    if not name:
        return "tool"

    sanitized = "".join(ch if ch.isalnum() or ch in {"_", ".", ":", "-"} else "_" for ch in name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"

    if not sanitized:
        sanitized = "tool"

    # Antigravity spec: max length 64
    return sanitized[:64]


def _build_tool_name_mapping(request: ClaudeMessagesRequest) -> dict[str, str]:
    """Build tool_use_id -> tool_name mapping from assistant tool_use blocks."""
    mapping: dict[str, str] = {}
    for msg in request.messages:
        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, ClaudeContentBlockToolUse):
                    mapping[block.id] = block.name
    return mapping


def _convert_image_block(block: ClaudeContentBlockImage) -> dict[str, Any] | None:
    """Convert Claude image block to Gemini part if possible."""
    source = block.source
    if isinstance(source, ClaudeContentBlockImageBase64Source):
        return {
            "inlineData": {
                "mimeType": source.media_type,
                "data": source.data,
            }
        }
    if isinstance(source, ClaudeContentBlockImageURLSource):
        return {
            "fileData": {
                "fileUri": source.url,
            }
        }
    if isinstance(source, dict):
        # Best-effort handling for unknown image source dicts
        if source.get("type") == "base64" and source.get("data") and source.get("media_type"):
            return {
                "inlineData": {
                    "mimeType": source.get("media_type"),
                    "data": source.get("data"),
                }
            }
        if source.get("type") == "url" and source.get("url"):
            return {
                "fileData": {
                    "fileUri": source.get("url"),
                }
            }
    return None


def _convert_thinking_block(block: ClaudeContentBlockThinking, is_antigravity: bool) -> dict[str, Any]:
    part: dict[str, Any] = {"text": block.thinking}
    if block.signature:
        part["thoughtSignature"] = block.signature
    if is_antigravity:
        part["thought"] = True
    return part


def _convert_tool_use_block(block: ClaudeContentBlockToolUse) -> dict[str, Any]:
    return {
        "functionCall": {
            "name": _sanitize_tool_name(block.name),
            "args": block.input,
            "id": block.id,
        }
    }


def _convert_tool_result_block(
    block: ClaudeContentBlockToolResult,
    tool_name_mapping: dict[str, str],
) -> dict[str, Any]:
    tool_name = tool_name_mapping.get(block.tool_use_id, "unknown_function")
    content = block.process_content()
    response = {"content": content}
    return {
        "functionResponse": {
            "name": _sanitize_tool_name(tool_name),
            "id": block.tool_use_id,
            "response": response,
        }
    }


def _convert_message_content_to_parts(
    request: ClaudeMessagesRequest,
    content: str | list[Any],
    is_antigravity: bool,
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]

    parts: list[dict[str, Any]] = []
    tool_name_mapping = _build_tool_name_mapping(request)

    for block in content:
        if isinstance(block, ClaudeContentBlockText):
            parts.append({"text": block.text})
        elif isinstance(block, ClaudeContentBlockThinking):
            parts.append(_convert_thinking_block(block, is_antigravity))
        elif isinstance(block, ClaudeContentBlockToolUse):
            parts.append(_convert_tool_use_block(block))
        elif isinstance(block, ClaudeContentBlockToolResult):
            parts.append(_convert_tool_result_block(block, tool_name_mapping))
        elif isinstance(block, ClaudeContentBlockImage):
            image_part = _convert_image_block(block)
            if image_part:
                parts.append(image_part)
        else:
            # Best-effort fallback
            parts.append({"text": json.dumps(block, ensure_ascii=False)})

    return parts


def _thinking_budget_to_level(budget: int | None) -> str | None:
    if budget is None:
        return None
    if budget <= 1024:
        return "low"
    if budget <= 8192:
        return "medium"
    return "high"


def _infer_thinking_level_from_model(model_id: str) -> str | None:
    lower = model_id.lower()
    if "-low" in lower:
        return "low"
    if "-medium" in lower:
        return "medium"
    if "-high" in lower:
        return "high"
    return None


def _build_generation_config(request: ClaudeMessagesRequest, model_id: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if request.temperature is not None:
        config["temperature"] = request.temperature
    if request.top_p is not None:
        config["topP"] = request.top_p
    if request.top_k is not None:
        config["topK"] = request.top_k
    if request.stop_sequences:
        config["stopSequences"] = request.stop_sequences
    if request.max_tokens is not None:
        config["maxOutputTokens"] = request.max_tokens

    if request.thinking is not None:
        if request.thinking.type == "enabled":
            budget = request.thinking.budget_tokens
            thinking_config: dict[str, Any] = {"includeThoughts": True}
            if "gemini-3" in model_id.lower():
                level = _infer_thinking_level_from_model(model_id) or _thinking_budget_to_level(budget)
                if level:
                    thinking_config["thinkingLevel"] = level
            else:
                if budget is not None:
                    thinking_config["thinkingBudget"] = budget
            config["thinkingConfig"] = thinking_config
        elif request.thinking.type == "disabled":
            config["thinkingConfig"] = {"includeThoughts": False}

    return config


def _build_tool_config(request: ClaudeMessagesRequest, model_id: str, is_antigravity: bool) -> dict[str, Any] | None:
    if not request.tools:
        return None

    mode = None
    allowed = None

    if request.tool_choice:
        choice = request.tool_choice
        if choice.type == "auto":
            mode = "AUTO"
        elif choice.type == "any":
            mode = "ANY"
        elif choice.type == "none":
            mode = "NONE"
        elif choice.type == "tool":
            mode = "ANY"
            allowed = [_sanitize_tool_name(choice.name)]

    if is_antigravity and "claude" in model_id.lower():
        if mode is None:
            mode = "VALIDATED"
        elif mode != "NONE":
            mode = "VALIDATED"

    if mode is None:
        return None

    function_calling: dict[str, Any] = {"mode": mode}
    if allowed:
        function_calling["allowedFunctionNames"] = allowed

    return {"functionCallingConfig": function_calling}


def _build_tools(request: ClaudeMessagesRequest) -> list[dict[str, Any]] | None:
    if not request.tools:
        return None

    function_declarations: list[dict[str, Any]] = []
    for tool in request.tools:
        parameters = clean_gemini_schema(tool.input_schema)
        if not parameters or not isinstance(parameters, dict):
            parameters = {
                "type": "object",
                "properties": {
                    "_placeholder": {
                        "type": "boolean",
                        "description": "Placeholder. Always pass true.",
                    }
                },
                "required": ["_placeholder"],
                "additionalProperties": False,
            }
        function_declarations.append(
            {
                "name": _sanitize_tool_name(tool.name),
                "description": tool.description or "",
                "parameters": parameters,
            }
        )

    if not function_declarations:
        return None

    return [{"functionDeclarations": function_declarations}]


def anthropic_to_gemini_request(
    request: ClaudeMessagesRequest,
    model_id: str,
    *,
    is_antigravity: bool = False,
    system_prefix: str | None = None,
) -> dict[str, Any]:
    """Convert Anthropic Messages request into Gemini GenerateContent request body."""
    contents: list[dict[str, Any]] = []

    for msg in request.messages:
        role = "user" if msg.role == "user" else "model"
        parts = _convert_message_content_to_parts(request, msg.content, is_antigravity)
        if parts:
            contents.append({"role": role, "parts": parts})

    system_text = request.extract_system_content()
    system_parts: list[dict[str, Any]] = []
    if system_prefix:
        system_parts.append({"text": system_prefix})
    if system_text:
        system_parts.append({"text": system_text})

    generation_config = _build_generation_config(request, model_id)

    # Antigravity requires maxOutputTokens > thinkingBudget
    if is_antigravity and generation_config.get("thinkingConfig"):
        thinking_budget = generation_config["thinkingConfig"].get("thinkingBudget")
        if thinking_budget and generation_config.get("maxOutputTokens", 0) <= thinking_budget:
            generation_config["maxOutputTokens"] = thinking_budget + 4000

    out: dict[str, Any] = {
        "contents": contents,
    }

    if system_parts:
        out["systemInstruction"] = {"parts": system_parts}
    if generation_config:
        out["generationConfig"] = generation_config

    tools = _build_tools(request)
    if tools:
        out["tools"] = tools

    tool_config = _build_tool_config(request, model_id, is_antigravity)
    if tool_config:
        out["toolConfig"] = tool_config

    return out
