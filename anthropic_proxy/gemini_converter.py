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
from .gemini_types import parse_gemini_request
from .signature_cache import get_cached_signature, get_tool_signature
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
        if not source.data:
            return {"text": "[image omitted: empty base64 data]"}
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
        if source.get("type") == "base64" and not source.get("data"):
            return {"text": "[image omitted: empty base64 data]"}
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


def _convert_tool_use_block(
    block: ClaudeContentBlockToolUse,
    signature: str | None = None,
) -> dict[str, Any]:
    part: dict[str, Any] = {
        "functionCall": {
            "name": _sanitize_tool_name(block.name),
            "args": block.input,
            "id": block.id,
        }
    }
    if signature:
        part["thoughtSignature"] = signature
    return part


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
    strip_unsigned_thinking: bool,
    session_id: str | None,
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]

    parts: list[dict[str, Any]] = []
    tool_name_mapping = _build_tool_name_mapping(request)

    for block in content:
        if isinstance(block, ClaudeContentBlockText):
            parts.append({"text": block.text})
        elif isinstance(block, ClaudeContentBlockThinking):
            signature = block.signature
            if strip_unsigned_thinking and not signature and session_id:
                signature = get_cached_signature(session_id, block.thinking)
            if strip_unsigned_thinking and not signature:
                continue
            part: dict[str, Any] = {"text": block.thinking}
            if signature:
                part["thoughtSignature"] = signature
            if is_antigravity:
                part["thought"] = True
            parts.append(part)
        elif isinstance(block, ClaudeContentBlockToolUse):
            signature = None
            if block.id:
                signature = get_tool_signature(session_id, block.id)
            parts.append(_convert_tool_use_block(block, signature))
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
    else:
        config["temperature"] = 1.0
    if request.top_p is not None:
        config["topP"] = request.top_p
    else:
        config["topP"] = 0.95
    if request.top_k is not None:
        config["topK"] = request.top_k
    else:
        config["topK"] = 64
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

    logger.debug("Gemini converter tools input: %d", len(request.tools))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Gemini converter tool names: %s",
            [tool.name for tool in request.tools],
        )

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

    logger.debug(
        "Gemini converter tools output (functionDeclarations): %d",
        len(function_declarations),
    )
    return [{"functionDeclarations": function_declarations}]


def anthropic_to_gemini_request(
    request: ClaudeMessagesRequest,
    model_id: str,
    *,
    is_antigravity: bool = False,
    system_prefix: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Convert Anthropic Messages request into Gemini GenerateContent request body."""
    contents: list[dict[str, Any]] = []
    strip_unsigned_thinking = is_antigravity and "claude" in model_id.lower()

    for msg in request.messages:
        role = "user" if msg.role == "user" else "model"
        parts = _convert_message_content_to_parts(
            request,
            msg.content,
            is_antigravity,
            strip_unsigned_thinking,
            session_id,
        )
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

    out: dict[str, Any] = {"contents": contents}
    if session_id:
        out["sessionId"] = session_id

    if system_parts:
        out["systemInstruction"] = {"parts": system_parts}
    if generation_config:
        out["generationConfig"] = generation_config

    tools = _build_tools(request)
    if tools:
        out["tools"] = tools
        logger.debug(
            "Gemini request tools attached: %d",
            len(tools[0].get("functionDeclarations", [])),
        )

    tool_config = _build_tool_config(request, model_id, is_antigravity)
    if tool_config:
        out["toolConfig"] = tool_config
        function_config = tool_config.get("functionCallingConfig", {})
        logger.debug(
            "Gemini request toolConfig mode=%s allowed=%s",
            function_config.get("mode"),
            function_config.get("allowedFunctionNames"),
        )

    return parse_gemini_request(out)


def _camel_to_snake(name: str) -> str:
    out = []
    for ch in name:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out).lstrip("_")


def _convert_keys_to_snake(value: Any) -> Any:
    if isinstance(value, dict):
        return {_camel_to_snake(k): _convert_keys_to_snake(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_keys_to_snake(item) for item in value]
    return value


def _system_parts_to_text(system_instruction: dict[str, Any]) -> str | None:
    parts = system_instruction.get("parts") if system_instruction else None
    if not parts:
        return None
    texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    joined = "\n".join(text for text in texts if text)
    return joined or None


def _build_sdk_tools(request: ClaudeMessagesRequest) -> list[dict[str, Any]] | None:
    tools = _build_tools(request)
    if not tools:
        return None

    function_decls = []
    for tool in tools:
        for decl in tool.get("functionDeclarations", []):
            function_decls.append(
                {
                    "name": decl.get("name", ""),
                    "description": decl.get("description", ""),
                    "parameters_json_schema": decl.get("parameters", {}),
                }
            )

    if not function_decls:
        return None

    return [{"function_declarations": function_decls}]


def anthropic_to_gemini_sdk_params(
    request: ClaudeMessagesRequest,
    model_id: str,
    *,
    is_antigravity: bool = False,
    system_prefix: str | None = None,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Convert Anthropic Messages request into Gemini SDK params."""
    body = anthropic_to_gemini_request(
        request,
        model_id,
        is_antigravity=is_antigravity,
        system_prefix=system_prefix,
        session_id=session_id,
    )

    contents = body.get("contents", [])
    config: dict[str, Any] = {}

    system_text = _system_parts_to_text(body.get("systemInstruction", {}))
    if system_text:
        config["system_instruction"] = system_text

    generation_config = body.get("generationConfig")
    if generation_config:
        config.update(_convert_keys_to_snake(generation_config))

    tools = _build_sdk_tools(request)
    if tools:
        config["tools"] = tools

    tool_config = body.get("toolConfig")
    if tool_config:
        config["tool_config"] = _convert_keys_to_snake(tool_config)

    return contents, config, body
