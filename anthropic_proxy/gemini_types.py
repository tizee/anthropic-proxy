"""
Gemini type helpers for streaming conversion.

Note: This module focuses on Anthropic→Gemini request shaping and Gemini→Anthropic
streaming response normalization. Non-streaming Gemini responses are intentionally
out of scope for now.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class GeminiFunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    args: dict[str, Any] | None = None
    partial_args: list[Any] | None = None
    will_continue: bool | None = None

    @field_validator("args", mode="before")
    @classmethod
    def _validate_args(cls, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return _coerce_args(value)


class GeminiFunctionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    response: dict[str, Any] | None = None


class GeminiPart(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str | None = None
    thought: bool | None = None
    thoughtSignature: str | None = None
    functionCall: GeminiFunctionCall | None = None
    functionResponse: GeminiFunctionResponse | None = None

    @field_validator("thoughtSignature", mode="before")
    @classmethod
    def _coerce_thought_signature(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)


class GeminiContent(BaseModel):
    model_config = ConfigDict(extra="allow")

    parts: list[GeminiPart] | None = None
    role: str | None = None


class GeminiCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: GeminiContent | None = None
    finishReason: str | None = None

    @field_validator("finishReason", mode="before")
    @classmethod
    def _coerce_finish_reason(cls, value: Any) -> str | None:
        if value is None:
            return None
        return str(_coerce_enum(value))


class GeminiUsageMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidatesTokenCount: int | None = None
    promptTokenCount: int | None = None
    totalTokenCount: int | None = None


class GeminiGenerateContentResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidates: list[GeminiCandidate] | None = None
    usageMetadata: GeminiUsageMetadata | None = None


class GeminiFunctionDeclaration(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str | None = None
    description: str | None = None
    parameters: dict[str, Any] | None = None


class GeminiTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    functionDeclarations: list[GeminiFunctionDeclaration] | None = None


class GeminiToolConfigFunctionCallingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: str | None = None
    allowedFunctionNames: list[str] | None = None


class GeminiToolConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    functionCallingConfig: GeminiToolConfigFunctionCallingConfig | None = None


class GeminiGenerateContentRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    contents: list[GeminiContent] | None = None
    systemInstruction: dict[str, Any] | None = None
    generationConfig: dict[str, Any] | None = None
    tools: list[GeminiTool] | None = None
    toolConfig: GeminiToolConfig | None = None
    sessionId: str | None = None


def _coerce_enum(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value


def _coerce_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _normalize_part(part: dict[str, Any]) -> dict[str, Any]:
    if "function_call" in part and "functionCall" not in part:
        part["functionCall"] = part.pop("function_call")
    if "thought_signature" in part and "thoughtSignature" not in part:
        part["thoughtSignature"] = part.pop("thought_signature")

    call = part.get("functionCall")
    if isinstance(call, dict):
        args = _coerce_args(call.get("args"))
        call["args"] = args
        if call.get("partialArgs") is None and "partial_args" in call:
            call["partialArgs"] = call.pop("partial_args")
        if call.get("willContinue") is None and "will_continue" in call:
            call["willContinue"] = call.pop("will_continue")

    response = part.get("functionResponse")
    if isinstance(response, dict):
        if response.get("willContinue") is None and "will_continue" in response:
            response["willContinue"] = response.pop("will_continue")

    return part


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    if "finish_reason" in candidate and "finishReason" not in candidate:
        candidate["finishReason"] = candidate.pop("finish_reason")
    finish_reason = candidate.get("finishReason")
    candidate["finishReason"] = _coerce_enum(finish_reason)

    content = candidate.get("content")
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            content["parts"] = [
                _normalize_part(part) if isinstance(part, dict) else part
                for part in parts
            ]
    return candidate


def _append_function_calls(payload: dict[str, Any], calls: list[Any]) -> None:
    parts: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        args = _coerce_args(call.get("args"))
        function_call: dict[str, Any] = {
            "name": call.get("name", ""),
            "args": args,
        }
        if call.get("id"):
            function_call["id"] = call["id"]
        parts.append({"functionCall": function_call})

    if not parts:
        return

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        payload["candidates"] = [{"content": {"parts": parts}}]
        return

    candidate = candidates[0]
    content = candidate.get("content")
    if not isinstance(content, dict):
        content = {}
        candidate["content"] = content
    existing_parts = content.get("parts")
    if not isinstance(existing_parts, list):
        existing_parts = []
        content["parts"] = existing_parts
    existing_parts.extend(parts)


def normalize_gemini_response(payload: dict[str, Any]) -> dict[str, Any]:
    calls = payload.get("functionCalls") or payload.get("function_calls")
    if isinstance(calls, list):
        _append_function_calls(payload, calls)

    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        payload["candidates"] = [
            _normalize_candidate(candidate) if isinstance(candidate, dict) else candidate
            for candidate in candidates
        ]

    if "usage_metadata" in payload and "usageMetadata" not in payload:
        payload["usageMetadata"] = payload.pop("usage_metadata")

    usage_metadata = payload.get("usageMetadata")
    if isinstance(usage_metadata, dict):
        if "candidates_token_count" in usage_metadata and "candidatesTokenCount" not in usage_metadata:
            usage_metadata["candidatesTokenCount"] = usage_metadata.pop("candidates_token_count")
        if "prompt_token_count" in usage_metadata and "promptTokenCount" not in usage_metadata:
            usage_metadata["promptTokenCount"] = usage_metadata.pop("prompt_token_count")
        if "total_token_count" in usage_metadata and "totalTokenCount" not in usage_metadata:
            usage_metadata["totalTokenCount"] = usage_metadata.pop("total_token_count")

    return payload


def parse_gemini_response(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_gemini_response(payload)
    parsed = GeminiGenerateContentResponse.model_validate(normalized)
    return parsed.model_dump(exclude_none=True)


def parse_gemini_request(payload: dict[str, Any]) -> dict[str, Any]:
    parsed = GeminiGenerateContentRequest.model_validate(payload)
    return parsed.model_dump(exclude_none=True)
