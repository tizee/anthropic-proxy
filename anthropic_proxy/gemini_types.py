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
    partialArgs: list[Any] | None = None
    willContinue: bool | None = None

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
    # Expect Vertex API camelCase field names in streaming responses.

    call = part.get("functionCall")
    if isinstance(call, dict):
        args = _coerce_args(call.get("args"))
        call["args"] = args

    response = part.get("functionResponse")
    if isinstance(response, dict):
        pass

    return part


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
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


def normalize_gemini_response(payload: dict[str, Any]) -> dict[str, Any]:
    # Vertex AI response format is camelCase; we only normalize enum values and args types.
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        payload["candidates"] = [
            _normalize_candidate(candidate)
            if isinstance(candidate, dict)
            else candidate
            for candidate in candidates
        ]

    return payload


def parse_gemini_response(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_gemini_response(payload)
    parsed = GeminiGenerateContentResponse.model_validate(normalized)
    return parsed.model_dump(exclude_none=True)


def parse_gemini_request(payload: dict[str, Any]) -> dict[str, Any]:
    parsed = GeminiGenerateContentRequest.model_validate(payload)
    return parsed.model_dump(exclude_none=True)
