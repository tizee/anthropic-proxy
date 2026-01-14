"""Gemini SDK request handling for Gemini/Antigravity providers."""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from fastapi import HTTPException
from google import genai
from google.genai import types

from .gemini_converter import anthropic_to_gemini_sdk_params
from .types import ClaudeMessagesRequest

logger = logging.getLogger(__name__)


def _build_http_options(
    *,
    base_url: str | None,
    headers: dict[str, str] | None,
    extra_body: dict[str, Any] | None,
    api_version: str | None,
) -> Any:
    options: dict[str, Any] = {}
    if base_url:
        options["base_url"] = base_url
    if headers:
        options["headers"] = headers
    if extra_body:
        options["extra_body"] = extra_body
    if api_version:
        options["api_version"] = api_version
    try:
        return types.HttpOptions(**options)
    except Exception:
        return options


def _coerce_tools(tools: list[dict[str, Any]]) -> list[Any]:
    converted: list[Any] = []
    for tool in tools:
        decls = tool.get("function_declarations", [])
        try:
            function_decls = [
                types.FunctionDeclaration(
                    name=decl.get("name", ""),
                    description=decl.get("description", ""),
                    parameters_json_schema=decl.get("parameters_json_schema", {}),
                )
                for decl in decls
            ]
            converted.append(types.Tool(function_declarations=function_decls))
        except Exception:
            converted.append(tool)
    return converted


def _coerce_chunk_dict(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return chunk
    if hasattr(chunk, "model_dump"):
        return chunk.model_dump(exclude_none=True)
    if hasattr(chunk, "dict"):
        return chunk.dict(exclude_none=True)
    if hasattr(chunk, "to_dict"):
        return chunk.to_dict()
    if hasattr(chunk, "__dict__"):
        return dict(chunk.__dict__)
    return {"text": getattr(chunk, "text", "")}


async def stream_gemini_sdk_request(
    *,
    request: ClaudeMessagesRequest,
    model_id: str,
    access_token: str,
    project_id: str | None,
    base_url: str,
    extra_headers: dict[str, str],
    is_antigravity: bool,
    system_prefix: str | None = None,
    request_envelope_extra: dict[str, Any] | None = None,
    use_request_envelope: bool = True,
    api_version: str | None = "v1internal",
) -> AsyncGenerator[dict[str, Any], None]:
    contents, config, raw_body = anthropic_to_gemini_sdk_params(
        request,
        model_id,
        is_antigravity=is_antigravity,
        system_prefix=system_prefix,
    )

    if "tools" in config:
        config["tools"] = _coerce_tools(config["tools"])

    extra_body = None
    if use_request_envelope:
        envelope: dict[str, Any] = {"request": raw_body}
        if project_id:
            envelope["project"] = project_id
        if model_id:
            envelope["model"] = model_id
        if request_envelope_extra:
            envelope.update(request_envelope_extra)
        extra_body = envelope

    headers = {"Authorization": f"Bearer {access_token}", **extra_headers}
    http_options = _build_http_options(
        base_url=base_url,
        headers=headers,
        extra_body=extra_body,
        api_version=api_version,
    )

    client = genai.Client(http_options=http_options)
    aclient = client.aio

    try:
        async for chunk in await aclient.models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=config,
        ):
            yield _coerce_chunk_dict(chunk)
    except Exception as exc:
        logger.error(f"Gemini SDK request failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Gemini SDK error: {exc}") from exc
    finally:
        try:
            await aclient.aclose()
        except Exception:
            pass
