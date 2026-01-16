"""Gemini SDK request handling for Gemini/Antigravity providers."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

import httpx
from fastapi import HTTPException
from google import genai
from google.genai import types

from .converters import anthropic_to_gemini_sdk_params
from .gemini_types import parse_gemini_response
from .types import ClaudeMessagesRequest

logger = logging.getLogger(__name__)


def _build_http_options(
    *,
    base_url: str | None,
    headers: dict[str, str] | None,
    extra_body: dict[str, Any] | None,
    api_version: str | None,
    base_url_resource_scope: types.ResourceScope | None = None,
    embed_api_version_in_base_url: bool = False,
) -> Any:
    options: dict[str, Any] = {}
    if base_url:
        if embed_api_version_in_base_url and api_version:
            options["base_url"] = f"{base_url.rstrip('/')}/{api_version}"
            api_version = None
        else:
            options["base_url"] = base_url
    if base_url_resource_scope:
        options["base_url_resource_scope"] = base_url_resource_scope
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
    use_vertexai: bool = False,
    use_code_assist: bool = False,
    system_prefix: str | None = None,
    request_envelope_extra: dict[str, Any] | None = None,
    use_request_envelope: bool = True,
    api_version: str | None = "v1internal",
    session_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    contents, config, raw_body = anthropic_to_gemini_sdk_params(
        request,
        model_id,
        is_antigravity=is_antigravity,
        system_prefix=system_prefix,
        session_id=session_id,
    )

    if "tools" in config:
        logger.debug(
            "Gemini SDK tools (pre-coerce): %d",
            len(config.get("tools") or []),
        )
        config["tools"] = _coerce_tools(config["tools"])
        logger.debug(
            "Gemini SDK tools (post-coerce): %d",
            len(config.get("tools") or []),
        )

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

    if use_code_assist:
        headers.setdefault("Accept", "text/event-stream")
        headers.setdefault("Content-Type", "application/json")

        if raw_body.get("tools") and "toolConfig" not in raw_body:
            raw_body["toolConfig"] = {"functionCallingConfig": {"mode": "VALIDATED"}}
        if raw_body.get("tools") is not None:
            logger.debug(
                "Code Assist tools (raw): %d",
                len(raw_body.get("tools") or []),
            )

        envelope: dict[str, Any] = {"request": raw_body}
        if project_id:
            envelope["project"] = project_id
        if model_id:
            envelope["model"] = model_id
        if request_envelope_extra:
            envelope.update(request_envelope_extra)

        url = f"{base_url.rstrip('/')}/v1internal:streamGenerateContent"
        try:
            # Use separate timeouts: short connect timeout, long read timeout for streaming
            timeout_config = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient() as http_client:
                async with http_client.stream(
                    "POST",
                    url,
                    headers=headers,
                    params={"alt": "sse"},
                    json=envelope,
                    timeout=timeout_config,
                ) as response:
                    is_error = getattr(
                        response,
                        "is_error",
                        response.status_code >= 400,
                    )
                    if is_error:
                        body_bytes = await response.aread()
                        body_text = body_bytes.decode("utf-8", errors="replace")
                        # Extract retry-after delay from 429 errors if available
                        retry_after = ""
                        if response.status_code == 429:
                            try:
                                error_data = json.loads(body_text)
                                if isinstance(error_data, dict):
                                    error = error_data.get("error", {})
                                    if isinstance(error, dict):
                                        details = error.get("details", [])
                                        for detail in details:
                                            if isinstance(detail, dict):
                                                retry_info = detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo"
                                                if retry_info and detail.get("retryDelay"):
                                                    retry_after = f" Retry after: {detail['retryDelay']}"
                                    elif error.get("status") == "RESOURCE_EXHAUSTED":
                                        # Also check for quota reset info
                                        for detail in error.get("details", []):
                                            if detail.get("reason") == "RATE_LIMIT_EXCEEDED":
                                                metadata = detail.get("metadata", {})
                                                if metadata.get("quotaResetTimeStamp"):
                                                    retry_after = f" Quota resets at: {metadata['quotaResetTimeStamp']}"
                            except (json.JSONDecodeError, KeyError, TypeError):
                                pass

                        logger.error(
                            "Gemini Code Assist error %s:%s %s",
                            response.status_code,
                            retry_after,
                            body_text[:1000],
                        )
                        detail = f"{response.status_code} {response.reason_phrase}"
                        if body_text:
                            detail = f"{detail}: {body_text[:1000]}"
                        raise HTTPException(
                            status_code=502, detail=f"Gemini Code Assist error: {detail}"
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data = line[len("data:") :].strip()
                        else:
                            data = line.strip()
                        if not data or data == "[DONE]":
                            continue
                        logger.debug(
                            "Code Assist raw chunk: %s",
                            data[:1000],
                        )
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(payload, dict) and "response" in payload:
                            payload = payload["response"]
                        if not isinstance(payload, dict):
                            continue
                        yield parse_gemini_response(payload)
        except httpx.HTTPStatusError as exc:
            detail = f"{exc.response.status_code} {exc.response.reason_phrase}"
            logger.error(f"Gemini Code Assist request failed: {detail}")
            raise HTTPException(status_code=502, detail=f"Gemini Code Assist error: {detail}") from exc
        except Exception as exc:
            logger.error(f"Gemini Code Assist request failed: {exc}")
            raise HTTPException(status_code=502, detail=f"Gemini Code Assist error: {exc}") from exc
        return

    http_options = _build_http_options(
        base_url=base_url,
        headers=headers,
        extra_body=extra_body,
        api_version=api_version,
        base_url_resource_scope=(
            types.ResourceScope.COLLECTION if use_vertexai else None
        ),
        embed_api_version_in_base_url=bool(use_vertexai),
    )

    client = genai.Client(vertexai=use_vertexai, http_options=http_options)
    aclient = client.aio

    try:
        async for chunk in await aclient.models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=config,
        ):
            yield parse_gemini_response(_coerce_chunk_dict(chunk))
    except Exception as exc:
        logger.error(f"Gemini SDK request failed: {exc}")
        raise HTTPException(status_code=502, detail=f"Gemini SDK error: {exc}") from exc
    finally:
        try:
            await aclient.aclose()
        except Exception:
            pass
