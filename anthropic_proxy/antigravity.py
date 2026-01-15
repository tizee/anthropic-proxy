"""
Antigravity (Google Internal) authentication and request handling.
Handles authentication and request proxying for Antigravity API.
"""

import json
import logging
import secrets
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from .auth_provider import OAuthPKCEAuth, OAuthProviderConfig
from .gemini_converter import anthropic_to_gemini_request
from .gemini_types import parse_gemini_response
from .types import ClaudeMessagesRequest

logger = logging.getLogger(__name__)

# Constants from opencode-antigravity-auth.NoeFabris
ANTIGRAVITY_CLIENT_ID = "_decode("MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlcC5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ==")"
ANTIGRAVITY_CLIENT_SECRET = "_decode("R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNYQzR6NnFEQWY=")"
ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback"
ANTIGRAVITY_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
ANTIGRAVITY_TOKEN_URL = "https://oauth2.googleapis.com/token"

# ============================================================================
# Antigravity Endpoints
# ============================================================================
#
# Antigravity has three types of endpoints, each requiring different access:
#
# 1. PROD Endpoint (Subscription-based):
#    - cloudcode-pa.googleapis.com
#    - For individual Gemini Code Assist subscribers
#    - Requires: Personal Gemini Code Assist subscription
#    - Auth: OAuth with Gemini Code Assist scopes
#
# 2. DAILY Sandbox Endpoint (Enterprise/Commercial):
#    - daily-cloudcode-pa.sandbox.googleapis.com
#    - For business users with Cloud License
#    - Requires: Google Cloud license for enterprise
#    - Returns 403 "lack a Gemini Code Assist license" for personal accounts
#
# 3. AUTOPUSH Sandbox Endpoint (Enterprise/Commercial):
#    - autopush-cloudcode-pa.sandbox.googleapis.com
#    - For business users with Cloud License (staging/testing)
#    - Requires: Google Cloud license for enterprise
#    - Returns 403 "lack a Gemini Code Assist license" for personal accounts
#
# IMPORTANT: For personal Gemini Code Assist subscribers, ONLY the PROD endpoint works.
# The sandbox endpoints are for enterprise customers with Cloud Licenses.
#
# ============================================================================

ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_AUTOPUSH = "https://autopush-cloudcode-pa.sandbox.googleapis.com"

# Primary request endpoints - PROD first for personal subscribers
ANTIGRAVITY_ENDPOINTS = [
    ANTIGRAVITY_ENDPOINT_PROD,      # Personal Gemini Code Assist subscription
    ANTIGRAVITY_ENDPOINT_DAILY,     # Fallback: Enterprise sandbox (may return 403 for personal accounts)
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,  # Fallback: Enterprise sandbox (may return 403 for personal accounts)
]

# Endpoints for quota fetching
ANTIGRAVITY_QUOTA_ENDPOINTS = [
    ANTIGRAVITY_ENDPOINT_PROD,
    ANTIGRAVITY_ENDPOINT_DAILY,
]

# Preferred endpoint order for project resolution (prod first)
ANTIGRAVITY_LOAD_ENDPOINTS = [
    ANTIGRAVITY_ENDPOINT_PROD,
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,
]

# Default to PROD endpoint (subscription-based)
ANTIGRAVITY_ENDPOINT = ANTIGRAVITY_ENDPOINTS[0]

ANTIGRAVITY_SCOPES = [
  "https://www.googleapis.com/auth/cloud-platform",
  "https://www.googleapis.com/auth/userinfo.email",
  "https://www.googleapis.com/auth/userinfo.profile",
  "https://www.googleapis.com/auth/cclog",
  "https://www.googleapis.com/auth/experimentsandconfigs",
]

ANTIGRAVITY_HEADERS = {
  "User-Agent": "antigravity/1.11.5 windows/amd64",
  "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
  "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}

ANTIGRAVITY_QUOTA_HEADERS = {
    "User-Agent": "antigravity",
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip",
}

# Reference: Antigravity system instruction required by gateway.
ANTIGRAVITY_SYSTEM_INSTRUCTION = (
    "You are Antigravity, a powerful agentic AI coding assistant designed by the Google DeepMind team working on Advanced Agentic Coding.\n"
    "You are pair programming with a USER to solve their coding task.\n"
)

# Default models (subject to upstream changes).
# These model names can become invalid if the provider updates or disables support.
DEFAULT_ANTIGRAVITY_MODELS = {
    "claude-opus-4-5-thinking": {
        "model_name": "claude-opus-4-5-thinking",
        "description": "Claude Opus 4.5 Thinking (Antigravity)",
        "provider": "antigravity"
    },
    "claude-sonnet-4-5": {
        "model_name": "claude-sonnet-4-5",
        "description": "Claude Sonnet 4.5 (Antigravity)",
        "provider": "antigravity"
    },
    "claude-sonnet-4-5-thinking": {
        "model_name": "claude-sonnet-4-5-thinking",
        "description": "Claude Sonnet 4.5 Thinking (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-3-pro": {
        "model_name": "gemini-3-pro",
        "description": "Gemini 3 Pro (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-3-pro-low": {
        "model_name": "gemini-3-pro-low",
        "description": "Gemini 3 Pro Low (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-3-pro-high": {
        "model_name": "gemini-3-pro-high",
        "description": "Gemini 3 Pro High (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-3-pro-preview": {
        "model_name": "gemini-3-pro-preview",
        "description": "Gemini 3 Pro Preview (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-3-flash": {
        "model_name": "gemini-3-flash",
        "description": "Gemini 3 Flash (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-2.5-pro": {
        "model_name": "gemini-2.5-pro",
        "description": "Gemini 2.5 Pro (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-2.5-flash": {
        "model_name": "gemini-2.5-flash",
        "description": "Gemini 2.5 Flash (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-2.5-flash-lite": {
        "model_name": "gemini-2.5-flash-lite",
        "description": "Gemini 2.5 Flash Lite (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-2.5-flash-thinking": {
        "model_name": "gemini-2.5-flash-thinking",
        "description": "Gemini 2.5 Flash Thinking (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-2.0-flash-exp": {
        "model_name": "gemini-2.0-flash-exp",
        "description": "Gemini 2.0 Flash Exp (Antigravity)",
        "provider": "antigravity"
    },
    "gemini-3-pro-image": {
        "model_name": "gemini-3-pro-image",
        "description": "Gemini 3 Pro Image (Antigravity)",
        "provider": "antigravity"
    },
}

ANTIGRAVITY_THINKING_HEADER = "interleaved-thinking-2025-05-14"


def _is_claude_model(model_name: str) -> bool:
    return "claude" in model_name.lower()


def _is_thinking_model(model_name: str) -> bool:
    return "thinking" in model_name.lower()


def _infer_thinking_budget(model_name: str) -> int | None:
    if not _is_thinking_model(model_name):
        return None
    lower = model_name.lower()
    if "-low" in lower:
        return 8192
    if "-high" in lower:
        return 32768
    if "-medium" in lower:
        return 16384
    return 16384


def _is_image_model(model_name: str) -> bool:
    return "image" in model_name.lower()


def _ensure_system_instruction(body: dict[str, Any], identity: str, inject_identity: bool) -> None:
    system_instruction = body.get("systemInstruction")
    if system_instruction is None:
        if not inject_identity:
            return
        system_instruction = {"parts": []}
        body["systemInstruction"] = system_instruction

    if not isinstance(system_instruction, dict):
        system_instruction = {"parts": []}
        body["systemInstruction"] = system_instruction

    parts = system_instruction.get("parts")
    if not isinstance(parts, list):
        parts = []
        system_instruction["parts"] = parts

    if inject_identity:
        already_present = any(
            isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and "You are Antigravity" in part.get("text", "")
            for part in parts
        )
        if not already_present:
            parts.insert(0, {"text": identity})

    system_instruction["role"] = "user"


def _extract_project_id(payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return None
    project = payload.get("cloudaicompanionProject")
    if isinstance(project, str) and project:
        return project
    if isinstance(project, dict):
        project_id = project.get("id")
        if isinstance(project_id, str) and project_id:
            return project_id
    return None


def _is_capacity_error(body_text: str) -> bool:
    if not body_text:
        return False
    return "No capacity available" in body_text or "RESOURCE_EXHAUSTED" in body_text


async def fetch_antigravity_quota_models(
    access_token: str,
    project_id: str,
    timeout: float = 10.0,
) -> tuple[dict[str, Any], str]:
    """Fetch available model quota data using Antigravity OAuth token."""
    if not project_id:
        raise HTTPException(
            status_code=400,
            detail="Antigravity Project ID is missing. Re-login required.",
        )
    payload = {"project": project_id}
    last_error: str | None = None

    async with httpx.AsyncClient() as http_client:
        for endpoint in ANTIGRAVITY_QUOTA_ENDPOINTS:
            url = f"{endpoint}/v1internal:fetchAvailableModels"
            try:
                response = await http_client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        **ANTIGRAVITY_QUOTA_HEADERS,
                    },
                    json=payload,
                    timeout=timeout,
                )
            except httpx.HTTPError as exc:
                last_error = str(exc)
                continue

            body_text = response.text
            if response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Antigravity quota auth expired. Run anthropic-proxy login --antigravity.",
                )
            if response.status_code == 403:
                raise HTTPException(
                    status_code=403,
                    detail="Antigravity quota access forbidden (403). Check subscription or project.",
                )
            if not response.is_success:
                detail = f"{response.status_code} {response.reason_phrase}"
                if body_text:
                    detail = f"{detail}: {body_text[:1000]}"
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = detail
                    continue
                raise HTTPException(status_code=response.status_code, detail=detail)

            if not body_text:
                raise HTTPException(status_code=502, detail="Antigravity quota response was empty.")
            try:
                data = response.json()
            except ValueError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Antigravity quota response parse failed: {exc}",
                ) from exc

            if not isinstance(data, dict):
                raise HTTPException(
                    status_code=502,
                    detail="Antigravity quota response format invalid.",
                )
            return data, endpoint

    raise HTTPException(
        status_code=502,
        detail=f"Antigravity quota fetch failed after fallbacks: {last_error or 'unknown error'}",
    )


def _extract_thinking_budget(request: ClaudeMessagesRequest, model_name: str) -> int | None:
    if request.thinking and request.thinking.type == "enabled":
        if request.thinking.budget_tokens is not None:
            return request.thinking.budget_tokens
    return _infer_thinking_budget(model_name)


def _build_antigravity_request(
    request: ClaudeMessagesRequest,
    model_name: str,
    session_id: str,
) -> tuple[dict[str, Any], bool]:
    body = anthropic_to_gemini_request(
        request,
        model_name,
        is_antigravity=True,
        system_prefix=None,
        session_id=session_id,
    )

    is_claude = _is_claude_model(model_name)
    is_thinking = _is_thinking_model(model_name)
    thinking_budget = _extract_thinking_budget(request, model_name)

    if is_claude and is_thinking and thinking_budget:
        generation_config = body.get("generationConfig") or {}
        generation_config["thinkingConfig"] = {
            "include_thoughts": True,
            "thinking_budget": thinking_budget,
        }
        max_tokens = generation_config.get(
            "maxOutputTokens",
            generation_config.get("max_output_tokens", 0),
        )
        if max_tokens <= thinking_budget:
            generation_config["maxOutputTokens"] = thinking_budget + 8192
        body["generationConfig"] = generation_config

    inject_identity = not _is_image_model(model_name)
    _ensure_system_instruction(
        body,
        ANTIGRAVITY_SYSTEM_INSTRUCTION,
        inject_identity,
    )

    return body, bool(is_claude and is_thinking and thinking_budget)


async def _stream_antigravity_internal(
    *,
    envelope: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 60.0,
) -> AsyncGenerator[dict[str, Any], None]:
    last_error = None
    async with httpx.AsyncClient() as http_client:
        for endpoint in ANTIGRAVITY_ENDPOINTS:
            url = f"{endpoint}/v1internal:streamGenerateContent"
            try:
                async with http_client.stream(
                    "POST",
                    url,
                    headers=headers,
                    params={"alt": "sse"},
                    json=envelope,
                    timeout=timeout,
                ) as response:
                    if response.status_code >= 400:
                        body_bytes = await response.aread()
                        body_text = body_bytes.decode("utf-8", errors="replace")
                        detail = f"{response.status_code} {response.reason_phrase}"
                        if body_text:
                            detail = f"{detail}: {body_text[:1000]}"

                        if (
                            response.status_code in {403, 404, 429, 500, 503, 529}
                            or response.status_code >= 500
                            or _is_capacity_error(body_text)
                        ):
                            logger.warning(
                                "Antigravity endpoint %s returned %s; trying fallback.",
                                endpoint,
                                detail,
                            )
                            last_error = detail
                            continue

                        raise HTTPException(
                            status_code=502,
                            detail=f"Antigravity error: {detail}",
                        )

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = line[len("data:") :].strip() if line.startswith("data:") else line.strip()
                        if not data or data == "[DONE]":
                            continue
                        logger.debug("Antigravity raw chunk: %s", data[:1000])
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(payload, dict) and "response" in payload:
                            payload = payload["response"]
                        if not isinstance(payload, dict):
                            continue
                        yield parse_gemini_response(payload)
                    return
            except httpx.HTTPError as exc:
                last_error = str(exc)
                logger.warning(
                    "Antigravity endpoint %s failed: %s",
                    endpoint,
                    last_error,
                )
                continue

    raise HTTPException(
        status_code=502,
        detail=f"Antigravity request failed after fallbacks: {last_error or 'unknown error'}",
    )
def _extract_session_id(request: ClaudeMessagesRequest) -> str | None:
    metadata = request.metadata
    if not isinstance(metadata, dict):
        return None
    for key in ("session_id", "sessionId", "conversation_id", "conversationId", "thread_id", "threadId"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


class AntigravityAuth(OAuthPKCEAuth):
    """Manages Antigravity authentication tokens."""

    def __init__(self):
        super().__init__(
            OAuthProviderConfig(
                provider_key="antigravity",
                display_name="Antigravity",
                client_id=ANTIGRAVITY_CLIENT_ID,
                client_secret=ANTIGRAVITY_CLIENT_SECRET,
                auth_url=ANTIGRAVITY_AUTH_URL,
                token_url=ANTIGRAVITY_TOKEN_URL,
                redirect_uri=ANTIGRAVITY_REDIRECT_URI,
                callback_path="/oauth-callback",
                port=51121,
                scopes=ANTIGRAVITY_SCOPES,
                auth_params={
                    "access_type": "offline",
                    "prompt": "consent",
                },
            )
        )

    def _extract_refresh_token(self, refresh_full: str | None) -> str | None:
        if not refresh_full:
            return None
        return refresh_full.split("|", 1)[0]

    def _split_refresh(self) -> tuple[str, str]:
        refresh_full = self._auth_data.get("refresh", "")
        if not refresh_full:
            return "", ""
        if "|" in refresh_full:
            token, project_id = refresh_full.split("|", 1)
            return token, project_id
        return refresh_full, ""

    def _post_login(self) -> None:
        print("Resolving Antigravity project...")
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                loop.create_task(self.verify_quota_access())
                return

            project_id = asyncio.run(self.verify_quota_access())
        except Exception as exc:
            self._auth_data = {}
            self._save()
            print(f"Antigravity login failed: {exc}")
            raise SystemExit(1) from None
        print(f"Resolved Antigravity project ID: {project_id}")

    def get_project_id(self) -> str | None:
        stored = self._auth_data.get("project_id")
        if isinstance(stored, str) and stored:
            return stored
        _, project_id = self._split_refresh()
        return project_id or None

    async def ensure_project_context(self, force: bool = False) -> None:
        """Resolve and store project ID if missing or force refresh."""
        self._load()
        refresh_token, project_id = self._split_refresh()
        if not refresh_token:
            return

        if project_id and not force:
            return

        access_token = self._auth_data.get("access")
        if not access_token:
            try:
                access_token = await self.get_access_token()
            except Exception as exc:
                logger.error(f"Failed to refresh access token for project lookup: {exc}")
                return

        try:
            async with httpx.AsyncClient() as client:
                req_body = {
                    "metadata": {
                        "ideType": "IDE_UNSPECIFIED",
                        "platform": "PLATFORM_UNSPECIFIED",
                        "pluginType": "GEMINI",
                    }
                }
                load_endpoints = list(dict.fromkeys(ANTIGRAVITY_LOAD_ENDPOINTS + ANTIGRAVITY_ENDPOINTS))
                for endpoint in load_endpoints:
                    res = await client.post(
                        f"{endpoint}/v1internal:loadCodeAssist",
                        json=req_body,
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            **ANTIGRAVITY_HEADERS,
                        },
                    )

                    if res.status_code != 200:
                        continue
                    data = res.json()
                    found_id = _extract_project_id(data)
                    if found_id:
                        if found_id != project_id or not self._auth_data.get("project_id"):
                            self._update_refresh_with_project(refresh_token, found_id)
                        return

                req_body["tierId"] = "FREE"
                for endpoint in ANTIGRAVITY_ENDPOINTS:
                    res = await client.post(
                        f"{endpoint}/v1internal:onboardUser",
                        json=req_body,
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            **ANTIGRAVITY_HEADERS,
                        },
                    )
                    if res.status_code == 200:
                        data = res.json()
                        new_id = (
                            data.get("response", {})
                            .get("cloudaicompanionProject", {})
                            .get("id")
                        )
                        if new_id:
                            self._update_refresh_with_project(refresh_token, new_id)
                            return

        except Exception as exc:
            logger.error(f"Failed to ensure project context: {exc}")

    async def verify_quota_access(self) -> str:
        """Ensure project id exists and quota can be fetched."""
        access_token = await self.get_access_token()
        await self.ensure_project_context(force=True)
        project_id = self.get_project_id()
        if not project_id:
            raise RuntimeError(
                "Antigravity Project ID could not be resolved. Re-login required."
            )
        await fetch_antigravity_quota_models(access_token, project_id)
        return project_id

    def _update_refresh_with_project(self, token: str, proj: str) -> None:
        new_refresh = f"{token}|{proj}"
        self._auth_data["refresh"] = new_refresh
        self._auth_data["project_id"] = proj
        self._save()
        logger.info(f"Updated Antigravity project context: {proj}")


antigravity_auth = AntigravityAuth()


async def handle_antigravity_request(
    request: ClaudeMessagesRequest,
    model_id: str,
    *,
    model_name: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Handle request to Antigravity backend, returning a generator of Gemini response chunks.
    """
    access_token = await antigravity_auth.get_access_token()
    project_id = antigravity_auth.get_project_id()

    if not project_id:
        await antigravity_auth.ensure_project_context()
        project_id = antigravity_auth.get_project_id()
        if not project_id:
            raise HTTPException(
                status_code=400,
                detail="Antigravity Project ID could not be resolved. Re-login required.",
            )

    session_id = _extract_session_id(request)
    if not session_id:
        session_id = f"antigravity-{secrets.token_hex(8)}"
        if request.metadata is None or not isinstance(request.metadata, dict):
            request.metadata = {}
        request.metadata.setdefault("session_id", session_id)

    target_model = model_name or model_id
    body, needs_thinking_header = _build_antigravity_request(
        request,
        target_model,
        session_id,
    )

    envelope = {
        "project": project_id,
        "model": target_model,
        "request": body,
        "userAgent": "antigravity",
        "requestType": "agent",
        "requestId": f"agent-{secrets.token_hex(8)}",
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        **ANTIGRAVITY_HEADERS,
    }
    if needs_thinking_header:
        headers["anthropic-beta"] = ANTIGRAVITY_THINKING_HEADER

    async for chunk in _stream_antigravity_internal(
        envelope=envelope,
        headers=headers,
    ):
        yield chunk
