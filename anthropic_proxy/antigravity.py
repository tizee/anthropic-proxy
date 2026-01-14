"""
Antigravity (Google Internal) authentication and request handling.
Handles authentication and request proxying for Antigravity API.
"""

import logging
import secrets
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from .auth_provider import OAuthPKCEAuth, OAuthProviderConfig
from .gemini_sdk import stream_gemini_sdk_request
from .types import ClaudeMessagesRequest

logger = logging.getLogger(__name__)

# Constants from opencode-antigravity-auth.NoeFabris
ANTIGRAVITY_CLIENT_ID = "_decode("MTA3MTAwNjA2MDU5MS10bWhzc2luMmgyMWxjcmUyMzV2dG9sb2poNGc0MDNlcC5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbQ==")"
ANTIGRAVITY_CLIENT_SECRET = "_decode("R09DU1BYLUs1OEZXUjQ4NkxkTEoxbUxCOHNYQzR6NnFEQWY=")"
ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback"
ANTIGRAVITY_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
ANTIGRAVITY_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Endpoints
ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com"
# Default to Daily Sandbox as per reference implementation
ANTIGRAVITY_ENDPOINT = ANTIGRAVITY_ENDPOINT_DAILY

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

            asyncio.run(self.ensure_project_context())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.ensure_project_context())
            else:
                loop.run_until_complete(self.ensure_project_context())
        print("Project configuration resolved.")

    def get_project_id(self) -> str | None:
        _, project_id = self._split_refresh()
        return project_id or None

    async def ensure_project_context(self) -> None:
        """Resolve and store project ID if missing."""
        self._load()
        refresh_token, project_id = self._split_refresh()
        if not refresh_token:
            return

        if project_id:
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
                res = await client.post(
                    f"{ANTIGRAVITY_ENDPOINT}/v1internal:loadCodeAssist",
                    json=req_body,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        **ANTIGRAVITY_HEADERS,
                    },
                )

                if res.status_code == 200:
                    data = res.json()
                    found_id = data.get("cloudaicompanionProject")
                    if found_id:
                        self._update_refresh_with_project(refresh_token, found_id)
                        return

                req_body["tierId"] = "FREE"
                res = await client.post(
                    f"{ANTIGRAVITY_ENDPOINT}/v1internal:onboardUser",
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

        except Exception as exc:
            logger.error(f"Failed to ensure project context: {exc}")

    def _update_refresh_with_project(self, token: str, proj: str) -> None:
        new_refresh = f"{token}|{proj}"
        self._auth_data["refresh"] = new_refresh
        self._save()
        logger.info(f"Updated Antigravity project context: {proj}")


antigravity_auth = AntigravityAuth()


async def handle_antigravity_request(
    request: ClaudeMessagesRequest,
    model_id: str,
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
                detail="Antigravity Project ID could not be resolved. Try re-login.",
            )

    session_id = _extract_session_id(request)

    async for chunk in stream_gemini_sdk_request(
        request=request,
        model_id=model_id,
        access_token=access_token,
        project_id=project_id,
        base_url=ANTIGRAVITY_ENDPOINT,
        extra_headers=ANTIGRAVITY_HEADERS,
        is_antigravity=True,
        system_prefix=ANTIGRAVITY_SYSTEM_INSTRUCTION,
        request_envelope_extra={
            "userAgent": "antigravity",
            "requestType": "agent",
            "requestId": f"agent-{secrets.token_hex(8)}",
        },
        session_id=session_id,
    ):
        yield chunk
