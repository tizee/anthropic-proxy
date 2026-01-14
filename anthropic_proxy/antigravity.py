"""
Antigravity (Google Internal) authentication and request handling.
Handles authentication and request proxying for Antigravity API.
"""

import logging
import secrets
from collections.abc import AsyncGenerator
from typing import Any

import httpx

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

# Default models
DEFAULT_ANTIGRAVITY_MODELS = {
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
    "gemini-3-pro-high": {
        "model_name": "gemini-3-pro-high",
        "description": "Gemini 3 Pro High (Antigravity)",
        "provider": "antigravity"
    },
    "gpt-oss-120b-medium": {
        "model_name": "gpt-oss-120b-medium",
        "description": "GPT-OSS 120B Medium (Antigravity)",
        "provider": "antigravity"
    }
}

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
        return refresh_full.split("|")[0]

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
        refresh = self._auth_data.get("refresh", "")
        parts = refresh.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        return None

    async def ensure_project_context(self) -> None:
        """Resolve and store project ID if missing."""
        self._load()
        refresh_full = self._auth_data.get("refresh", "")
        if not refresh_full:
            return

        parts = refresh_full.split("|")
        refresh_token = parts[0]
        project_id = parts[1] if len(parts) > 1 else ""

        if project_id:
            return

        access_token = self._auth_data.get("access")
        if not access_token:
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
                    else:
                        self._update_refresh_with_project(
                            refresh_token, "rising-fact-p41fc"
                        )

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
             # Fallback to default if still missing, or error
             project_id = "rising-fact-p41fc"

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
    ):
        yield chunk
