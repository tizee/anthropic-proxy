"""
Gemini CLI (Google) subscription support.
Handles authentication and request proxying for Gemini plan.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from .auth_provider import OAuthPKCEAuth, OAuthProviderConfig
from .gemini_sdk import stream_gemini_sdk_request
from .types import ClaudeMessagesRequest

logger = logging.getLogger(__name__)

# Constants
GEMINI_CLIENT_ID = "_decode("NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVqLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29t")"
GEMINI_CLIENT_SECRET = "_decode("R09DU1BYLTR1SGdNUG0tMW83U2stZ2VWNkN1NWNsWEZzeGw=")"
GEMINI_REDIRECT_URI = "http://localhost:8085/oauth2callback"
GEMINI_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GEMINI_TOKEN_URL = "https://oauth2.googleapis.com/token"
GEMINI_CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"

GEMINI_SCOPES = [
  "https://www.googleapis.com/auth/cloud-platform",
  "https://www.googleapis.com/auth/userinfo.email",
  "https://www.googleapis.com/auth/userinfo.profile",
]

CODE_ASSIST_HEADERS = {
  "User-Agent": "google-api-nodejs-client/9.15.1",
  "X-Goog-Api-Client": "gl-node/22.17.0",
  "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
}

# Default models (subject to upstream changes).
# These model names can become invalid if the provider updates or disables support.
DEFAULT_GEMINI_MODELS = {
    "gemini-3-pro-preview": {
        "model_name": "gemini-3-pro-preview",
        "description": "Gemini 3 Pro Preview",
    },
    "gemini-3-flash-preview": {
        "model_name": "gemini-3-flash-preview",
        "description": "Gemini 3 Flash Preview",
    },
    "gemini-2.5-pro": {
        "model_name": "gemini-2.5-pro",
        "description": "Gemini 2.5 Pro",
    },
    "gemini-2.5-flash": {
        "model_name": "gemini-2.5-flash",
        "description": "Gemini 2.5 Flash",
    },
    "gemini-2.5-flash-lite": {
        "model_name": "gemini-2.5-flash-lite",
        "description": "Gemini 2.5 Flash Lite",
    },
}

class GeminiAuth(OAuthPKCEAuth):
    """Manages Gemini authentication tokens."""

    def __init__(self):
        super().__init__(
            OAuthProviderConfig(
                provider_key="google",
                display_name="Gemini",
                client_id=GEMINI_CLIENT_ID,
                client_secret=GEMINI_CLIENT_SECRET,
                auth_url=GEMINI_AUTH_URL,
                token_url=GEMINI_TOKEN_URL,
                redirect_uri=GEMINI_REDIRECT_URI,
                callback_path="/oauth2callback",
                port=8085,
                scopes=GEMINI_SCOPES,
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
        print("Resolving Google Cloud project...")
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
        if len(parts) >= 3 and parts[2]:
            return parts[2]
        if len(parts) >= 2 and parts[1]:
            return parts[1]
        return None

    async def ensure_project_context(self):
        """Resolve and store project ID if missing."""
        self._load()
        refresh_full = self._auth_data.get("refresh", "")
        if not refresh_full:
            return

        parts = refresh_full.split("|")
        refresh_token = parts[0]
        project_id = parts[1] if len(parts) > 1 else ""
        managed_id = parts[2] if len(parts) > 2 else ""

        if managed_id or project_id:
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
                    f"{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                    json=req_body,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        **CODE_ASSIST_HEADERS
                    }
                )

                if res.status_code == 200:
                    data = res.json()
                    found_id = data.get("cloudaicompanionProject")
                    if found_id:
                        self._update_refresh_with_project(refresh_token, project_id, found_id)
                        return

                req_body["tierId"] = "FREE"
                res = await client.post(
                    f"{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                    json=req_body,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        **CODE_ASSIST_HEADERS
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    new_id = data.get("response", {}).get("cloudaicompanionProject", {}).get("id")
                    if new_id:
                        self._update_refresh_with_project(refresh_token, project_id, new_id)

        except Exception as e:
            logger.error(f"Failed to ensure project context: {e}")

    def _update_refresh_with_project(self, token, proj, managed):
        new_refresh = f"{token}|{proj}|{managed}"
        self._auth_data["refresh"] = new_refresh
        self._save()
        logger.info(f"Updated Gemini project context: {managed}")


gemini_auth = GeminiAuth()


async def handle_gemini_request(
    request: ClaudeMessagesRequest,
    model_id: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Handle request to Gemini backend, returning a generator of Gemini response chunks.
    """
    access_token = await gemini_auth.get_access_token()
    project_id = gemini_auth.get_project_id()

    if not project_id:
        await gemini_auth.ensure_project_context()
        project_id = gemini_auth.get_project_id()
        if not project_id:
             raise HTTPException(status_code=400, detail="Gemini Project ID could not be resolved. Try re-login.")

    async for chunk in stream_gemini_sdk_request(
        request=request,
        model_id=model_id,
        access_token=access_token,
        project_id=project_id,
        base_url=GEMINI_CODE_ASSIST_ENDPOINT,
        extra_headers=CODE_ASSIST_HEADERS,
        is_antigravity=False,
    ):
        yield chunk
