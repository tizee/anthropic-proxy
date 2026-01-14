"""
Codex subscription support.
Handles authentication and request proxying for Codex plan.
"""

import base64
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from .auth_provider import OAuthPKCEAuth, OAuthProviderConfig

logger = logging.getLogger(__name__)

# Constants
CODEX_AUTH_URL = "https://auth.openai.com/oauth/token"
CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CODEX_API_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"

# Default models available via Codex subscription (from AUTH.md)
DEFAULT_CODEX_MODELS = {
    "gpt-5.2-codex": {
        "model_name": "gpt-5.2-codex",
        "description": "Newest flagship Codex model",
        "reasoning_effort": "high"
    },
    "gpt-5.1-codex": {
        "model_name": "gpt-5.1-codex",
        "description": "Standard coding-specialized model",
        "reasoning_effort": "medium"
    },
    "gpt-5.1-codex-max": {
        "model_name": "gpt-5.1-codex-max",
        "description": "High-performance variant",
        "reasoning_effort": "high"
    },
    "gpt-5.1-codex-mini": {
        "model_name": "gpt-5.1-codex-mini",
        "description": "Fast, lightweight coding model",
        "reasoning_effort": "medium"
    },
    "gpt-5.2": {
        "model_name": "gpt-5.2",
        "description": "General purpose flagship",
        "reasoning_effort": "high"
    },
    "gpt-5.1": {
        "model_name": "gpt-5.1",
        "description": "General purpose standard",
        "reasoning_effort": "medium"
    },
    # Legacy/Alias mappings
    "gpt-5-codex": {
        "model_name": "gpt-5.1-codex",
        "description": "Alias for gpt-5.1-codex",
        "reasoning_effort": "medium"
    }
}


class CodexAuth(OAuthPKCEAuth):
    """Manages Codex authentication tokens."""

    def __init__(self):
        super().__init__(
            OAuthProviderConfig(
                provider_key="openai",
                display_name="Codex",
                client_id=CODEX_CLIENT_ID,
                client_secret=None,
                auth_url=CODEX_AUTHORIZE_URL,
                token_url=CODEX_AUTH_URL,
                redirect_uri=CODEX_REDIRECT_URI,
                callback_path="/auth/callback",
                port=1455,
                auth_params={
                    "scope": "openid profile email offline_access",
                    "id_token_add_organizations": "true",
                    "codex_cli_simplified_flow": "true",
                    "originator": "codex_cli_rs",
                },
                include_redirect_uri_on_refresh=True,
            )
        )

    def _extract_account_id(self, token: str) -> str | None:
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return None

            payload_b64 = parts[1]
            payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)

            payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
            payload = json.loads(payload_json)

            auth_claim = payload.get("https://api.openai.com/auth", {})
            return auth_claim.get("chatgpt_account_id")
        except Exception as exc:
            logger.debug(f"Error extracting account ID: {exc}")
            return None

    def _post_token_exchange(self, data: dict[str, Any]) -> None:
        try:
            account_id = self._extract_account_id(data["access_token"])
            if account_id:
                self._auth_data["accountId"] = account_id
        except Exception as exc:
            logger.warning(f"Failed to extract account ID from token: {exc}")

    async def _refresh_token(self, refresh_token_raw: str, refresh_full: str | None) -> str:
        new_access = await super()._refresh_token(refresh_token_raw, None)
        try:
            account_id = self._extract_account_id(new_access)
            if account_id:
                self._auth_data["accountId"] = account_id
                self._save()
        except Exception as exc:
            logger.warning(f"Failed to extract account ID from token: {exc}")
        return new_access

    def get_account_id(self) -> str | None:
        return self._auth_data.get("accountId")


# Global auth instance
codex_auth = CodexAuth()

async def handle_codex_request(openai_request: dict, model_id: str) -> AsyncGenerator[dict[str, Any], None]:
    """
    Handle a request to the Codex backend.
    
    Args:
        openai_request: The request body in OpenAI format.
        model_id: The original model ID (for logging).
        
    Returns:
        AsyncGenerator: Generator yielding OpenAI chunk dicts.
    """

    # Get valid token (auto-refreshes)
    access_token = await codex_auth.get_access_token()
    account_id = codex_auth.get_account_id()
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_cli_rs",
    }

    if account_id:
        headers["chatgpt-account-id"] = account_id

    openai_request["stream"] = True

    client = httpx.AsyncClient(timeout=60.0)

    try:
        async with client.stream("POST", CODEX_API_URL, json=openai_request, headers=headers) as response:
            if response.status_code != 200:
                error_text = await response.read()
                logger.error(f"Codex API error {response.status_code}: {error_text.decode('utf-8', errors='replace')}")
                raise HTTPException(status_code=response.status_code, detail=f"Codex API Error: {response.status_code}")

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    if not payload:
                        continue

                    try:
                        chunk = json.loads(payload)
                        yield chunk
                    except json.JSONDecodeError:
                        pass

    except httpx.RequestError as e:
        logger.error(f"Codex network error: {e}")
        raise HTTPException(status_code=502, detail=f"Codex network error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Codex unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Codex unexpected error: {e}")
    finally:
        await client.aclose()
