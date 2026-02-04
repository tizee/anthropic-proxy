"""
Codex subscription support.
Handles authentication and request proxying for Codex plan.
"""

import base64
import json
import logging
import time
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

# Token refresh interval in seconds (7 days, stricter than codex_cli_rs 8 days)
TOKEN_REFRESH_INTERVAL = 7 * 24 * 60 * 60

# Default models available via Codex subscription (subject to upstream changes).
# These model names can become invalid if the provider updates or disables support.
DEFAULT_CODEX_MODELS = {
    "gpt-5.2-codex": {
        "model_name": "gpt-5.2-codex",
        "description": "Newest flagship Codex model",
        "reasoning_effort": "high"
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
}

def _is_codex_usage_limit_error(error_text: bytes) -> bool:
    try:
        text = error_text.decode("utf-8", errors="replace")
    except Exception:
        return False

    if "usage_limit_reached" in text:
        return True

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False

    error_obj = data.get("error", {})
    if isinstance(error_obj, dict):
        code = error_obj.get("code") or error_obj.get("type")
        if isinstance(code, str) and "usage_limit_reached" in code:
            return True
    return False


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

    def _is_token_stale(self) -> bool:
        """Check if token needs refresh based on 8-day interval (codex_cli_rs behavior)."""
        last_refresh = self._auth_data.get("last_refresh")
        if not last_refresh:
            return True
        return time.time() > (last_refresh + TOKEN_REFRESH_INTERVAL)

    async def get_access_token(self) -> str:
        """Get access token, refreshing if stale (8-day interval) or expired."""
        self._load()
        if not self._auth_data:
            raise HTTPException(
                status_code=401,
                detail="No Codex auth data found in auth.json",
            )

        access_token = self._auth_data.get("access")
        refresh_full = self._auth_data.get("refresh")
        expires = self._auth_data.get("expires", 0)

        # Check if token is stale (8-day interval) or about to expire (5 min buffer)
        needs_refresh = self._is_token_stale() or not access_token or time.time() > (expires - 300)

        if needs_refresh:
            if self._is_token_stale():
                logger.info("Codex token is stale (8+ days since last refresh), refreshing...")
            else:
                logger.info("Codex token expired, refreshing...")
            refresh_token = self._extract_refresh_token(refresh_full)
            if not refresh_token:
                raise HTTPException(status_code=401, detail="No refresh token available")
            return await self._refresh_token(refresh_token, refresh_full)

        return access_token


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
                error_str = error_text.decode('utf-8', errors='replace')
                logger.error(f"Codex API error {response.status_code}: {error_str}")
                if response.status_code == 404 and _is_codex_usage_limit_error(error_text):
                    raise HTTPException(status_code=429, detail=f"Codex usage limit reached: {error_str}")
                raise HTTPException(status_code=response.status_code, detail=f"Codex API error {response.status_code}: {error_str}")

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
