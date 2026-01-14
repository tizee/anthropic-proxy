"""
Codex subscription support.
Handles authentication and request proxying for Codex plan.
"""

import base64
import hashlib
import json
import logging
import secrets
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import AsyncGenerator
from http.server import BaseHTTPRequestHandler

import httpx
from fastapi import HTTPException

from .config_manager import load_auth_file, save_auth_file

logger = logging.getLogger(__name__)

# Constants
CODEX_AUTH_URL = "https://auth.openai.com/oauth/token"
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


class CodexAuth:
    """Manages Codex authentication tokens."""

    def __init__(self):
        self._auth_data = {}
        self._load()

    def _load(self):
        """Load auth data from file."""
        full_data = load_auth_file()
        # We expect data under "openai" key as per AUTH.md
        self._auth_data = full_data.get("openai", {})

    def _save(self):
        """Save auth data to file."""
        full_data = load_auth_file()
        full_data["openai"] = self._auth_data
        save_auth_file(full_data)

    async def get_access_token(self) -> str:
        """Get a valid access token, refreshing if necessary."""
        self._load()  # Reload to ensure we have latest (in case updated by another process)

        if not self._auth_data:
            raise HTTPException(status_code=401, detail="No Codex auth data found in auth.json")

        access_token = self._auth_data.get("access")
        refresh_token = self._auth_data.get("refresh")
        expires = self._auth_data.get("expires", 0)

        # Check if expired or about to expire (within 5 minutes)
        if not access_token or time.time() > (expires - 300):
            logger.info("Codex token expired or missing, refreshing...")
            if not refresh_token:
                raise HTTPException(status_code=401, detail="No refresh token available for Codex")

            return await self._refresh_token(refresh_token)

        return access_token

    async def _refresh_token(self, refresh_token: str) -> str:
        """Refresh the access token."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    CODEX_AUTH_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": CODEX_CLIENT_ID,
                        "redirect_uri": CODEX_REDIRECT_URI,
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

            new_access = data["access_token"]
            new_refresh = data.get("refresh_token", refresh_token) # Sometimes refresh token doesn't rotate?
            expires_in = data.get("expires_in", 3600)

            # Update state
            self._auth_data["access"] = new_access
            self._auth_data["refresh"] = new_refresh
            self._auth_data["expires"] = int(time.time() + expires_in)

            # Extract account ID if not present or just update it
            try:
                account_id = self._extract_account_id(new_access)
                if account_id:
                    self._auth_data["accountId"] = account_id
            except Exception as e:
                logger.warning(f"Failed to extract account ID from token: {e}")

            self._save()
            logger.info("Codex token refreshed successfully")
            return new_access

        except httpx.HTTPError as e:
            logger.error(f"Failed to refresh Codex token: {e}")
            if hasattr(e, "response") and e.response:
                 logger.error(f"Refresh error response: {e.response.text}")
            raise HTTPException(status_code=401, detail="Failed to refresh Codex token")

    def _extract_account_id(self, token: str) -> str | None:
        """Extract chatgpt_account_id from JWT token."""
        try:
            # JWT is header.payload.signature
            parts = token.split(".")
            if len(parts) < 2:
                return None

            payload_b64 = parts[1]
            # Add padding if needed
            payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)

            payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
            payload = json.loads(payload_json)

            # Look for https://api.openai.com/auth claim
            auth_claim = payload.get("https://api.openai.com/auth", {})
            return auth_claim.get("chatgpt_account_id")

        except Exception as e:
            logger.debug(f"Error extracting account ID: {e}")
            return None

    def get_account_id(self) -> str | None:
        """Get the cached account ID."""
        return self._auth_data.get("accountId")

    def login(self):
        """Initiate the browser-based login flow."""
        logger.info("Starting Codex login flow...")

        # 1. Generate PKCE
        verifier, challenge = self._generate_pkce()
        state = secrets.token_urlsafe(16)

        # 2. Start local server
        auth_code = None
        server_error = None

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal auth_code, server_error
                try:
                    parsed_url = urllib.parse.urlparse(self.path)
                    query = urllib.parse.parse_qs(parsed_url.query)

                    if parsed_url.path != "/auth/callback":
                        self.send_error(404, "Not Found")
                        return

                    if "error" in query:
                        server_error = query["error"][0]
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(b"Auth error. You can close this window.")
                        return

                    if "code" in query:
                        returned_state = query.get("state", [""])[0]
                        if returned_state != state:
                            server_error = "State mismatch"
                            self.send_error(400, "Invalid state")
                            return

                        auth_code = query["code"][0]
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html")
                        self.end_headers()
                        self.wfile.write(b"<h1>Authentication successful!</h1><p>You can close this window and return to the terminal.</p>")
                    else:
                        self.send_error(400, "No code returned")

                except Exception as e:
                    server_error = str(e)
                    logger.error(f"Callback handler error: {e}")

            def log_message(self, format, *args):
                # Silence server logs
                pass

        # Use port 1455 as per spec
        port = 1455
        try:
            server = socketserver.TCPServer(("localhost", port), CallbackHandler)
        except OSError as e:
            logger.error(f"Could not start local server on port {port}: {e}")
            print(f"Error: Port {port} is in use. Please free it and try again.")
            return

        # Run server in thread
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        # 3. Construct Auth URL
        params = {
            "client_id": CODEX_CLIENT_ID,
            "redirect_uri": CODEX_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid profile email offline_access",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": "codex_cli_rs",
        }
        auth_url = f"https://auth.openai.com/oauth/authorize?{urllib.parse.urlencode(params)}"

        print(f"\nOpening browser to: {auth_url}\n")
        print("If the browser doesn't open, copy the URL above.")

        # 4. Open Browser
        self._open_browser(auth_url)

        # 5. Wait for code
        print("Waiting for authentication...")
        try:
            while auth_code is None and server_error is None:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nLogin cancelled.")
            server.shutdown()
            server.server_close()
            return

        server.shutdown()
        server.server_close()

        if server_error:
            logger.error(f"Authentication failed: {server_error}")
            print(f"Authentication failed: {server_error}")
            return

        # 6. Exchange code for token
        logger.info("Exchanging code for tokens...")
        self._exchange_code_for_token(auth_code, verifier)
        print("Successfully logged in to Codex!")

    def _generate_pkce(self):
        verifier = secrets.token_urlsafe(32)
        digest = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        return verifier, challenge

    def _open_browser(self, url):
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", url])
            elif sys.platform == "win32":
                subprocess.Popen(["start", url], shell=True)
            elif sys.platform == "linux":
                subprocess.Popen(["xdg-open", url])
            else:
                print(f"Please open the following URL in your browser: {url}")
        except Exception as e:
            logger.warning(f"Failed to open browser: {e}")

    def _exchange_code_for_token(self, code, verifier):
        try:
            response = httpx.post(
                CODEX_AUTH_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": CODEX_REDIRECT_URI,
                    "client_id": CODEX_CLIENT_ID,
                    "code_verifier": verifier,
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()

            self._auth_data["access"] = data["access_token"]
            self._auth_data["refresh"] = data["refresh_token"]
            self._auth_data["expires"] = int(time.time() + data["expires_in"])

            # Extract account ID
            try:
                account_id = self._extract_account_id(data["access_token"])
                if account_id:
                    self._auth_data["accountId"] = account_id
            except Exception as e:
                logger.warning(f"Failed to extract account ID from token: {e}")

            self._save()

        except httpx.HTTPError as e:
            logger.error(f"Token exchange failed: {e}")
            if hasattr(e, "response") and e.response:
                 logger.error(f"Exchange error response: {e.response.text}")
            raise Exception("Failed to exchange authentication code for token")


# Global auth instance
codex_auth = CodexAuth()


from typing import Any

# ... imports ...

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
