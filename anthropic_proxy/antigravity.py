"""
Antigravity (Google Internal) authentication and request handling.
Handles authentication and request proxying for Antigravity API.
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
from typing import Any

import httpx
from fastapi import HTTPException

from .config_manager import load_auth_file, save_auth_file
from .gemini_converter import anthropic_to_gemini_request
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

class AntigravityAuth:
    """Manages Antigravity authentication tokens."""

    def __init__(self):
        self._auth_data = {}
        self._load()

    def _load(self):
        """Load auth data from file."""
        full_data = load_auth_file()
        # Store under "antigravity" key to distinguish from standard Gemini
        self._auth_data = full_data.get("antigravity", {})

    def _save(self):
        """Save auth data to file."""
        full_data = load_auth_file()
        full_data["antigravity"] = self._auth_data
        save_auth_file(full_data)

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

    def login(self):
        """Initiate the browser-based login flow."""
        logger.info("Starting Antigravity login flow...")

        verifier, challenge = self._generate_pkce()
        state = secrets.token_urlsafe(16)

        auth_code = None
        server_error = None

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal auth_code, server_error
                try:
                    parsed_url = urllib.parse.urlparse(self.path)
                    query = urllib.parse.parse_qs(parsed_url.query)

                    if parsed_url.path != "/oauth-callback":
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
                pass

        port = 51121
        try:
            server = socketserver.TCPServer(("localhost", port), CallbackHandler)
        except OSError as e:
            logger.error(f"Could not start local server on port {port}: {e}")
            print(f"Error: Port {port} is in use. Please free it and try again.")
            return

        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        params = {
            "client_id": ANTIGRAVITY_CLIENT_ID,
            "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(ANTIGRAVITY_SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = f"{ANTIGRAVITY_AUTH_URL}?{urllib.parse.urlencode(params)}"

        print(f"\nOpening browser to: {auth_url}\n")
        print("If the browser doesn't open, copy the URL above.")

        self._open_browser(auth_url)

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

        logger.info("Exchanging code for tokens...")

        try:
            self._exchange_code_for_token_sync(auth_code, verifier)
            print("Successfully logged in to Antigravity!")

            print("Resolving Project Context...")
            import asyncio
            asyncio.run(self.ensure_project_context())
            print("Project configuration resolved.")

        except Exception as e:
            print(f"Login setup failed: {e}")
            logger.error(f"Login setup failed: {e}")

    def _exchange_code_for_token_sync(self, code, verifier):
        response = httpx.post(
            ANTIGRAVITY_TOKEN_URL,
            data={
                "client_id": ANTIGRAVITY_CLIENT_ID,
                "client_secret": ANTIGRAVITY_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": ANTIGRAVITY_REDIRECT_URI,
                "code_verifier": verifier,
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()

        self._auth_data["access"] = data["access_token"]
        self._auth_data["refresh"] = data["refresh_token"]
        self._auth_data["expires"] = int(time.time() + data["expires_in"])
        self._save()

    async def get_access_token(self) -> str:
        """Get valid access token, auto-refreshing if needed."""
        self._load()
        if not self._auth_data:
             raise HTTPException(status_code=401, detail="No Antigravity auth data found")

        access_token = self._auth_data.get("access")
        refresh_full = self._auth_data.get("refresh")
        expires = self._auth_data.get("expires", 0)

        if not access_token or time.time() > (expires - 300):
            logger.info("Antigravity token expired, refreshing...")
            if not refresh_full:
                 raise HTTPException(status_code=401, detail="No refresh token available")

            parts = refresh_full.split("|")
            refresh_token = parts[0]

            return await self._refresh_token(refresh_token, refresh_full)

        return access_token

    async def _refresh_token(self, refresh_token_raw: str, full_refresh_string: str) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    ANTIGRAVITY_TOKEN_URL,
                    data={
                        "client_id": ANTIGRAVITY_CLIENT_ID,
                        "client_secret": ANTIGRAVITY_CLIENT_SECRET,
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token_raw,
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()

                new_access = data["access_token"]
                expires_in = data.get("expires_in", 3600)

                self._auth_data["access"] = new_access
                self._auth_data["expires"] = int(time.time() + expires_in)
                self._auth_data["refresh"] = full_refresh_string

                self._save()
                return new_access
            except Exception as e:
                logger.error(f"Antigravity refresh failed: {e}")
                raise HTTPException(status_code=401, detail="Failed to refresh Antigravity token")

    def get_project_id(self) -> str | None:
        refresh = self._auth_data.get("refresh", "")
        parts = refresh.split("|")
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
                # Use loadCodeAssist to find project
                res = await client.post(
                    f"{ANTIGRAVITY_ENDPOINT}/v1internal:loadCodeAssist",
                    json=req_body,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        **ANTIGRAVITY_HEADERS
                    }
                )

                if res.status_code == 200:
                    data = res.json()
                    found_id = data.get("cloudaicompanionProject")
                    if found_id:
                        self._update_refresh_with_project(refresh_token, found_id)
                        return

                # If loadCodeAssist fails, try onboardUser
                req_body["tierId"] = "FREE"
                res = await client.post(
                    f"{ANTIGRAVITY_ENDPOINT}/v1internal:onboardUser",
                    json=req_body,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        **ANTIGRAVITY_HEADERS
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    new_id = data.get("response", {}).get("cloudaicompanionProject", {}).get("id")
                    if new_id:
                        self._update_refresh_with_project(refresh_token, new_id)
                    else:
                        # Fallback to default project ID from reference
                        self._update_refresh_with_project(refresh_token, "rising-fact-p41fc")

        except Exception as e:
            logger.error(f"Failed to ensure project context: {e}")

    def _update_refresh_with_project(self, token, proj):
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

    action = "streamGenerateContent"
    effective_model = model_id

    converted_body = anthropic_to_gemini_request(
        request,
        effective_model,
        is_antigravity=True,
        system_prefix=ANTIGRAVITY_SYSTEM_INSTRUCTION,
    )

    wrapped_body = {
        "project": project_id,
        "model": effective_model,
        "request": converted_body,
        "userAgent": "antigravity",
        "requestId": f"agent-{secrets.token_hex(8)}"
    }

    url = f"{ANTIGRAVITY_ENDPOINT}/v1internal:{action}?alt=sse"

    client = httpx.AsyncClient(timeout=60.0)

    try:
        async with client.stream("POST", url, json=wrapped_body, headers={
             "Authorization": f"Bearer {access_token}",
             **ANTIGRAVITY_HEADERS
        }) as res:
            if res.status_code != 200:
                err = await res.read()
                logger.error(f"Antigravity API Error: {res.status_code} {err}")
                raise HTTPException(status_code=res.status_code, detail=f"Antigravity Error: {err.decode('utf-8', errors='replace')}")

            async for line in res.aiter_lines():
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if not payload: continue
                    try:
                        data = json.loads(payload)
                        inner = data.get("response")
                        if inner:
                            yield inner
                    except json.JSONDecodeError:
                        pass
    except httpx.RequestError as e:
        logger.error(f"Antigravity network error: {e}")
        raise HTTPException(status_code=502, detail=f"Antigravity network error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Antigravity unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Antigravity unexpected error: {e}")
    finally:
        await client.aclose()
