"""Shared OAuth PKCE auth utilities for provider logins."""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from typing import Any

import httpx
from fastapi import HTTPException

from .config_manager import load_auth_file, save_auth_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OAuthProviderConfig:
    provider_key: str
    display_name: str
    client_id: str
    client_secret: str | None
    auth_url: str
    token_url: str
    redirect_uri: str
    callback_path: str
    port: int
    scopes: list[str] = field(default_factory=list)
    auth_params: dict[str, str] = field(default_factory=dict)
    token_params: dict[str, str] = field(default_factory=dict)
    refresh_params: dict[str, str] = field(default_factory=dict)
    include_redirect_uri_on_refresh: bool = False


class OAuthPKCEAuth:
    """Base OAuth PKCE auth flow with browser login and token refresh."""

    def __init__(self, config: OAuthProviderConfig):
        self.config = config
        self._auth_data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        full_data = load_auth_file()
        self._auth_data = full_data.get(self.config.provider_key, {})

    def _save(self) -> None:
        full_data = load_auth_file()
        full_data[self.config.provider_key] = self._auth_data
        save_auth_file(full_data)

    def has_auth(self) -> bool:
        return bool(self._auth_data.get("refresh") or self._auth_data.get("access"))

    def _generate_pkce(self) -> tuple[str, str]:
        verifier = secrets.token_urlsafe(32)
        digest = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        return verifier, challenge

    def _open_browser(self, url: str) -> None:
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", url])
            elif sys.platform == "win32":
                subprocess.Popen(["start", url], shell=True)
            elif sys.platform == "linux":
                subprocess.Popen(["xdg-open", url])
            else:
                print(f"Please open the following URL in your browser: {url}")
        except Exception as exc:
            logger.warning(f"Failed to open browser: {exc}")

    def _build_auth_params(self, state: str, challenge: str) -> dict[str, str]:
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        if self.config.scopes and "scope" not in self.config.auth_params:
            params["scope"] = " ".join(self.config.scopes)
        params.update(self.config.auth_params)
        return params

    def _build_token_exchange_data(self, code: str, verifier: str) -> dict[str, str]:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
            "code_verifier": verifier,
        }
        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret
        data.update(self.config.token_params)
        return data

    def _store_token_data(self, data: dict[str, Any]) -> None:
        self._auth_data["access"] = data["access_token"]
        if data.get("refresh_token"):
            self._auth_data["refresh"] = data["refresh_token"]
        self._auth_data["expires"] = int(time.time() + data.get("expires_in", 3600))

    def _post_token_exchange(self, data: dict[str, Any]) -> None:
        return None

    def _exchange_code_for_token_sync(self, code: str, verifier: str) -> None:
        try:
            response = httpx.post(
                self.config.token_url,
                data=self._build_token_exchange_data(code, verifier),
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            self._store_token_data(data)
            self._post_token_exchange(data)
            self._save()
        except httpx.HTTPError as exc:
            logger.error(f"Token exchange failed: {exc}")
            if hasattr(exc, "response") and exc.response:
                logger.error(f"Exchange error response: {exc.response.text}")
            raise Exception("Failed to exchange authentication code for token") from exc

    def login(self) -> None:
        logger.info(f"Starting {self.config.display_name} login flow...")

        verifier, challenge = self._generate_pkce()
        state = secrets.token_urlsafe(16)
        auth_code = None
        server_error = None

        callback_path = self.config.callback_path

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                nonlocal auth_code, server_error
                try:
                    parsed_url = urllib.parse.urlparse(self.path)
                    query = urllib.parse.parse_qs(parsed_url.query)

                    if parsed_url.path != callback_path:
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
                        self.wfile.write(
                            b"<h1>Authentication successful!</h1>"
                            b"<p>You can close this window and return to the terminal.</p>"
                        )
                    else:
                        self.send_error(400, "No code returned")

                except Exception as exc:
                    server_error = str(exc)
                    logger.error(f"Callback handler error: {exc}")

            def log_message(self, format, *args):  # noqa: A003
                pass

        port = self.config.port
        try:
            server = socketserver.TCPServer(("localhost", port), CallbackHandler)
        except OSError as exc:
            logger.error(f"Could not start local server on port {port}: {exc}")
            print(f"Error: Port {port} is in use. Please free it and try again.")
            return

        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        params = self._build_auth_params(state, challenge)
        auth_url = f"{self.config.auth_url}?{urllib.parse.urlencode(params)}"

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
        self._exchange_code_for_token_sync(auth_code, verifier)
        print(f"Successfully logged in to {self.config.display_name}!")

        self._post_login()

    def _post_login(self) -> None:
        return None

    def _extract_refresh_token(self, refresh_full: str | None) -> str | None:
        return refresh_full

    async def get_access_token(self) -> str:
        self._load()
        if not self._auth_data:
            raise HTTPException(
                status_code=401,
                detail=f"No {self.config.display_name} auth data found in auth.json",
            )

        access_token = self._auth_data.get("access")
        refresh_full = self._auth_data.get("refresh")
        expires = self._auth_data.get("expires", 0)

        if not access_token or time.time() > (expires - 300):
            logger.info(f"{self.config.display_name} token expired, refreshing...")
            refresh_token = self._extract_refresh_token(refresh_full)
            if not refresh_token:
                raise HTTPException(status_code=401, detail="No refresh token available")
            return await self._refresh_token(refresh_token, refresh_full)

        return access_token

    async def _refresh_token(self, refresh_token_raw: str, refresh_full: str | None) -> str:
        async with httpx.AsyncClient() as client:
            try:
                data = {
                    "client_id": self.config.client_id,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token_raw,
                }
                if self.config.client_secret:
                    data["client_secret"] = self.config.client_secret
                if self.config.include_redirect_uri_on_refresh:
                    data["redirect_uri"] = self.config.redirect_uri
                data.update(self.config.refresh_params)

                response = await client.post(
                    self.config.token_url,
                    data=data,
                    timeout=30.0,
                )
                response.raise_for_status()
                token_data = response.json()

                new_access = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)

                self._auth_data["access"] = new_access
                self._auth_data["expires"] = int(time.time() + expires_in)
                if refresh_full:
                    self._auth_data["refresh"] = refresh_full
                else:
                    self._auth_data["refresh"] = token_data.get(
                        "refresh_token", refresh_token_raw
                    )
                self._save()
                return new_access
            except Exception as exc:
                logger.error(f"{self.config.display_name} refresh failed: {exc}")
                raise HTTPException(
                    status_code=401,
                    detail=f"Failed to refresh {self.config.display_name} token",
                ) from exc
