import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import sys
import time
import base64
import json
import hashlib
import httpx

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.codex import CodexAuth, handle_codex_request, _is_codex_usage_limit_error

class TestCodexAuth(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.auth_data = {
            "openai": {
                "access": "old_access",
                "refresh": "refresh_token",
                "expires": time.time() - 100, # Expired
                "accountId": "old_account"
            }
        }
        self.auth = CodexAuth()

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    @patch("anthropic_proxy.auth_provider.save_auth_file")
    @patch("httpx.AsyncClient")
    async def test_get_access_token_refresh(self, mock_client, mock_save, mock_load):
        # Setup mocks
        mock_load.return_value = self.auth_data
        
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_in": 3600
        }
        
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_http_client
        
        # Call get_access_token
        token = await self.auth.get_access_token()
        
        # Verify
        self.assertEqual(token, "new_access")
        mock_http_client.post.assert_called_once()
        mock_save.assert_called_once()
        
        # Verify saved data
        saved_data = mock_save.call_args[0][0]
        self.assertEqual(saved_data["openai"]["access"], "new_access")
        self.assertEqual(saved_data["openai"]["refresh"], "new_refresh")
        self.assertAlmostEqual(saved_data["openai"]["expires"], time.time() + 3600, delta=10)

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    async def test_get_access_token_valid(self, mock_load):
        # Setup valid token (expires in 1 hour, refreshed recently)
        valid_auth = {
            "openai": {
                "access": "valid_access",
                "refresh": "refresh_token",
                "expires": time.time() + 3600,
                "accountId": "account_id",
                "last_refresh": int(time.time())  # Recent refresh prevents stale check
            }
        }
        mock_load.return_value = valid_auth

        # Call get_access_token
        token = await self.auth.get_access_token()

        # Verify
        self.assertEqual(token, "valid_access")

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    @patch("anthropic_proxy.auth_provider.save_auth_file")
    @patch("httpx.AsyncClient")
    async def test_get_access_token_proactive_refresh(self, mock_client, mock_save, mock_load):
        """Test that token is refreshed if it expires in < 5 minutes."""
        # Setup token expiring in 4 minutes (should trigger refresh)
        near_expiry_auth = {
            "openai": {
                "access": "old_access",
                "refresh": "refresh_token",
                "expires": time.time() + 240, 
                "accountId": "account_id"
            }
        }
        mock_load.return_value = near_expiry_auth
        
        # Mock successful refresh
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access",
            "refresh_token": "new_refresh",
            "expires_in": 3600
        }
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_http_client

        # Call get_access_token
        token = await self.auth.get_access_token()
        
        # Verify it refreshed
        self.assertEqual(token, "new_access")
        mock_http_client.post.assert_called_once()

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    @patch("httpx.AsyncClient")
    async def test_get_access_token_refresh_fail(self, mock_client, mock_load):
        """Test handling of failed refresh (e.g. revoked token)."""
        # Setup expired token
        expired_auth = {
            "openai": {
                "access": "old_access",
                "refresh": "bad_refresh_token",
                "expires": time.time() - 100, 
            }
        }
        mock_load.return_value = expired_auth
        
        # Mock failed refresh (400 Bad Request)
        mock_http_client = AsyncMock()
        # Simulate raise_for_status raising an error
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "invalid_grant"
        error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=mock_response)
        
        mock_http_client.post.return_value = mock_response
        mock_response.raise_for_status.side_effect = error
        
        mock_client.return_value.__aenter__.return_value = mock_http_client

        # Verify it raises HTTPException 401
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as cm:
            await self.auth.get_access_token()
        
        self.assertEqual(cm.exception.status_code, 401)
        self.assertIn("Failed to refresh Codex token", cm.exception.detail)

    def test_extract_account_id(self):
        # Create a dummy JWT with account ID
        payload = {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "test-account-id"
            }
        }
        payload_json = json.dumps(payload)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()
        token = f"header.{payload_b64}.signature"
        
        account_id = self.auth._extract_account_id(token)
        self.assertEqual(account_id, "test-account-id")

    def test_usage_limit_error_detection(self):
        payload = b'{"error":{"code":"usage_limit_reached"}}'
        self.assertTrue(_is_codex_usage_limit_error(payload))

    def test_generate_pkce(self):
        verifier, challenge = self.auth._generate_pkce()
        self.assertTrue(len(verifier) > 0)
        self.assertTrue(len(challenge) > 0)
        
        # Verify challenge is base64 encoded SHA256 of verifier
        digest = hashlib.sha256(verifier.encode()).digest()
        expected_challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        self.assertEqual(challenge, expected_challenge)

    @patch("anthropic_proxy.codex.codex_auth.get_account_id", return_value=None)
    @patch("anthropic_proxy.codex.codex_auth.get_access_token", new_callable=AsyncMock)
    @patch("anthropic_proxy.codex.httpx.AsyncClient")
    async def test_handle_codex_usage_limit_maps_429(
        self, mock_client, mock_get_access_token, _mock_account_id
    ):
        mock_get_access_token.return_value = "access-token"

        class FakeStream:
            def __init__(self, response):
                self._response = response

            async def __aenter__(self):
                return self._response

            async def __aexit__(self, exc_type, exc, tb):
                return False

        class FakeClient:
            def __init__(self, response):
                self._response = response

            def stream(self, *args, **kwargs):
                return FakeStream(self._response)

            async def aclose(self):
                return None

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.aread = AsyncMock(
            return_value=b'{"error":{"code":"usage_limit_reached"}}'
        )
        mock_client.return_value = FakeClient(mock_response)

        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as cm:
            await handle_codex_request({"model": "gpt-5.2-codex"}, "gpt-5.2-codex").__anext__()

        self.assertEqual(cm.exception.status_code, 429)

    @patch("anthropic_proxy.auth_provider.socketserver.TCPServer")
    @patch("anthropic_proxy.auth_provider.threading.Thread")
    @patch("anthropic_proxy.codex.CodexAuth._open_browser")
    @patch("builtins.print")
    def test_login_starts_server(self, mock_print, mock_open_browser, mock_thread, mock_server):
        # Mock server context manager
        server_instance = MagicMock()
        mock_server.return_value = server_instance
        
        # We need to simulate the auth flow completing or interrupt it, otherwise the loop waits forever
        # So we'll patch the while loop condition or raise KeyboardInterrupt
        
        # Actually, since I can't easily mock the local variables inside login(), 
        # I'll rely on checking if server was initialized.
        # But login() blocks until auth_code is set.
        # I can mock `time.sleep` to raise KeyboardInterrupt to break the loop.
        
        with patch("time.sleep", side_effect=KeyboardInterrupt):
            self.auth.login()
            
        # Verify server was started on port 1455
        mock_server.assert_called_with(("localhost", 1455), unittest.mock.ANY)
        server_instance.serve_forever.assert_not_called() # It's called in a thread
        mock_thread.assert_called()
        mock_thread.return_value.start.assert_called()
        
        # Verify browser was opened
        mock_open_browser.assert_called()

if __name__ == "__main__":
    unittest.main()
