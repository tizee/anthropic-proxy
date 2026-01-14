import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import os
import sys
import time

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.antigravity import AntigravityAuth, ANTIGRAVITY_ENDPOINT


class TestAntigravityAuth(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.auth_data = {
            "antigravity": {
                "access": "old_access",
                "refresh": "refresh_token|project",
                "expires": time.time() - 100,
            }
        }
        self.auth = AntigravityAuth()

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    @patch("anthropic_proxy.auth_provider.save_auth_file")
    @patch("httpx.AsyncClient")
    async def test_get_access_token_refresh(self, mock_client, mock_save, mock_load):
        mock_load.return_value = self.auth_data

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access",
            "expires_in": 3600,
        }

        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_http_client

        token = await self.auth.get_access_token()

        self.assertEqual(token, "new_access")
        mock_http_client.post.assert_called_once()
        mock_save.assert_called_once()

        saved_data = mock_save.call_args[0][0]
        self.assertEqual(saved_data["antigravity"]["access"], "new_access")
        self.assertEqual(saved_data["antigravity"]["refresh"], "refresh_token|project")
        self.assertAlmostEqual(saved_data["antigravity"]["expires"], time.time() + 3600, delta=10)

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    async def test_get_access_token_valid(self, mock_load):
        valid_auth = {
            "antigravity": {
                "access": "valid_access",
                "refresh": "refresh_token|project",
                "expires": time.time() + 3600,
            }
        }
        mock_load.return_value = valid_auth

        token = await self.auth.get_access_token()
        self.assertEqual(token, "valid_access")

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    @patch("httpx.AsyncClient")
    async def test_get_access_token_refresh_fail(self, mock_client, mock_load):
        expired_auth = {
            "antigravity": {
                "access": "old_access",
                "refresh": "bad_refresh|project",
                "expires": time.time() - 100,
            }
        }
        mock_load.return_value = expired_auth

        mock_http_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 400
        error = Exception("bad request")
        mock_response.raise_for_status.side_effect = error

        mock_http_client.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_http_client

        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as cm:
            await self.auth.get_access_token()

        self.assertEqual(cm.exception.status_code, 401)

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    @patch("anthropic_proxy.auth_provider.save_auth_file")
    @patch("httpx.AsyncClient")
    async def test_ensure_project_context_load(self, mock_client, mock_save, mock_load):
        auth_data = {
            "antigravity": {
                "access": "access",
                "refresh": "refresh_token",
                "expires": time.time() + 3600,
            }
        }
        mock_load.return_value = auth_data

        mock_http_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cloudaicompanionProject": "project-321"}
        mock_http_client.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_http_client

        await self.auth.ensure_project_context()

        mock_http_client.post.assert_called_once()
        called_url = mock_http_client.post.call_args[0][0]
        self.assertEqual(
            called_url,
            f"{ANTIGRAVITY_ENDPOINT}/v1internal:loadCodeAssist",
        )

        saved_data = mock_save.call_args[0][0]
        self.assertEqual(saved_data["antigravity"]["refresh"], "refresh_token|project-321")

    @patch("anthropic_proxy.auth_provider.OAuthPKCEAuth._open_browser")
    @patch("anthropic_proxy.auth_provider.socketserver.TCPServer")
    @patch("anthropic_proxy.auth_provider.threading.Thread")
    @patch("anthropic_proxy.auth_provider.time.sleep", side_effect=KeyboardInterrupt)
    def test_login_starts_server(self, mock_sleep, mock_thread, mock_server, mock_open):
        server_instance = MagicMock()
        mock_server.return_value = server_instance

        self.auth.login()

        mock_server.assert_called_with(("localhost", 51121), unittest.mock.ANY)
        mock_thread.assert_called()
        mock_thread.return_value.start.assert_called()


if __name__ == "__main__":
    unittest.main()
