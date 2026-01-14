import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic_proxy.gemini import GEMINI_CODE_ASSIST_ENDPOINT, GeminiAuth, handle_gemini_request
from anthropic_proxy.types import ClaudeMessagesRequest


class TestGeminiAuth(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.auth_data = {
            "google": {
                "access": "old_access",
                "refresh": "refresh_token|project|managed",
                "expires": time.time() - 100,
            }
        }
        self.auth = GeminiAuth()

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
        self.assertEqual(saved_data["google"]["access"], "new_access")
        self.assertEqual(saved_data["google"]["refresh"], "refresh_token|project|managed")
        self.assertAlmostEqual(saved_data["google"]["expires"], time.time() + 3600, delta=10)

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    async def test_get_access_token_valid(self, mock_load):
        valid_auth = {
            "google": {
                "access": "valid_access",
                "refresh": "refresh_token|project|managed",
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
            "google": {
                "access": "old_access",
                "refresh": "bad_refresh|project|managed",
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
            "google": {
                "access": "access",
                "refresh": "refresh_token",
                "expires": time.time() + 3600,
            }
        }
        mock_load.return_value = auth_data

        mock_http_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cloudaicompanionProject": "project-123"}
        mock_http_client.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_http_client

        await self.auth.ensure_project_context()

        mock_http_client.post.assert_called_once()
        called_url = mock_http_client.post.call_args[0][0]
        self.assertEqual(
            called_url,
            f"{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
        )

        saved_data = mock_save.call_args[0][0]
        self.assertEqual(saved_data["google"]["refresh"], "refresh_token||project-123")

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    @patch("anthropic_proxy.auth_provider.save_auth_file")
    @patch("httpx.AsyncClient")
    async def test_ensure_project_context_onboard_user(self, mock_client, mock_save, mock_load):
        auth_data = {
            "google": {
                "access": "access",
                "refresh": "refresh_token",
                "expires": time.time() + 3600,
            }
        }
        mock_load.return_value = auth_data

        load_response = MagicMock()
        load_response.status_code = 404
        load_response.json.return_value = {}

        onboard_response = MagicMock()
        onboard_response.status_code = 200
        onboard_response.json.return_value = {
            "response": {"cloudaicompanionProject": {"id": "managed-999"}}
        }

        mock_http_client = AsyncMock()
        mock_http_client.post.side_effect = [load_response, onboard_response]
        mock_client.return_value.__aenter__.return_value = mock_http_client

        await self.auth.ensure_project_context()

        self.assertEqual(mock_http_client.post.call_count, 2)
        called_url = mock_http_client.post.call_args_list[1][0][0]
        self.assertEqual(
            called_url,
            f"{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
        )

        saved_data = mock_save.call_args[0][0]
        self.assertEqual(saved_data["google"]["refresh"], "refresh_token||managed-999")

    def test_get_project_id_prefers_managed(self):
        self.auth._auth_data = {"refresh": "token|project|managed"}
        self.assertEqual(self.auth.get_project_id(), "managed")

    @patch("anthropic_proxy.auth_provider.OAuthPKCEAuth._open_browser")
    @patch("anthropic_proxy.auth_provider.socketserver.TCPServer")
    @patch("anthropic_proxy.auth_provider.threading.Thread")
    @patch("anthropic_proxy.auth_provider.time.sleep", side_effect=KeyboardInterrupt)
    def test_login_starts_server(self, mock_sleep, mock_thread, mock_server, mock_open):
        server_instance = MagicMock()
        mock_server.return_value = server_instance

        self.auth.login()

        mock_server.assert_called_with(("localhost", 8085), unittest.mock.ANY)
        mock_thread.assert_called()
        mock_thread.return_value.start.assert_called()

    @patch("anthropic_proxy.gemini.stream_gemini_sdk_request")
    @patch("anthropic_proxy.gemini.gemini_auth.get_project_id")
    @patch("anthropic_proxy.gemini.gemini_auth.get_access_token", new_callable=AsyncMock)
    async def test_handle_gemini_request_uses_code_assist(
        self, mock_get_access, mock_get_project, mock_stream
    ):
        mock_get_access.return_value = "access-token"
        mock_get_project.return_value = "project-123"

        async def fake_stream(*args, **kwargs):
            yield {"chunk": "ok"}

        mock_stream.side_effect = fake_stream

        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )

        chunks = [chunk async for chunk in handle_gemini_request(request, "gemini-2.5-pro")]
        self.assertEqual(chunks, [{"chunk": "ok"}])
        self.assertTrue(mock_stream.called)
        self.assertTrue(mock_stream.call_args.kwargs["use_code_assist"])

    @patch("anthropic_proxy.gemini.stream_gemini_sdk_request")
    @patch("anthropic_proxy.gemini.gemini_auth.get_project_id")
    @patch("anthropic_proxy.gemini.gemini_auth.get_access_token", new_callable=AsyncMock)
    async def test_handle_gemini_request_applies_model_fallback(
        self, mock_get_access, mock_get_project, mock_stream
    ):
        mock_get_access.return_value = "access-token"
        mock_get_project.return_value = "project-123"

        async def fake_stream(*args, **kwargs):
            yield {"chunk": "ok"}

        mock_stream.side_effect = fake_stream

        request = ClaudeMessagesRequest(
            model="gemini-2.5-flash-image",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )

        chunks = [
            chunk
            async for chunk in handle_gemini_request(
                request, "gemini-2.5-flash-image"
            )
        ]
        self.assertEqual(chunks, [{"chunk": "ok"}])
        self.assertEqual(mock_stream.call_args.kwargs["model_id"], "gemini-2.5-flash")


if __name__ == "__main__":
    unittest.main()
