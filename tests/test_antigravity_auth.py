import unittest
from contextlib import asynccontextmanager
from unittest.mock import patch, MagicMock, AsyncMock
import os
import sys
import time

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.antigravity import (
    AntigravityAuth,
    ANTIGRAVITY_DEFAULT_PROJECT_ID,
    ANTIGRAVITY_ENDPOINTS,
    ANTIGRAVITY_LOAD_ENDPOINTS,
    ANTIGRAVITY_THINKING_HEADER,
    handle_antigravity_request,
)
from anthropic_proxy.types import ClaudeMessage, ClaudeMessagesRequest


class FakeStreamResponse:
    def __init__(self, status_code=200, lines=None, body=b"", reason="OK"):
        self.status_code = status_code
        self.reason_phrase = reason
        self._lines = lines or []
        self._body = body
        self.headers = {}

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body


@asynccontextmanager
async def stream_context(response):
    yield response


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
            f"{ANTIGRAVITY_LOAD_ENDPOINTS[0]}/v1internal:loadCodeAssist",
        )

        saved_data = mock_save.call_args[0][0]
        self.assertEqual(saved_data["antigravity"]["refresh"], "refresh_token|project-321")

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    @patch("anthropic_proxy.auth_provider.save_auth_file")
    @patch("httpx.AsyncClient")
    async def test_ensure_project_context_load_object_id(self, mock_client, mock_save, mock_load):
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
        mock_response.json.return_value = {"cloudaicompanionProject": {"id": "project-obj"}}
        mock_http_client.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_http_client

        await self.auth.ensure_project_context()

        saved_data = mock_save.call_args[0][0]
        self.assertEqual(saved_data["antigravity"]["refresh"], "refresh_token|project-obj")

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    @patch("anthropic_proxy.auth_provider.save_auth_file")
    @patch("httpx.AsyncClient")
    async def test_ensure_project_context_onboard_user(self, mock_client, mock_save, mock_load):
        auth_data = {
            "antigravity": {
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
            "response": {"cloudaicompanionProject": {"id": "project-456"}}
        }

        mock_http_client = AsyncMock()
        mock_http_client.post.side_effect = (
            [load_response for _ in ANTIGRAVITY_LOAD_ENDPOINTS] + [onboard_response]
        )
        mock_client.return_value.__aenter__.return_value = mock_http_client

        await self.auth.ensure_project_context()

        self.assertEqual(mock_http_client.post.call_count, len(ANTIGRAVITY_LOAD_ENDPOINTS) + 1)
        called_url = mock_http_client.post.call_args_list[-1][0][0]
        self.assertEqual(
            called_url,
            f"{ANTIGRAVITY_ENDPOINTS[0]}/v1internal:onboardUser",
        )

        saved_data = mock_save.call_args[0][0]
        self.assertEqual(saved_data["antigravity"]["refresh"], "refresh_token|project-456")

    @patch("anthropic_proxy.auth_provider.load_auth_file")
    @patch("anthropic_proxy.auth_provider.save_auth_file")
    @patch("httpx.AsyncClient")
    async def test_ensure_project_context_refreshes_access(self, mock_client, mock_save, mock_load):
        auth_data = {
            "antigravity": {
                "refresh": "refresh_token",
                "expires": time.time() + 3600,
            }
        }
        mock_load.return_value = auth_data

        self.auth.get_access_token = AsyncMock(return_value="new_access")

        mock_http_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cloudaicompanionProject": "project-999"}
        mock_http_client.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value = mock_http_client

        await self.auth.ensure_project_context()

        called_headers = mock_http_client.post.call_args[1]["headers"]
        self.assertEqual(called_headers["Authorization"], "Bearer new_access")

    def test_get_project_id(self):
        self.auth._auth_data = {"refresh": "token|project"}
        self.assertEqual(self.auth.get_project_id(), "project")

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

    @patch("anthropic_proxy.antigravity.antigravity_auth")
    @patch("anthropic_proxy.antigravity.httpx.AsyncClient")
    async def test_handle_antigravity_request_uses_default_project(self, mock_client, mock_auth):
        mock_auth.get_access_token = AsyncMock(return_value="token")
        mock_auth.get_project_id = MagicMock(return_value=None)
        mock_auth.ensure_project_context = AsyncMock()

        ok_response = FakeStreamResponse(
            status_code=200,
            lines=['data: {"response":{"candidates":[]}}'],
        )

        mock_http_client = MagicMock()
        mock_http_client.stream.return_value = stream_context(ok_response)
        mock_client.return_value.__aenter__.return_value = mock_http_client

        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1,
            messages=[ClaudeMessage(role="user", content="hi")],
        )

        gen = handle_antigravity_request(request, "claude-sonnet-4-5")
        await gen.__anext__()

        payload = mock_http_client.stream.call_args.kwargs["json"]
        self.assertEqual(payload["project"], ANTIGRAVITY_DEFAULT_PROJECT_ID)

    @patch("anthropic_proxy.antigravity.antigravity_auth")
    @patch("anthropic_proxy.antigravity.httpx.AsyncClient")
    async def test_handle_antigravity_request_fallback_endpoints(self, mock_client, mock_auth):
        mock_auth.get_access_token = AsyncMock(return_value="token")
        mock_auth.get_project_id = MagicMock(return_value="project-1")

        error_response = FakeStreamResponse(
            status_code=403,
            body=b"forbidden",
            reason="Forbidden",
        )
        ok_response = FakeStreamResponse(
            status_code=200,
            lines=['data: {"response":{"candidates":[]}}'],
        )

        mock_http_client = MagicMock()
        mock_http_client.stream.side_effect = [
            stream_context(error_response),
            stream_context(ok_response),
        ]
        mock_client.return_value.__aenter__.return_value = mock_http_client

        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1,
            messages=[ClaudeMessage(role="user", content="hi")],
        )

        gen = handle_antigravity_request(
            request,
            "antigravity/claude-sonnet-4-5",
            model_name="claude-sonnet-4-5",
        )
        chunk = await gen.__anext__()

        self.assertIn("candidates", chunk)
        self.assertEqual(mock_http_client.stream.call_count, 2)
        first_url = mock_http_client.stream.call_args_list[0][0][1]
        second_url = mock_http_client.stream.call_args_list[1][0][1]
        self.assertTrue(first_url.startswith(ANTIGRAVITY_ENDPOINTS[0]))
        self.assertTrue(second_url.startswith(ANTIGRAVITY_ENDPOINTS[1]))

    @patch("anthropic_proxy.antigravity.antigravity_auth")
    @patch("anthropic_proxy.antigravity.httpx.AsyncClient")
    async def test_handle_antigravity_request_capacity_fallback(self, mock_client, mock_auth):
        mock_auth.get_access_token = AsyncMock(return_value="token")
        mock_auth.get_project_id = MagicMock(return_value="project-1")

        error_response = FakeStreamResponse(
            status_code=400,
            body=b"No capacity available",
            reason="Bad Request",
        )
        ok_response = FakeStreamResponse(
            status_code=200,
            lines=['data: {"response":{"candidates":[]}}'],
        )

        mock_http_client = MagicMock()
        mock_http_client.stream.side_effect = [
            stream_context(error_response),
            stream_context(ok_response),
        ]
        mock_client.return_value.__aenter__.return_value = mock_http_client

        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1,
            messages=[ClaudeMessage(role="user", content="hi")],
        )

        gen = handle_antigravity_request(
            request,
            "antigravity/claude-sonnet-4-5",
            model_name="claude-sonnet-4-5",
        )
        chunk = await gen.__anext__()

        self.assertIn("candidates", chunk)
        self.assertEqual(mock_http_client.stream.call_count, 2)

    @patch("anthropic_proxy.antigravity.antigravity_auth")
    @patch("anthropic_proxy.antigravity.httpx.AsyncClient")
    async def test_handle_antigravity_request_rate_limit_fallback(self, mock_client, mock_auth):
        mock_auth.get_access_token = AsyncMock(return_value="token")
        mock_auth.get_project_id = MagicMock(return_value="project-1")

        for status_code, reason in ((429, "Too Many Requests"), (503, "Service Unavailable")):
            error_response = FakeStreamResponse(
                status_code=status_code,
                body=b"error",
                reason=reason,
            )
            ok_response = FakeStreamResponse(
                status_code=200,
                lines=['data: {"response":{"candidates":[]}}'],
            )

            mock_http_client = MagicMock()
            mock_http_client.stream.side_effect = [
                stream_context(error_response),
                stream_context(ok_response),
            ]
            mock_client.return_value.__aenter__.return_value = mock_http_client

            request = ClaudeMessagesRequest(
                model="claude-sonnet-4-5",
                max_tokens=1,
                messages=[ClaudeMessage(role="user", content="hi")],
            )

            gen = handle_antigravity_request(
                request,
                "antigravity/claude-sonnet-4-5",
                model_name="claude-sonnet-4-5",
            )
            chunk = await gen.__anext__()

            self.assertIn("candidates", chunk)
            self.assertEqual(mock_http_client.stream.call_count, 2)

    @patch("anthropic_proxy.antigravity.antigravity_auth")
    @patch("anthropic_proxy.antigravity.httpx.AsyncClient")
    async def test_antigravity_thinking_header_and_config(self, mock_client, mock_auth):
        mock_auth.get_access_token = AsyncMock(return_value="token")
        mock_auth.get_project_id = MagicMock(return_value="project-1")

        ok_response = FakeStreamResponse(
            status_code=200,
            lines=['data: {"response":{"candidates":[]}}'],
        )

        mock_http_client = MagicMock()
        mock_http_client.stream.return_value = stream_context(ok_response)
        mock_client.return_value.__aenter__.return_value = mock_http_client

        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5-thinking",
            max_tokens=1,
            messages=[ClaudeMessage(role="user", content="hi")],
        )

        gen = handle_antigravity_request(
            request,
            "antigravity/claude-sonnet-4-5-thinking",
            model_name="claude-sonnet-4-5-thinking",
        )
        await gen.__anext__()

        call_kwargs = mock_http_client.stream.call_args.kwargs
        headers = call_kwargs["headers"]
        payload = call_kwargs["json"]
        thinking_config = payload["request"]["generationConfig"]["thinkingConfig"]

        self.assertEqual(headers.get("anthropic-beta"), ANTIGRAVITY_THINKING_HEADER)
        self.assertEqual(headers.get("Accept"), "text/event-stream")
        self.assertIn("include_thoughts", thinking_config)
        self.assertIn("thinking_budget", thinking_config)

    @patch("anthropic_proxy.antigravity.antigravity_auth")
    @patch("anthropic_proxy.antigravity.httpx.AsyncClient")
    async def test_antigravity_image_model_skips_identity(self, mock_client, mock_auth):
        mock_auth.get_access_token = AsyncMock(return_value="token")
        mock_auth.get_project_id = MagicMock(return_value="project-1")

        ok_response = FakeStreamResponse(
            status_code=200,
            lines=['data: {"response":{"candidates":[]}}'],
        )

        mock_http_client = MagicMock()
        mock_http_client.stream.return_value = stream_context(ok_response)
        mock_client.return_value.__aenter__.return_value = mock_http_client

        request = ClaudeMessagesRequest(
            model="gemini-3-pro-image",
            max_tokens=1,
            messages=[ClaudeMessage(role="user", content="hi")],
        )

        gen = handle_antigravity_request(
            request,
            "antigravity/gemini-3-pro-image",
            model_name="gemini-3-pro-image",
        )
        await gen.__anext__()

        payload = mock_http_client.stream.call_args.kwargs["json"]
        self.assertNotIn("systemInstruction", payload["request"])


if __name__ == "__main__":
    unittest.main()
