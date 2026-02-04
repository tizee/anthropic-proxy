"""
Unit tests for the /openai/v1/responses endpoint.

This endpoint provides direct OpenAI Responses API access to Codex backend.
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import os
import sys
import json

# Add the parent directory to the sys.path to allow imports from the server module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import is_codex_model
from anthropic_proxy.server import create_response


class DummyRequest:
    """Mock request object for testing."""

    def __init__(self, raw_body: bytes, headers=None):
        self._body = raw_body
        self.headers = headers or {}
        self.url = type("Url", (), {"path": "/openai/v1/responses"})()

    async def body(self):
        return self._body


class TestCodexResponsesEndpointValidation(unittest.IsolatedAsyncioTestCase):
    """Tests for request validation in the responses endpoint."""

    async def test_unknown_model_returns_400(self):
        """Test that unknown model returns 400 error."""
        with patch.dict(
            "anthropic_proxy.server.CUSTOM_OPENAI_MODELS", {}, clear=True
        ):
            body = json.dumps(
                {
                    "model": "unknown-model",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")

            with self.assertRaises(Exception) as context:
                await create_response(DummyRequest(body))

            self.assertIn("400", str(context.exception))

    async def test_non_codex_model_returns_400(self):
        """Test that non-Codex model returns 400 error."""
        mock_models = {
            "openai-model": {
                "model_id": "openai-model",
                "model_name": "gpt-4",
                "provider": "openai",
            }
        }

        with patch.dict(
            "anthropic_proxy.server.CUSTOM_OPENAI_MODELS", mock_models, clear=True
        ):
            body = json.dumps(
                {
                    "model": "openai-model",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")

            with self.assertRaises(Exception) as context:
                await create_response(DummyRequest(body))

            error_msg = str(context.exception)
            self.assertIn("400", error_msg)
            self.assertIn("not a Codex model", error_msg)


class TestCodexResponsesEndpointStreaming(unittest.IsolatedAsyncioTestCase):
    """Tests for streaming mode in the responses endpoint."""

    async def test_streaming_mode_returns_streaming_response(self):
        """Test that streaming mode returns StreamingResponse."""
        model_id = "codex/gpt-5.2-codex"
        mock_models = {
            model_id: {
                "model_id": model_id,
                "model_name": "gpt-5.2-codex",
                "provider": "codex",
            }
        }

        async def mock_generator():
            yield {"id": "chunk-1", "object": "response.output_item.delta"}
            yield {"id": "chunk-2", "object": "response.output_item.delta"}

        with patch.dict(
            "anthropic_proxy.server.CUSTOM_OPENAI_MODELS", mock_models, clear=True
        ), patch(
            "anthropic_proxy.server.handle_codex_request", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = mock_generator()

            body = json.dumps(
                {
                    "model": model_id,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")

            from fastapi.responses import StreamingResponse

            response = await create_response(DummyRequest(body))

            self.assertIsInstance(response, StreamingResponse)
            self.assertEqual(response.media_type, "text/event-stream")

            # Verify handle_codex_request was called
            mock_handle.assert_called_once()
            call_args = mock_handle.call_args[0]
            self.assertEqual(call_args[1], model_id)

    async def test_streaming_response_headers(self):
        """Test that streaming response has correct headers."""
        model_id = "codex/gpt-5.2-codex"
        mock_models = {
            model_id: {
                "model_id": model_id,
                "model_name": "gpt-5.2-codex",
                "provider": "codex",
            }
        }

        async def mock_generator():
            yield {"id": "chunk-1"}

        with patch.dict(
            "anthropic_proxy.server.CUSTOM_OPENAI_MODELS", mock_models, clear=True
        ), patch(
            "anthropic_proxy.server.handle_codex_request", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = mock_generator()

            body = json.dumps(
                {
                    "model": model_id,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")

            response = await create_response(DummyRequest(body))

            # Check CORS and streaming headers
            self.assertEqual(response.headers.get("Cache-Control"), "no-cache")
            self.assertEqual(response.headers.get("Connection"), "keep-alive")
            self.assertEqual(response.headers.get("Access-Control-Allow-Origin"), "*")


class TestCodexResponsesEndpointNonStreaming(unittest.IsolatedAsyncioTestCase):
    """Tests for non-streaming mode in the responses endpoint."""

    async def test_non_streaming_returns_json_response(self):
        """Test that non-streaming mode returns JSONResponse."""
        model_id = "codex/gpt-5.2-codex"
        mock_models = {
            model_id: {
                "model_id": model_id,
                "model_name": "gpt-5.2-codex",
                "provider": "codex",
            }
        }

        expected_response = {
            "id": "resp-123",
            "object": "response",
            "output": [{"type": "message", "content": "Hello!"}],
        }

        async def mock_generator():
            yield {"id": "chunk-1", "partial": True}
            yield expected_response

        with patch.dict(
            "anthropic_proxy.server.CUSTOM_OPENAI_MODELS", mock_models, clear=True
        ), patch(
            "anthropic_proxy.server.handle_codex_request", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = mock_generator()

            body = json.dumps(
                {
                    "model": model_id,
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")

            from fastapi.responses import JSONResponse

            response = await create_response(DummyRequest(body))

            self.assertIsInstance(response, JSONResponse)

            # Parse response body
            response_body = json.loads(response.body.decode())
            self.assertEqual(response_body, expected_response)

    async def test_non_streaming_empty_chunks_raises_500(self):
        """Test that empty response from Codex raises 500 error."""
        model_id = "codex/gpt-5.2-codex"
        mock_models = {
            model_id: {
                "model_id": model_id,
                "model_name": "gpt-5.2-codex",
                "provider": "codex",
            }
        }

        async def mock_generator():
            if False:  # Never yields
                yield None

        with patch.dict(
            "anthropic_proxy.server.CUSTOM_OPENAI_MODELS", mock_models, clear=True
        ), patch(
            "anthropic_proxy.server.handle_codex_request", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = mock_generator()

            body = json.dumps(
                {
                    "model": model_id,
                    "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")

            with self.assertRaises(Exception) as context:
                await create_response(DummyRequest(body))

            self.assertIn("500", str(context.exception))


class TestCodexResponsesEndpointModelOverride(unittest.IsolatedAsyncioTestCase):
    """Tests for model name override behavior."""

    async def test_model_name_override_in_request(self):
        """Test that model name is overridden to actual Codex model name."""
        model_id = "codex/gpt-5.2-codex"
        mock_models = {
            model_id: {
                "model_id": model_id,
                "model_name": "gpt-5.2-codex",
                "provider": "codex",
            }
        }

        async def mock_generator():
            yield {"id": "chunk-1"}

        captured_request = None

        async def capture_handle_codex(request, mid):
            nonlocal captured_request
            captured_request = request
            async def gen():
                yield {"id": "chunk-1"}
            return gen()

        with patch.dict(
            "anthropic_proxy.server.CUSTOM_OPENAI_MODELS", mock_models, clear=True
        ), patch(
            "anthropic_proxy.server.handle_codex_request", new=capture_handle_codex
        ):
            body = json.dumps(
                {
                    "model": model_id,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")

            await create_response(DummyRequest(body))

            self.assertIsNotNone(captured_request)
            # Model name should be overridden to the actual Codex model name
            self.assertEqual(captured_request["model"], "gpt-5.2-codex")


class TestCodexResponsesEndpointIsCodexModel(unittest.TestCase):
    """Tests for is_codex_model function used by the endpoint."""

    def setUp(self):
        self.mock_models = {
            "model-codex": {"provider": "codex", "model_name": "gpt-5.2-codex"},
            "model-openai": {"provider": "openai", "model_name": "gpt-4"},
            "model-default": {"model_name": "some-model"},
        }

        patcher_models = patch.dict(
            "anthropic_proxy.client.CUSTOM_OPENAI_MODELS", self.mock_models, clear=True
        )
        self.addCleanup(patcher_models.stop)
        patcher_models.start()

    def test_is_codex_model_true(self):
        """Test that a model with provider='codex' is identified as codex model."""
        result = is_codex_model("model-codex")
        self.assertTrue(result)

    def test_is_codex_model_false(self):
        """Test that a model with other provider is not identified as codex model."""
        result = is_codex_model("model-openai")
        self.assertFalse(result)

    def test_is_codex_model_default(self):
        """Test that a model without provider is not identified as codex model."""
        result = is_codex_model("model-default")
        self.assertFalse(result)

    def test_is_codex_model_unknown(self):
        """Test that unknown model is not codex model."""
        result = is_codex_model("unknown")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
