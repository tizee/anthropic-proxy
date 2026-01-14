import json
import unittest
from unittest.mock import Mock, patch

from fastapi.responses import StreamingResponse

from anthropic_proxy.server import create_message


class TestGeminiRouting(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_streaming_does_not_await_generator(self):
        model_id = "gemini/gemini-3-pro-preview"
        mock_models = {
            model_id: {
                "model_id": model_id,
                "model_name": "gemini-3-pro-preview",
                "api_base": "https://cloudcode-pa.googleapis.com",
                "api_key": "dummy",
                "can_stream": True,
                "format": "gemini",
                "direct": False,
                "provider": "gemini",
                "extra_headers": {},
                "extra_body": {},
                "temperature": 1.0,
                "reasoning_effort": None,
                "max_tokens": 8192,
                "context": 128000,
                "max_input_tokens": 128000,
            }
        }

        async def dummy_gen():
            if False:
                yield None

        body = json.dumps(
            {
                "model": model_id,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode("utf-8")

        class DummyRequest:
            headers = {}
            url = type("Url", (), {"path": "/v1/messages"})()

            def __init__(self, raw_body: bytes):
                self._body = raw_body

            async def body(self):
                return self._body

        with patch.dict(
            "anthropic_proxy.server.CUSTOM_OPENAI_MODELS", mock_models, clear=True
        ), patch(
            "anthropic_proxy.server.handle_gemini_request", new=Mock(return_value=dummy_gen())
        ):
            response = await create_message(DummyRequest(body))

        self.assertIsInstance(response, StreamingResponse)


if __name__ == "__main__":
    unittest.main()
