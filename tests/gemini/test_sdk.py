import json
import unittest
from unittest.mock import patch

from google.genai import types as genai_types

from anthropic_proxy.gemini_sdk import _build_http_options, stream_gemini_sdk_request
from anthropic_proxy.gemini_types import parse_gemini_response
from anthropic_proxy.types import ClaudeMessagesRequest


class TestGeminiSDK(unittest.IsolatedAsyncioTestCase):
    def test_gemini_http_options_embed_api_version(self):
        http_options = _build_http_options(
            base_url="https://cloudcode-pa.googleapis.com",
            headers={"Authorization": "Bearer token"},
            extra_body=None,
            api_version="v1internal",
            base_url_resource_scope=genai_types.ResourceScope.COLLECTION,
            embed_api_version_in_base_url=True,
        )

        self.assertEqual(
            http_options.base_url, "https://cloudcode-pa.googleapis.com/v1internal"
        )
        self.assertIsNone(http_options.api_version)
        self.assertEqual(
            http_options.base_url_resource_scope, genai_types.ResourceScope.COLLECTION
        )

    @patch("anthropic_proxy.gemini_sdk.httpx.AsyncClient")
    async def test_gemini_code_assist_stream_unwraps_response(self, mock_client):
        class DummyResponse:
            status_code = 200
            reason_phrase = "OK"

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def aiter_lines(self):
                yield 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}}'
                yield "data: [DONE]"

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def stream(self, method, url, **kwargs):
                self.method = method
                self.url = url
                self.kwargs = kwargs
                return DummyResponse()

        mock_client.return_value = DummyClient()

        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )

        chunks = []
        async for chunk in stream_gemini_sdk_request(
            request=request,
            model_id="gemini-2.5-pro",
            access_token="token",
            project_id="project-123",
            base_url="https://cloudcode-pa.googleapis.com",
            extra_headers={},
            is_antigravity=False,
            use_code_assist=True,
        ):
            chunks.append(chunk)

        self.assertEqual(mock_client.return_value.method, "POST")
        self.assertIn("v1internal:streamGenerateContent", mock_client.return_value.url)
        self.assertTrue(chunks)
        self.assertEqual(
            chunks[0]["candidates"][0]["content"]["parts"][0]["text"], "hi"
        )

    @patch("anthropic_proxy.gemini_sdk.httpx.AsyncClient")
    async def test_gemini_code_assist_injects_tool_config(self, mock_client):
        class DummyResponse:
            status_code = 200
            reason_phrase = "OK"

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def aiter_lines(self):
                yield "data: [DONE]"

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def stream(self, method, url, **kwargs):
                self.method = method
                self.url = url
                self.kwargs = kwargs
                return DummyResponse()

        mock_client.return_value = DummyClient()

        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )

        async for _chunk in stream_gemini_sdk_request(
            request=request,
            model_id="gemini-2.5-pro",
            access_token="token",
            project_id="project-123",
            base_url="https://cloudcode-pa.googleapis.com",
            extra_headers={},
            is_antigravity=False,
            use_code_assist=True,
        ):
            pass

        request_body = mock_client.return_value.kwargs["json"]["request"]
        self.assertIn("toolConfig", request_body)
        self.assertEqual(
            request_body["toolConfig"]["functionCallingConfig"]["mode"], "VALIDATED"
        )

    @patch("anthropic_proxy.gemini_sdk.httpx.AsyncClient")
    async def test_gemini_code_assist_normalizes_function_calls(self, mock_client):
        class DummyResponse:
            status_code = 200
            reason_phrase = "OK"

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            async def aiter_lines(self):
                yield 'data: {"response": {"functionCalls": [{"name": "get_weather", "args": {"city": "SF"}}]}}'
                yield "data: [DONE]"

        class DummyClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return None

            def stream(self, method, url, **kwargs):
                self.method = method
                self.url = url
                self.kwargs = kwargs
                return DummyResponse()

        mock_client.return_value = DummyClient()

        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )

        chunks = []
        async for chunk in stream_gemini_sdk_request(
            request=request,
            model_id="gemini-2.5-pro",
            access_token="token",
            project_id="project-123",
            base_url="https://cloudcode-pa.googleapis.com",
            extra_headers={},
            is_antigravity=False,
            use_code_assist=True,
        ):
            chunks.append(chunk)

        self.assertEqual(
            chunks[0]["candidates"][0]["content"]["parts"][0]["functionCall"]["name"],
            "get_weather",
        )

    def test_normalize_function_calls_from_sdk_types(self):
        call = genai_types.FunctionCall(
            name="get_weather",
            args={"city": "SF"},
            id="call-1",
        )
        payload = {"functionCalls": [call.model_dump(exclude_none=True)]}

        normalized = parse_gemini_response(payload)
        parts = normalized["candidates"][0]["content"]["parts"]
        self.assertEqual(parts[0]["functionCall"]["name"], "get_weather")
        self.assertEqual(parts[0]["functionCall"]["args"]["city"], "SF")
        self.assertEqual(parts[0]["functionCall"]["id"], "call-1")

    def test_normalize_function_calls_from_snake_case(self):
        call = genai_types.FunctionCall(
            name="get_weather",
            args={"city": "SF"},
        )
        payload = {"function_calls": [call.model_dump(exclude_none=True)]}

        normalized = parse_gemini_response(payload)
        parts = normalized["candidates"][0]["content"]["parts"]
        self.assertEqual(parts[0]["functionCall"]["name"], "get_weather")

    def test_normalize_function_calls_from_part_function_call(self):
        call = genai_types.FunctionCall(
            name="get_weather",
            args={"city": "SF"},
        )
        payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "function_call": call.model_dump(exclude_none=True),
                            }
                        ]
                    }
                }
            ]
        }

        normalized = parse_gemini_response(payload)
        parts = normalized["candidates"][0]["content"]["parts"]
        self.assertIn("functionCall", parts[0])
        self.assertEqual(parts[0]["functionCall"]["name"], "get_weather")

    def test_normalize_function_calls_parses_json_args(self):
        payload = {
            "functionCalls": [
                {"name": "get_weather", "args": json.dumps({"city": "SF"})}
            ]
        }

        normalized = parse_gemini_response(payload)
        parts = normalized["candidates"][0]["content"]["parts"]
        self.assertEqual(parts[0]["functionCall"]["args"]["city"], "SF")


if __name__ == "__main__":
    unittest.main()
