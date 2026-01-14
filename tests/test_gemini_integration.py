import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import is_gemini_model, load_gemini_models, CUSTOM_OPENAI_MODELS
from anthropic_proxy.converter import clean_gemini_schema
from anthropic_proxy.gemini_converter import anthropic_to_gemini_sdk_params
from anthropic_proxy.gemini_sdk import _build_http_options, stream_gemini_sdk_request
from google.genai import types as genai_types
from anthropic_proxy.gemini import handle_gemini_request, GeminiAuth
from anthropic_proxy.types import ClaudeMessagesRequest
from anthropic_proxy.cli import cmd_login, parse_args

class TestGeminiIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Clear models
        CUSTOM_OPENAI_MODELS.clear()
        
    def test_is_gemini_model(self):
        CUSTOM_OPENAI_MODELS["gemini-test"] = {"provider": "gemini"}
        CUSTOM_OPENAI_MODELS["other-test"] = {"provider": "openai"}
        
        self.assertTrue(is_gemini_model("gemini-test"))
        self.assertFalse(is_gemini_model("other-test"))
        self.assertFalse(is_gemini_model("unknown"))

    @patch("anthropic_proxy.client.gemini_auth")
    def test_load_gemini_models_with_auth(self, mock_auth):
        # Simulate auth present
        mock_auth.has_auth.return_value = True
        
        load_gemini_models()
        
        self.assertIn("gemini/gemini-2.5-flash", CUSTOM_OPENAI_MODELS)
        self.assertEqual(CUSTOM_OPENAI_MODELS["gemini/gemini-2.5-flash"]["provider"], "gemini")

    @patch("anthropic_proxy.client.gemini_auth")
    def test_load_gemini_models_no_auth(self, mock_auth):
        # Simulate no auth
        mock_auth.has_auth.return_value = False
        
        load_gemini_models()
        
        self.assertNotIn("gemini/gemini-2.5-flash", CUSTOM_OPENAI_MODELS)

    @patch("anthropic_proxy.gemini.stream_gemini_sdk_request")
    @patch("anthropic_proxy.gemini.gemini_auth")
    async def test_handle_gemini_request_stream(self, mock_auth, mock_stream):
        # Setup Auth
        mock_auth.get_access_token = AsyncMock(return_value="access_token")
        mock_auth.get_project_id = MagicMock(return_value="project_id")
        gemini_chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}]},
                    "finishReason": "STOP",
                }
            ]
        }

        async def fake_stream(**kwargs):
            yield gemini_chunk

        mock_stream.side_effect = lambda **kwargs: fake_stream(**kwargs)
        
        # Run Handler
        claude_req = ClaudeMessagesRequest(
            model="gemini-flash",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
        )
        
        chunks = []
        async for chunk in handle_gemini_request(claude_req, "model-id"):
            chunks.append(chunk)
            
        # Verify
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk["candidates"][0]["content"]["parts"][0]["text"], "Hello")
        self.assertEqual(chunk["candidates"][0]["finishReason"], "STOP")

    def test_cli_login_gemini(self):
        with patch("sys.argv", ["anthropic-proxy", "login", "--gemini"]):
            args = parse_args()
            self.assertTrue(args.gemini)

            with patch("anthropic_proxy.gemini.gemini_auth.login") as mock_login:
                cmd_login(args)
                mock_login.assert_called_once()

    def test_gemini_sdk_params_flatten_generation_config(self):
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=123,
            messages=[{"role": "user", "content": "hi"}],
        )

        _, config, _ = anthropic_to_gemini_sdk_params(request, "gemini-2.5-pro")

        self.assertNotIn("generation_config", config)
        self.assertIn("temperature", config)
        self.assertIn("top_p", config)
        self.assertIn("top_k", config)
        self.assertEqual(config["max_output_tokens"], 123)

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

            def raise_for_status(self):
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

    def test_clean_gemini_schema_strips_code_assist_unsupported_keys(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {"type": "number", "exclusiveMinimum": 0},
                "name": {"type": "string"},
            },
            "propertyNames": {"pattern": "^[a-z]+$"},
        }

        cleaned = clean_gemini_schema(schema)

        self.assertNotIn("exclusiveMinimum", cleaned["properties"]["value"])
        self.assertNotIn("propertyNames", cleaned)
        self.assertIn("exclusiveMinimum: 0", cleaned["properties"]["value"]["description"])

if __name__ == "__main__":
    unittest.main()
