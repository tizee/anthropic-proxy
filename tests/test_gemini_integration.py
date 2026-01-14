import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import httpx
import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import is_gemini_model, load_gemini_models, CUSTOM_OPENAI_MODELS
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
        mock_auth._auth_data = {"refresh": "token"}
        
        load_gemini_models()
        
        self.assertIn("gemini-2.5-flash", CUSTOM_OPENAI_MODELS)
        self.assertEqual(CUSTOM_OPENAI_MODELS["gemini-2.5-flash"]["provider"], "gemini")

    @patch("anthropic_proxy.client.gemini_auth")
    def test_load_gemini_models_no_auth(self, mock_auth):
        # Simulate no auth
        mock_auth._auth_data = {}
        
        load_gemini_models()
        
        self.assertNotIn("gemini-2.5-flash", CUSTOM_OPENAI_MODELS)

    @patch("anthropic_proxy.gemini.gemini_auth")
    @patch("httpx.AsyncClient")
    async def test_handle_gemini_request_stream(self, mock_client_cls, mock_auth):
        # Setup Auth
        mock_auth.get_access_token = AsyncMock(return_value="access_token")
        mock_auth.get_project_id = MagicMock(return_value="project_id")
        
        # Setup Gemini Response stream
        mock_client = MagicMock() # Client itself shouldn't be AsyncMock to control methods
        # But we need to support 'async with client' if used as ctx, but here we instantiate it
        mock_client_cls.return_value = mock_client
        
        # aclose must be awaitable
        mock_client.aclose = AsyncMock()
        
        # httpx.AsyncClient.stream returns a Context Manager (not awaitable itself)
        # So stream should be a MagicMock
        mock_stream_ctx = MagicMock()
        mock_client.stream.return_value = mock_stream_ctx
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Context manager returns response on __aenter__
        mock_stream_ctx.__aenter__.return_value = mock_response
        
        # Sample Gemini SSE response lines
        # data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}]}}
        
        gemini_chunk = {
            "response": {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hello"}]},
                        "finishReason": "STOP"
                    }
                ]
            }
        }
        
        lines = [
            f"data: {json.dumps(gemini_chunk)}",
            "", # empty line
            "data: [DONE]" # Not standard Gemini but good for testing loop break if we handled it, but logic parses json
        ]
        
        async def aiter_lines():
            for line in lines:
                yield line
                
        mock_response.aiter_lines = aiter_lines
        mock_client.stream.return_value.__aenter__.return_value = mock_response
        
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

if __name__ == "__main__":
    unittest.main()
