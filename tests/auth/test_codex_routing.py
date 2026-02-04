import unittest
from unittest.mock import patch, AsyncMock
import os
import sys
import json

# Add the parent directory to the sys.path to allow imports from the server module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import is_codex_model, load_models_config
from anthropic_proxy.server import create_message

class TestCodexRouting(unittest.TestCase):
    def setUp(self):
        self.mock_models = {
            "model-codex": {"provider": "codex", "direct": False},
            "model-openai": {"provider": "openai", "direct": False},
            "model-default": {"direct": False},
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

class TestCodexModelNameRouting(unittest.IsolatedAsyncioTestCase):
    async def test_codex_model_rejected_on_anthropic_endpoint(self):
        """Codex models must use /openai/v1/responses, not /anthropic/v1/messages."""
        model_id = "codex/gpt-5.2-codex"
        mock_models = {
            model_id: {
                "model_id": model_id,
                "model_name": "gpt-5.2-codex",
                "api_base": "https://chatgpt.com/backend-api/codex/responses",
                "api_key": "dummy",
                "can_stream": True,
                "format": "openai",
                "direct": False,
                "provider": "codex",
                "extra_headers": {},
                "extra_body": {},
                "temperature": 1.0,
                "reasoning_effort": None,
                "max_tokens": 8192,
                "context": 128000,
                "max_input_tokens": 128000,
            }
        }

        with patch.dict(
            "anthropic_proxy.server.CUSTOM_OPENAI_MODELS", mock_models, clear=True
        ):
            body = json.dumps(
                {
                    "model": model_id,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode("utf-8")

            class DummyRequest:
                headers = {"Authorization": "Bearer sk-test"}
                url = type("Url", (), {"path": "/v1/messages"})()

                def __init__(self, raw_body: bytes):
                    self._body = raw_body

                async def body(self):
                    return self._body

            with self.assertRaises(Exception) as context:
                await create_message(DummyRequest(body))

            error_msg = str(context.exception)
            self.assertIn("400", error_msg)
            self.assertIn("/openai/v1/responses", error_msg)

if __name__ == "__main__":
    unittest.main()
