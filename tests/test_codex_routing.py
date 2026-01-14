import unittest
from unittest.mock import patch
import os
import sys

# Add the parent directory to the sys.path to allow imports from the server module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import is_codex_model, load_models_config

class TestCodexRouting(unittest.TestCase):
    def setUp(self):
        self.mock_models = {
            "model-codex": {"provider": "codex", "direct": False},
            "model-openai": {"provider": "openai", "direct": False},
            "model-default": {"direct": False}, # Defaults to openai
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
