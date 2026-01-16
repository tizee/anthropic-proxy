import unittest
from unittest.mock import patch
import os
import sys

# Add the parent directory to the sys.path to allow imports from the server module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import get_model_format, is_direct_mode_model


class TestFormatRouting(unittest.TestCase):
    def setUp(self):
        self.mock_models = {
            "model-anthropic": {"max_input_tokens": 50000, "format": "anthropic"},
            "model-openai": {"max_input_tokens": 30000, "format": "openai"},
            "model-legacy-direct": {"max_input_tokens": 30000, "direct": True},
        }

        patcher_models = patch.dict(
            "anthropic_proxy.client.CUSTOM_OPENAI_MODELS", self.mock_models, clear=True
        )
        self.addCleanup(patcher_models.stop)
        patcher_models.start()

    def test_get_model_format(self):
        """Test that format is returned for known models."""
        result = get_model_format("model-anthropic")
        self.assertEqual(result, "anthropic")

    def test_is_direct_mode_model_with_format_anthropic(self):
        """Test that a model with format=anthropic is identified as direct mode."""
        result = is_direct_mode_model("model-anthropic")
        self.assertTrue(result)

    def test_is_direct_mode_model_with_format_openai(self):
        """Test that a model with format=openai is not identified as direct mode."""
        result = is_direct_mode_model("model-openai")
        self.assertFalse(result)

    def test_is_direct_mode_model_with_legacy_direct_true(self):
        """Test that legacy direct=True is still recognized when format is missing."""
        result = is_direct_mode_model("model-legacy-direct")
        self.assertTrue(result)

    def test_is_direct_mode_model_unknown_model(self):
        """Test that an unknown model returns False for direct mode."""
        result = is_direct_mode_model("unknown-model")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
