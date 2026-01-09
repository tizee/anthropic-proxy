import unittest
from unittest.mock import patch
import os
import sys

# Add the parent directory to the sys.path to allow imports from the server module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import is_direct_mode_model


class TestDirectModeDetection(unittest.TestCase):
    def setUp(self):
        self.mock_models = {
            "model-direct": {"max_input_tokens": 50000, "direct": True},
            "model-openai-compatible": {"max_input_tokens": 30000, "direct": False},
        }

        patcher_models = patch.dict(
            "anthropic_proxy.client.CUSTOM_OPENAI_MODELS", self.mock_models, clear=True
        )
        self.addCleanup(patcher_models.stop)
        patcher_models.start()

    def test_is_direct_mode_model_with_explicit_direct_true(self):
        """Test that a model with direct=True is identified as direct mode."""
        result = is_direct_mode_model("model-direct")
        self.assertTrue(result)

    def test_is_direct_mode_model_with_explicit_direct_false(self):
        """Test that a model with direct=False is not identified as direct mode."""
        result = is_direct_mode_model("model-openai-compatible")
        self.assertFalse(result)

    def test_is_direct_mode_model_unknown_model(self):
        """Test that an unknown model returns False for direct mode."""
        result = is_direct_mode_model("unknown-model")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
