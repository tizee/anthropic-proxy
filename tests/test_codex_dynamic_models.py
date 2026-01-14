import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import load_codex_models, CUSTOM_OPENAI_MODELS, is_codex_model

class TestCodexDynamicModels(unittest.TestCase):
    def setUp(self):
        # Clear models before each test
        CUSTOM_OPENAI_MODELS.clear()

    @patch("anthropic_proxy.client.codex_auth")
    def test_dynamic_injection_with_auth(self, mock_auth):
        # Simulate auth data present
        mock_auth.has_auth.return_value = True
        
        load_codex_models()
        
        # Verify models are registered
        self.assertIn("codex/gpt-5.2-codex", CUSTOM_OPENAI_MODELS)
        self.assertTrue(is_codex_model("codex/gpt-5.2-codex"))
        self.assertEqual(CUSTOM_OPENAI_MODELS["codex/gpt-5.2-codex"]["provider"], "codex")
        self.assertEqual(CUSTOM_OPENAI_MODELS["codex/gpt-5.2-codex"]["reasoning_effort"], "high")

    @patch("anthropic_proxy.client.codex_auth")
    def test_no_injection_without_auth(self, mock_auth):
        # Simulate no auth data
        mock_auth.has_auth.return_value = False
        
        load_codex_models()
        
        # Verify no models registered
        self.assertNotIn("codex/gpt-5.2-codex", CUSTOM_OPENAI_MODELS)

    @patch("anthropic_proxy.client.codex_auth")
    def test_no_overwrite_user_config(self, mock_auth):
        # Simulate auth data present
        mock_auth.has_auth.return_value = True
        
        # Pre-register a model (simulating models.yaml)
        CUSTOM_OPENAI_MODELS["codex/gpt-5.2-codex"] = {
            "model_id": "codex/gpt-5.2-codex",
            "provider": "user-defined", # Different provider to prove no overwrite
            "reasoning_effort": "custom"
        }
        
        load_codex_models()
        
        # Verify user config was preserved
        self.assertEqual(CUSTOM_OPENAI_MODELS["codex/gpt-5.2-codex"]["provider"], "user-defined")
        self.assertEqual(CUSTOM_OPENAI_MODELS["codex/gpt-5.2-codex"]["reasoning_effort"], "custom")
        
        # Verify other models were still injected
        self.assertIn("codex/gpt-5.2-codex", CUSTOM_OPENAI_MODELS)

if __name__ == "__main__":
    unittest.main()
