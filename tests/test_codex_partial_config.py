import unittest
import tempfile
import os
import sys
import yaml
from pathlib import Path

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import load_models_config, CUSTOM_OPENAI_MODELS, CODEX_API_URL

class TestCodexPartialConfig(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.config_path = self.temp_file.name
        CUSTOM_OPENAI_MODELS.clear()

    def tearDown(self):
        os.unlink(self.config_path)

    def test_partial_codex_config(self):
        """Test that a partial codex config is hydrated with defaults IF provider is codex."""
        config_data = [
            {
                "model_id": "gpt-5.2-codex",
                "provider": "codex", # Explicit provider required
                "reasoning_effort": "low", # Override default (medium)
                "context": "16K" # Custom context
            }
        ]
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
            
        load_models_config(self.config_path)
        
        model = CUSTOM_OPENAI_MODELS.get("gpt-5.2-codex")
        self.assertIsNotNone(model)
        
        # Check injected defaults
        self.assertEqual(model["api_base"], CODEX_API_URL)
        self.assertEqual(model["provider"], "codex")
        self.assertEqual(model["api_key"], "codex-auth")
        
        # Check user overrides
        self.assertEqual(model["reasoning_effort"], "low")
        self.assertEqual(model["context"], 16000)

    def test_missing_provider_ignored(self):
        """Test that a model without provider: codex is NOT hydrated with defaults."""
        config_data = [
            {
                "model_id": "gpt-5.2-codex",
                # Missing provider: codex
                "reasoning_effort": "low"
            }
        ]
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
            
        load_models_config(self.config_path)
        
        # Should be rejected because api_base is missing
        self.assertNotIn("gpt-5.2-codex", CUSTOM_OPENAI_MODELS)

    def test_unknown_model_no_defaults(self):
        """Test that an unknown model is still rejected if api_base is missing."""
        config_data = [
            {
                "model_id": "unknown-model",
                "reasoning_effort": "low"
            }
        ]
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
            
        load_models_config(self.config_path)
        
        self.assertNotIn("unknown-model", CUSTOM_OPENAI_MODELS)

if __name__ == "__main__":
    unittest.main()
