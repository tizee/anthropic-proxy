import unittest
import tempfile
import os
import sys
import yaml
from pathlib import Path

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import load_models_config, CUSTOM_OPENAI_MODELS, is_codex_model

class TestCodexConfig(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.config_path = self.temp_file.name

    def tearDown(self):
        os.unlink(self.config_path)

    def test_load_codex_config(self):
        """Test loading a config with codex provider."""
        config_data = [
            {
                "model_id": "gpt-4o-codex",
                "model_name": "gpt-4o",
                "api_base": "https://chatgpt.com/backend-api/codex/responses",
                "provider": "codex",
                "can_stream": True
            },
            {
                "model_id": "claude-direct",
                "api_base": "https://api.anthropic.com",
                "direct": True
            }
        ]
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_data, f)
            
        load_models_config(self.config_path)
        
        # Verify Codex model
        self.assertIn("gpt-4o-codex", CUSTOM_OPENAI_MODELS)
        self.assertEqual(CUSTOM_OPENAI_MODELS["gpt-4o-codex"]["provider"], "codex")
        self.assertTrue(is_codex_model("gpt-4o-codex"))
        
        # Verify other model defaults
        self.assertEqual(CUSTOM_OPENAI_MODELS["claude-direct"]["provider"], "anthropic")
        self.assertFalse(is_codex_model("claude-direct"))

if __name__ == "__main__":
    unittest.main()
