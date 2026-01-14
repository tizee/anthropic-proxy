import unittest
from unittest.mock import patch

from anthropic_proxy.client import CUSTOM_OPENAI_MODELS, is_gemini_model, load_gemini_models


class TestGeminiClient(unittest.TestCase):
    def setUp(self):
        CUSTOM_OPENAI_MODELS.clear()

    def test_is_gemini_model(self):
        CUSTOM_OPENAI_MODELS["gemini-test"] = {"provider": "gemini"}
        CUSTOM_OPENAI_MODELS["other-test"] = {"provider": "openai"}

        self.assertTrue(is_gemini_model("gemini-test"))
        self.assertFalse(is_gemini_model("other-test"))
        self.assertFalse(is_gemini_model("unknown"))

    @patch("anthropic_proxy.client.gemini_auth")
    def test_load_gemini_models_with_auth(self, mock_auth):
        mock_auth.has_auth.return_value = True

        load_gemini_models()

        self.assertIn("gemini/gemini-2.5-flash", CUSTOM_OPENAI_MODELS)
        self.assertEqual(CUSTOM_OPENAI_MODELS["gemini/gemini-2.5-flash"]["provider"], "gemini")

    @patch("anthropic_proxy.client.gemini_auth")
    def test_load_gemini_models_no_auth(self, mock_auth):
        mock_auth.has_auth.return_value = False

        load_gemini_models()

        self.assertNotIn("gemini/gemini-2.5-flash", CUSTOM_OPENAI_MODELS)


if __name__ == "__main__":
    unittest.main()
