import unittest

from anthropic_proxy.signature_cache import cache_tool_signature, clear_all_cache
from anthropic_proxy.types import ClaudeContentBlockToolUse, ClaudeMessage

from anthropic_proxy.gemini_converter import anthropic_to_gemini_sdk_params
from anthropic_proxy.types import ClaudeMessagesRequest


class TestGeminiConverter(unittest.TestCase):
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

    def test_tool_use_includes_cached_thought_signature(self):
        clear_all_cache()
        cache_tool_signature("sess-1", "tool-1", "sig-1")

        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=16,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="tool-1",
                            name="get_weather",
                            input={"city": "SF"},
                        )
                    ],
                )
            ],
        )

        _, _, body = anthropic_to_gemini_sdk_params(
            request,
            "gemini-2.5-pro",
            session_id="sess-1",
        )

        parts = body["contents"][0]["parts"]
        self.assertEqual(parts[0]["thoughtSignature"], "sig-1")

        clear_all_cache()

    def test_tool_use_includes_cached_thought_signature_without_session(self):
        clear_all_cache()
        cache_tool_signature(None, "tool-2", "sig-2")

        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=16,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockToolUse(
                            type="tool_use",
                            id="tool-2",
                            name="get_weather",
                            input={"city": "SF"},
                        )
                    ],
                )
            ],
        )

        _, _, body = anthropic_to_gemini_sdk_params(
            request,
            "gemini-2.5-pro",
        )

        parts = body["contents"][0]["parts"]
        self.assertEqual(parts[0]["thoughtSignature"], "sig-2")

        clear_all_cache()


if __name__ == "__main__":
    unittest.main()
