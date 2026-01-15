import unittest

from anthropic_proxy.signature_cache import cache_tool_signature, clear_all_cache
from anthropic_proxy.types import ClaudeContentBlockToolUse, ClaudeMessage
from anthropic_proxy.converters import ensure_tool_ids
from anthropic_proxy.converters import anthropic_to_gemini_sdk_params
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

    def test_ensure_tool_ids_assigns_ids_to_function_calls(self):
        """Test that ensure_tool_ids assigns IDs to functionCall parts."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}}},
                    {"functionCall": {"name": "get_time", "args": {}}},
                ],
            }
        ]
        result = ensure_tool_ids(contents)

        # Check that IDs were assigned
        parts = result[0]["parts"]
        self.assertEqual(parts[0]["functionCall"]["id"], "tool-call-1")
        self.assertEqual(parts[1]["functionCall"]["id"], "tool-call-2")

    def test_ensure_tool_ids_preserves_existing_ids(self):
        """Test that ensure_tool_ids preserves existing functionCall IDs."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}, "id": "custom-id"}}
                ],
            }
        ]
        result = ensure_tool_ids(contents)

        # Check that existing ID was preserved
        self.assertEqual(result[0]["parts"][0]["functionCall"]["id"], "custom-id")

    def test_ensure_tool_ids_matches_response_ids_fifo(self):
        """Test that ensure_tool_ids matches functionResponse IDs using FIFO order."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}}},
                    {"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "get_weather", "response": {"content": "72F"}}},
                    {"functionResponse": {"name": "get_weather", "response": {"content": "45F"}}},
                ],
            },
        ]
        result = ensure_tool_ids(contents)

        # Check that functionResponses got the correct IDs in FIFO order
        self.assertEqual(result[1]["parts"][0]["functionResponse"]["id"], "tool-call-1")
        self.assertEqual(result[1]["parts"][1]["functionResponse"]["id"], "tool-call-2")

    def test_ensure_tool_ids_preserves_existing_response_ids(self):
        """Test that ensure_tool_ids preserves existing functionResponse IDs."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "get_weather", "id": "my-id", "response": {"content": "72F"}}},
                ],
            },
        ]
        result = ensure_tool_ids(contents)

        # Check that existing response ID was preserved
        self.assertEqual(result[1]["parts"][0]["functionResponse"]["id"], "my-id")

    def test_ensure_tool_ids_handles_multiple_function_names(self):
        """Test that ensure_tool_ids handles calls to different functions."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}}},
                    {"functionCall": {"name": "get_time", "args": {}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "get_weather", "response": {"content": "72F"}}},
                    {"functionResponse": {"name": "get_time", "response": {"content": "10:30 AM"}}},
                ],
            },
        ]
        result = ensure_tool_ids(contents)

        # Check that each functionResponse got the correct ID
        self.assertEqual(result[1]["parts"][0]["functionResponse"]["id"], "tool-call-1")
        self.assertEqual(result[1]["parts"][1]["functionResponse"]["id"], "tool-call-2")

    def test_ensure_tool_ids_handles_mixed_parts(self):
        """Test that ensure_tool_ids handles mixed functionCall and text parts."""
        contents = [
            {
                "role": "model",
                "parts": [
                    {"text": "I'll check the weather."},
                    {"functionCall": {"name": "get_weather", "args": {"city": "SF"}}},
                ],
            },
        ]
        result = ensure_tool_ids(contents)

        # Check that text part was preserved and functionCall got an ID
        self.assertEqual(result[0]["parts"][0]["text"], "I'll check the weather.")
        self.assertEqual(result[0]["parts"][1]["functionCall"]["id"], "tool-call-1")

    def test_ensure_tool_ids_handles_empty_contents(self):
        """Test that ensure_tool_ids handles empty contents."""
        result = ensure_tool_ids([])
        self.assertEqual(result, [])

    def test_ensure_tool_ids_handles_contents_without_parts(self):
        """Test that ensure_tool_ids handles messages without parts."""
        contents = [{"role": "user"}, {"role": "model"}]
        result = ensure_tool_ids(contents)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["role"], "model")


if __name__ == "__main__":
    unittest.main()
