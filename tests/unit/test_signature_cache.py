import unittest
from unittest.mock import patch

from anthropic_proxy.converters import anthropic_to_gemini_request
from anthropic_proxy.signature_cache import (
    cache_signature,
    clear_all_cache,
    get_cached_signature,
    _CACHE_TTL_SECONDS,
)
from anthropic_proxy.types import (
    ClaudeContentBlockThinking,
    ClaudeMessage,
    ClaudeMessagesRequest,
)


class TestSignatureCache(unittest.TestCase):
    def setUp(self):
        clear_all_cache()

    def tearDown(self):
        clear_all_cache()

    def test_cache_round_trip(self):
        cache_signature("sess-1", "think", "sig-1")
        self.assertEqual(get_cached_signature("sess-1", "think"), "sig-1")

    def test_cache_expired(self):
        with patch("anthropic_proxy.signature_cache.time.time", side_effect=[0, _CACHE_TTL_SECONDS + 1]):
            cache_signature("sess-2", "think", "sig-2")
            self.assertIsNone(get_cached_signature("sess-2", "think"))

    def test_restore_signature_for_gemini(self):
        cache_signature("sess-3", "thought-text", "sig-3")
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5-thinking",
            max_tokens=1,
            messages=[
                ClaudeMessage(
                    role="assistant",
                    content=[
                        ClaudeContentBlockThinking(
                            type="thinking",
                            thinking="thought-text",
                            signature=None,
                        )
                    ],
                )
            ],
        )

        body = anthropic_to_gemini_request(
            request,
            model_id="claude-sonnet-4-5-thinking",
            session_id="sess-3",
        )

        parts = body["contents"][0]["parts"]
        self.assertEqual(parts[0]["text"], "thought-text")
        self.assertEqual(body["sessionId"], "sess-3")


if __name__ == "__main__":
    unittest.main()
