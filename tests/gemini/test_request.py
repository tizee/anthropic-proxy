import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from anthropic_proxy.gemini import handle_gemini_request
from anthropic_proxy.types import ClaudeMessagesRequest


class TestGeminiRequest(unittest.IsolatedAsyncioTestCase):
    @patch("anthropic_proxy.gemini.stream_gemini_sdk_request")
    @patch("anthropic_proxy.gemini.gemini_auth")
    async def test_handle_gemini_request_stream(self, mock_auth, mock_stream):
        mock_auth.get_access_token = AsyncMock(return_value="access_token")
        mock_auth.get_project_id = MagicMock(return_value="project_id")
        gemini_chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}]},
                    "finishReason": "STOP",
                }
            ]
        }

        async def fake_stream(**kwargs):
            yield gemini_chunk

        mock_stream.side_effect = lambda **kwargs: fake_stream(**kwargs)

        claude_req = ClaudeMessagesRequest(
            model="gemini-flash",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
        )

        chunks = []
        async for chunk in handle_gemini_request(claude_req, "model-id"):
            chunks.append(chunk)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk["candidates"][0]["content"]["parts"][0]["text"], "Hello")
        self.assertEqual(chunk["candidates"][0]["finishReason"], "STOP")


if __name__ == "__main__":
    unittest.main()
