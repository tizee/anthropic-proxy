import json
import unittest

from google.genai import types as genai_types
from fastapi import HTTPException

from anthropic_proxy.gemini_streaming import convert_gemini_streaming_response_to_anthropic
from anthropic_proxy.signature_cache import clear_all_cache, get_tool_signature
from anthropic_proxy.types import ClaudeMessagesRequest

from tests.gemini.helpers import build_function_call_part, build_response_with_parts


def _extract_event_payloads(events: list[str], event_type: str) -> list[dict]:
    payloads: list[dict] = []
    for raw in events:
        event_name = None
        data = None
        for line in raw.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if event_name == event_type and data is not None:
            payloads.append(data)
    return payloads


class TestGeminiStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_emits_tool_use_from_sdk_types(self):
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        )

        part = build_function_call_part(name="get_weather", args={"city": "SF"})
        chunk = build_response_with_parts(parts=[part])

        async def gen():
            yield chunk

        events = [
            event
            async for event in convert_gemini_streaming_response_to_anthropic(
                gen(), request
            )
        ]
        block_starts = _extract_event_payloads(events, "content_block_start")
        tool_blocks = [b for b in block_starts if b["content_block"]["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["content_block"]["name"], "get_weather")

        deltas = _extract_event_payloads(events, "content_block_delta")
        input_deltas = [d for d in deltas if d["delta"]["type"] == "input_json_delta"]
        self.assertTrue(input_deltas)
        self.assertIn("city", input_deltas[0]["delta"]["partial_json"])

    async def test_streaming_tool_use_sets_stop_reason(self):
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )

        part = build_function_call_part(name="get_weather", args={"city": "SF"})
        chunk = build_response_with_parts(
            parts=[part],
            finish_reason=genai_types.FinishReason.UNEXPECTED_TOOL_CALL,
        )

        async def gen():
            yield chunk

        events = [
            event
            async for event in convert_gemini_streaming_response_to_anthropic(
                gen(), request
            )
        ]
        message_deltas = _extract_event_payloads(events, "message_delta")
        self.assertTrue(message_deltas)
        self.assertEqual(message_deltas[-1]["delta"]["stop_reason"], "tool_use")

    async def test_streaming_reads_usage_metadata(self):
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )

        chunk = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "hi"}]},
                    "finishReason": genai_types.FinishReason.STOP,
                }
            ],
            "usageMetadata": {"candidatesTokenCount": 7},
        }

        async def gen():
            yield chunk

        events = [
            event
            async for event in convert_gemini_streaming_response_to_anthropic(
                gen(), request
            )
        ]
        message_deltas = _extract_event_payloads(events, "message_delta")
        self.assertTrue(message_deltas)
        self.assertEqual(message_deltas[-1]["usage"]["output_tokens"], 7)

    async def test_streaming_maps_error_finish_reason_to_refusal(self):
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )

        chunk = build_response_with_parts(
            parts=[genai_types.Part(text="blocked")],
            finish_reason=genai_types.FinishReason.SAFETY,
        )

        async def gen():
            yield chunk

        events = [event async for event in convert_gemini_streaming_response_to_anthropic(gen(), request)]
        message_deltas = _extract_event_payloads(events, "message_delta")
        self.assertTrue(message_deltas)
        self.assertEqual(message_deltas[-1]["delta"]["stop_reason"], "refusal")

    async def test_streaming_caches_thought_signature_for_tool(self):
        clear_all_cache()
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            metadata={"session_id": "sess-1"},
            tools=[
                {
                    "name": "get_weather",
                    "description": "",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )

        part = genai_types.Part(
            functionCall=genai_types.FunctionCall(
                name="get_weather",
                args={"city": "SF"},
                id="tool-abc",
            ),
            thoughtSignature=b"sig-123",
        )
        chunk = build_response_with_parts(parts=[part])

        async def gen():
            yield chunk

        _ = [event async for event in convert_gemini_streaming_response_to_anthropic(gen(), request)]
        self.assertEqual(get_tool_signature("sess-1", "tool-abc"), "sig-123")
        clear_all_cache()

    async def test_streaming_emits_partial_args_on_finish(self):
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        )

        chunk1 = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "id": "tool-1",
                                    "partialArgs": [
                                        {
                                            "jsonPath": "$.location",
                                            "stringValue": "San ",
                                        }
                                    ],
                                    "willContinue": True,
                                }
                            }
                        ]
                    }
                }
            ]
        }
        chunk2 = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "id": "tool-1",
                                    "partialArgs": [
                                        {
                                            "jsonPath": "$.location",
                                            "stringValue": "Francisco",
                                        }
                                    ],
                                    "willContinue": False,
                                }
                            }
                        ]
                    },
                    "finishReason": genai_types.FinishReason.STOP,
                }
            ]
        }

        async def gen():
            yield chunk1
            yield chunk2

        events = [event async for event in convert_gemini_streaming_response_to_anthropic(gen(), request)]
        deltas = _extract_event_payloads(events, "content_block_delta")
        input_deltas = [d for d in deltas if d["delta"]["type"] == "input_json_delta"]
        self.assertTrue(input_deltas)
        self.assertIn("San Francisco", input_deltas[-1]["delta"]["partial_json"])

    async def test_streaming_emits_error_event_on_http_exception(self):
        request = ClaudeMessagesRequest(
            model="gemini-2.5-pro",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )

        async def gen():
            raise HTTPException(status_code=502, detail="Antigravity error: 403 Forbidden")
            yield  # pragma: no cover

        events = [event async for event in convert_gemini_streaming_response_to_anthropic(gen(), request)]
        error_events = _extract_event_payloads(events, "error")
        self.assertTrue(error_events)
        self.assertIn("Antigravity error", error_events[0]["error"]["message"])


if __name__ == "__main__":
    unittest.main()
