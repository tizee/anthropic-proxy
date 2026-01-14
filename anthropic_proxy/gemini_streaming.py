"""Gemini streaming response conversion to Anthropic SSE events."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from typing import Any, AsyncGenerator

from .types import ClaudeMessagesRequest, generate_unique_id

logger = logging.getLogger(__name__)


class GeminiStreamingConverter:
    """Convert Gemini streaming responses into Anthropic SSE events."""

    def __init__(self, original_request: ClaudeMessagesRequest):
        self.original_request = original_request
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.content_block_index = 0
        self.current_content_blocks: list[dict[str, Any]] = []
        self.text_block_started = False
        self.thinking_block_started = False
        self.tool_block_started = False
        self.pending_tool_calls: dict[str, deque[str]] = {}
        self.seen_tool_ids: set[str] = set()
        self.accumulated_text = ""
        self.accumulated_thinking = ""
        self.has_sent_stop_reason = False
        self.input_tokens = original_request.calculate_tokens()
        self.output_tokens = 0

    def _send_message_start_event(self) -> str:
        message_data = {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.original_request.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": 1,
                },
            },
        }
        return f"event: message_start\ndata: {json.dumps(message_data)}\n\n"

    def _send_ping_event(self) -> str:
        return f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    def _send_content_block_start_event(self, block_type: str, **kwargs) -> str:
        content_block = {"type": block_type, **kwargs}
        if block_type == "text":
            content_block["text"] = ""
        elif block_type == "tool_use":
            content_block.setdefault("id", generate_unique_id("tool"))
            content_block.setdefault("name", "")
            content_block.setdefault("input", {})
        event_data = {
            "type": "content_block_start",
            "index": self.content_block_index,
            "content_block": content_block,
        }
        return f"event: content_block_start\ndata: {json.dumps(event_data)}\n\n"

    def _send_content_block_delta_event(self, delta_type: str, content: str) -> str:
        delta = {"type": delta_type}
        if delta_type == "text_delta":
            delta["text"] = content
        elif delta_type == "input_json_delta":
            delta["partial_json"] = content
        elif delta_type == "thinking_delta":
            delta["thinking"] = content
        elif delta_type == "signature_delta":
            delta["signature"] = content
        event_data = {
            "type": "content_block_delta",
            "index": self.content_block_index,
            "delta": delta,
        }
        return f"event: content_block_delta\ndata: {json.dumps(event_data)}\n\n"

    def _send_content_block_stop_event(self) -> str:
        event_data = {"type": "content_block_stop", "index": self.content_block_index}
        return f"event: content_block_stop\ndata: {json.dumps(event_data)}\n\n"

    def _send_message_delta_event(self, stop_reason: str, output_tokens: int) -> str:
        event_data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
        return f"event: message_delta\ndata: {json.dumps(event_data)}\n\n"

    def _send_message_stop_event(self) -> str:
        return f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    def _send_done_event(self) -> str:
        return f"event: done\ndata: {json.dumps({'type': 'done'})}\n\n"

    def _open_text_block(self) -> str:
        self.text_block_started = True
        return self._send_content_block_start_event("text")

    def _open_thinking_block(self, signature: str | None) -> list[str]:
        self.thinking_block_started = True
        events = [self._send_content_block_start_event("thinking")]
        if signature:
            events.append(self._send_content_block_delta_event("signature_delta", signature))
        return events

    def _open_tool_block(self, tool_id: str, name: str) -> str:
        self.tool_block_started = True
        return self._send_content_block_start_event("tool_use", id=tool_id, name=name, input={})

    def _close_current_block(self) -> list[str]:
        if self.text_block_started or self.thinking_block_started or self.tool_block_started:
            events = [self._send_content_block_stop_event()]
            self.content_block_index += 1
            self.text_block_started = False
            self.thinking_block_started = False
            self.tool_block_started = False
            return events
        return []

    def _is_thinking_part(self, part: dict[str, Any]) -> bool:
        return part.get("thought") is True or part.get("thoughtSignature") is not None

    def _get_tool_id(self, tool_id: str | None) -> str:
        if tool_id and tool_id not in self.seen_tool_ids:
            self.seen_tool_ids.add(tool_id)
            return tool_id
        new_id = generate_unique_id("toolu")
        self.seen_tool_ids.add(new_id)
        return new_id

    async def process_chunk(self, gemini_chunk: dict[str, Any]) -> AsyncGenerator[str, None]:
        candidates = gemini_chunk.get("candidates", [])
        if not candidates:
            return

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        finish_reason = candidate.get("finishReason")

        for part in parts:
            if not isinstance(part, dict):
                continue

            if "functionCall" in part:
                call = part.get("functionCall", {})
                tool_name = call.get("name", "")
                tool_id = self._get_tool_id(call.get("id"))
                args = call.get("args", {})
                args_json = json.dumps(args, ensure_ascii=False)

                for event in self._close_current_block():
                    yield event

                yield self._open_tool_block(tool_id, tool_name)
                yield self._send_content_block_delta_event("input_json_delta", args_json)
                yield self._send_content_block_stop_event()
                self.content_block_index += 1
                self.tool_block_started = False
                continue

            if self._is_thinking_part(part):
                signature = part.get("thoughtSignature")
                text = part.get("text", "")
                if not self.thinking_block_started:
                    for event in self._close_current_block():
                        yield event
                    for event in self._open_thinking_block(signature):
                        yield event
                if text:
                    self.accumulated_thinking += text
                    yield self._send_content_block_delta_event("thinking_delta", text)
                continue

            text = part.get("text")
            if text:
                if not self.text_block_started:
                    for event in self._close_current_block():
                        yield event
                    yield self._open_text_block()
                self.accumulated_text += text
                yield self._send_content_block_delta_event("text_delta", text)

        if finish_reason:
            if finish_reason == "STOP":
                stop_reason = "end_turn"
            elif finish_reason == "MAX_TOKENS":
                stop_reason = "max_tokens"
            else:
                stop_reason = "tool_use" if any(
                    isinstance(p, dict) and p.get("functionCall")
                    for p in parts
                ) else "end_turn"

            for event in self._close_current_block():
                yield event

            output_tokens = gemini_chunk.get("usageMetadata", {}).get("candidatesTokenCount", 0)
            self.output_tokens = output_tokens or self.output_tokens
            yield self._send_message_delta_event(stop_reason, self.output_tokens)
            yield self._send_message_stop_event()
            yield self._send_done_event()
            self.has_sent_stop_reason = True


async def convert_gemini_streaming_response_to_anthropic(
    response_generator: AsyncGenerator[dict[str, Any], None],
    original_request: ClaudeMessagesRequest,
    model_id: str = "",
):
    converter = GeminiStreamingConverter(original_request)

    try:
        yield converter._send_message_start_event()
        yield converter._send_ping_event()

        async for chunk in response_generator:
            async for event in converter.process_chunk(chunk):
                yield event

        if not converter.has_sent_stop_reason:
            for event in converter._close_current_block():
                yield event
            yield converter._send_message_delta_event("end_turn", converter.output_tokens)
            yield converter._send_message_stop_event()
            yield converter._send_done_event()
    finally:
        logger.debug(
            "Gemini streaming completed for model %s (text=%d, thinking=%d)",
            model_id,
            len(converter.accumulated_text),
            len(converter.accumulated_thinking),
        )
