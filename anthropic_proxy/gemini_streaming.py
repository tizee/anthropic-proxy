"""Gemini streaming response conversion to Anthropic SSE events."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from typing import Any, AsyncGenerator

from .gemini_types import parse_gemini_response
from .signature_cache import cache_signature, cache_tool_signature
from .types import ClaudeMessagesRequest, generate_unique_id


def _extract_session_id(request: ClaudeMessagesRequest) -> str | None:
    metadata = request.metadata
    if not isinstance(metadata, dict):
        return None
    for key in ("session_id", "sessionId", "conversation_id", "conversationId", "thread_id", "threadId"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None

logger = logging.getLogger(__name__)


class GeminiStreamingConverter:
    """Convert Gemini streaming responses into Anthropic SSE events."""

    def __init__(self, original_request: ClaudeMessagesRequest, session_id: str | None):
        self.original_request = original_request
        self.session_id = session_id
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
        self.expected_tool_count = len(original_request.tools or [])
        self.last_thought_signature: str | None = None
        self.tool_code_buffer = ""
        self.in_tool_code = False
        self.pending_tool_signature_ids: deque[str] = deque()

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
        return (
            part.get("thought") is True
            or part.get("thoughtSignature") is not None
            or part.get("thought_signature") is not None
        )

    def _get_tool_id(self, tool_id: str | None) -> str:
        if tool_id and tool_id not in self.seen_tool_ids:
            self.seen_tool_ids.add(tool_id)
            return tool_id
        new_id = generate_unique_id("toolu")
        self.seen_tool_ids.add(new_id)
        return new_id

    def _remember_thought_signature(self, signature: str | None) -> None:
        if not signature:
            return
        self.last_thought_signature = signature
        while self.pending_tool_signature_ids:
            pending_id = self.pending_tool_signature_ids.popleft()
            cache_tool_signature(self.session_id, pending_id, signature)

    def _infer_tool_name(self, payload: dict[str, Any]) -> str:
        tool_name = payload.get("tool_name") or payload.get("name")
        if tool_name:
            return str(tool_name)

        if not self.original_request.tools:
            return ""

        keys = set(payload.keys())
        if "todos" in keys:
            return "TodoWrite"
        if "file_path" in keys and "content" in keys:
            return "Write"
        if "command" in keys:
            return "Bash"

        best_name = ""
        best_score = -1
        for tool in self.original_request.tools:
            schema = getattr(tool, "input_schema", None)
            if not isinstance(schema, dict):
                continue
            properties = schema.get("properties")
            if not isinstance(properties, dict):
                continue
            required = schema.get("required")
            required_set = set(required) if isinstance(required, list) else set()
            if required_set and not required_set.issubset(keys):
                continue
            matched = len(keys & set(properties.keys()))
            unknown = len(keys - set(properties.keys()))
            score = matched - unknown
            if score > best_score:
                best_score = score
                best_name = tool.name
        return best_name

    def _extract_tool_code_calls(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        combined = f"{self.tool_code_buffer}{text}"
        emitted: list[str] = []
        calls: list[dict[str, Any]] = []

        while combined:
            if not self.in_tool_code:
                start = combined.find("<tool_code>")
                if start == -1:
                    emitted.append(combined)
                    combined = ""
                    continue
                emitted.append(combined[:start])
                combined = combined[start + len("<tool_code>") :]
                self.in_tool_code = True
                self.tool_code_buffer = ""
                continue

            end = combined.find("</tool_code>")
            if end == -1:
                self.tool_code_buffer = combined
                combined = ""
                continue

            block = f"{self.tool_code_buffer}{combined[:end]}".strip()
            self.tool_code_buffer = ""
            self.in_tool_code = False
            combined = combined[end + len("</tool_code>") :]
            try:
                payload = json.loads(block)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                calls.append(payload)

        return "".join(emitted), calls

    async def process_chunk(self, gemini_chunk: dict[str, Any]) -> AsyncGenerator[str, None]:
        gemini_chunk = parse_gemini_response(gemini_chunk)
        candidates = gemini_chunk.get("candidates", [])
        if not candidates:
            return

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        finish_reason = candidate.get("finishReason")

        function_call_parts = [
            part
            for part in parts
            if isinstance(part, dict) and "functionCall" in part
        ]
        if self.expected_tool_count:
            logger.debug(
                "Gemini chunk parts=%d function_calls=%d finishReason=%s",
                len(parts),
                len(function_call_parts),
                finish_reason,
            )

        for part in parts:
            if not isinstance(part, dict):
                continue

            if "functionCall" in part:
                call = part.get("functionCall", {})
                tool_name = call.get("name", "")
                tool_id = self._get_tool_id(call.get("id"))
                args = call.get("args", {})
                signature = part.get("thoughtSignature")
                if signature:
                    self.last_thought_signature = signature
                if self.last_thought_signature:
                    cache_tool_signature(self.session_id, tool_id, self.last_thought_signature)
                if self.expected_tool_count:
                    logger.debug(
                        "Gemini functionCall name=%s id=%s args_keys=%s",
                        tool_name,
                        tool_id,
                        list(args.keys()) if isinstance(args, dict) else type(args).__name__,
                    )
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
                signature = part.get("thoughtSignature") or part.get("thought_signature")
                text = part.get("text", "")
                if signature:
                    self._remember_thought_signature(signature)
                if signature and text and self.session_id:
                    cache_signature(self.session_id, text, signature)
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
                remaining_text, tool_calls = self._extract_tool_code_calls(text)
                for call_payload in tool_calls:
                    tool_name = self._infer_tool_name(call_payload)
                    tool_id = self._get_tool_id(None)
                    if self.last_thought_signature:
                        cache_tool_signature(self.session_id, tool_id, self.last_thought_signature)
                    else:
                        self.pending_tool_signature_ids.append(tool_id)

                    for event in self._close_current_block():
                        yield event

                    yield self._open_tool_block(tool_id, tool_name)
                    args_json = json.dumps(call_payload, ensure_ascii=False)
                    yield self._send_content_block_delta_event("input_json_delta", args_json)
                    yield self._send_content_block_stop_event()
                    self.content_block_index += 1
                    self.tool_block_started = False

                if remaining_text:
                    if not self.text_block_started:
                        for event in self._close_current_block():
                            yield event
                        yield self._open_text_block()
                    self.accumulated_text += remaining_text
                    yield self._send_content_block_delta_event("text_delta", remaining_text)

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

            usage_metadata = gemini_chunk.get("usageMetadata") or {}
            output_tokens = usage_metadata.get("candidatesTokenCount", 0)
            self.output_tokens = output_tokens or self.output_tokens
            yield self._send_message_delta_event(stop_reason, self.output_tokens)
            yield self._send_message_stop_event()
            yield self._send_done_event()
            self.has_sent_stop_reason = True


async def convert_gemini_streaming_response_to_anthropic(
    response_generator: AsyncGenerator[dict[str, Any], None],
    original_request: ClaudeMessagesRequest,
    model_id: str = "",
    session_id: str | None = None,
):
    if session_id is None:
        session_id = _extract_session_id(original_request)
    converter = GeminiStreamingConverter(original_request, session_id)

    try:
        logger.debug(
            "Gemini streaming start model=%s expected_tools=%d",
            model_id,
            len(original_request.tools or []),
        )
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
