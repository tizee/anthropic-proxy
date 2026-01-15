"""
Thinking Recovery Module for Antigravity Claude Models

When Claude's conversation history gets corrupted (thinking blocks stripped/malformed),
this module provides a "last resort" recovery by closing the current turn and starting fresh.

Philosophy: "Let it crash and start again" - Instead of trying to fix corrupted state,
we abandon the corrupted turn and let Claude generate fresh thinking.

Based on antigravity-auth plugin's thinking-recovery.ts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ConversationState:
    """Analysis result of conversation state."""

    in_tool_loop: bool = False
    turn_start_idx: int = -1
    turn_has_thinking: bool = False
    last_model_idx: int = -1
    last_model_has_thinking: bool = False
    last_model_has_tool_calls: bool = False


def _is_thinking_part(part: dict[str, Any]) -> bool:
    # Check for standard format: {"text": "...", "thought": true}
    if part.get("thought") is True:
        return True
    # Also check for nested format (should not exist, but handle for robustness): {"thinking": {...}}
    if isinstance(part.get("thinking"), dict):
        return True
    return False


def _is_function_response_part(part: dict[str, Any]) -> bool:
    return "functionResponse" in part and part["functionResponse"] is not None


def _is_function_call_part(part: dict[str, Any]) -> bool:
    return "functionCall" in part and part["functionCall"] is not None


def _is_tool_result_message(msg: dict[str, Any]) -> bool:
    if msg.get("role") != "user":
        return False
    parts = msg.get("parts")
    if not isinstance(parts, list):
        return False
    return any(_is_function_response_part(p) for p in parts if isinstance(p, dict))


def _message_has_thinking(msg: dict[str, Any]) -> bool:
    parts = msg.get("parts")
    if not isinstance(parts, list):
        return False
    return any(_is_thinking_part(p) for p in parts if isinstance(p, dict))


def _message_has_tool_calls(msg: dict[str, Any]) -> bool:
    parts = msg.get("parts")
    if not isinstance(parts, list):
        return False
    return any(_is_function_call_part(p) for p in parts if isinstance(p, dict))


def analyze_conversation_state(contents: list[dict[str, Any]]) -> ConversationState:
    """
    Analyzes conversation state to detect tool use loops and thinking mode issues.

    Key insight: A "turn" can span multiple assistant messages in a tool-use loop.
    We need to find the TURN START (first assistant message after last real user message)
    and check if THAT message had thinking, not just the last assistant message.

    Args:
        contents: List of Gemini content messages with role and parts

    Returns:
        ConversationState with analysis results
    """
    state = ConversationState()

    if not contents:
        return state

    # First pass: Find the last "real" user message (not a tool result)
    last_real_user_idx = -1
    for i, msg in enumerate(contents):
        if msg.get("role") == "user" and not _is_tool_result_message(msg):
            last_real_user_idx = i

    # Second pass: Analyze conversation and find turn boundaries
    for i, msg in enumerate(contents):
        if msg.get("role") == "model":
            has_thinking = _message_has_thinking(msg)
            has_tool_calls = _message_has_tool_calls(msg)

            # Track if this is the turn start
            if i > last_real_user_idx and state.turn_start_idx == -1:
                state.turn_start_idx = i
                state.turn_has_thinking = has_thinking

            state.last_model_idx = i
            state.last_model_has_tool_calls = has_tool_calls
            state.last_model_has_thinking = has_thinking

    # Determine if we're in a tool loop
    # We're in a tool loop if the conversation ends with a tool result
    if contents:
        last_msg = contents[-1]
        if last_msg.get("role") == "user" and _is_tool_result_message(last_msg):
            state.in_tool_loop = True

    return state


def _strip_all_thinking_blocks(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strips all thinking blocks from messages, including nested format."""
    result = []
    for content in contents:
        parts = content.get("parts")
        if not isinstance(parts, list):
            result.append(content)
            continue

        filtered_parts = [p for p in parts if isinstance(p, dict) and not _is_thinking_part(p)]

        if not filtered_parts:
            # All parts were filtered out, skip this content
            continue

        result.append({**content, "parts": filtered_parts})

    return result


def _count_trailing_tool_results(contents: list[dict[str, Any]]) -> int:
    """Counts tool results at the end of the conversation."""
    count = 0

    for i in range(len(contents) - 1, -1, -1):
        msg = contents[i]

        if msg.get("role") == "user":
            parts = msg.get("parts")
            if isinstance(parts, list):
                function_responses = [
                    p for p in parts
                    if isinstance(p, dict) and _is_function_response_part(p)
                ]

                if function_responses:
                    count += len(function_responses)
                else:
                    break  # Real user message, stop counting
        elif msg.get("role") == "model":
            break  # Stop at the model that made the tool calls

    return count


def close_tool_loop_for_thinking(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Closes an incomplete tool loop by injecting synthetic messages to start a new turn.

    This is the "let it crash and start again" recovery mechanism.

    When we detect:
    - We're in a tool loop (conversation ends with functionResponse)
    - The tool call was made WITHOUT thinking (thinking was stripped/corrupted)
    - We NOW want to enable thinking

    Instead of trying to fix the corrupted state, we:
    1. Strip ALL thinking blocks (removes any corrupted ones)
    2. Add synthetic MODEL message to complete the non-thinking turn
    3. Add synthetic USER message to start a NEW turn

    This allows Claude to generate fresh thinking for the new turn.

    Args:
        contents: List of Gemini content messages

    Returns:
        Updated contents list with synthetic messages appended
    """
    # Strip any old/corrupted thinking first
    stripped_contents = _strip_all_thinking_blocks(contents)

    # Count tool results from the end of the conversation
    tool_result_count = _count_trailing_tool_results(stripped_contents)

    # Build synthetic model message content based on tool count
    if tool_result_count == 0:
        synthetic_model_content = "[Processing previous context.]"
    elif tool_result_count == 1:
        synthetic_model_content = "[Tool execution completed.]"
    else:
        synthetic_model_content = f"[{tool_result_count} tool executions completed.]"

    # Step 1: Inject synthetic MODEL message to complete the non-thinking turn
    synthetic_model: dict[str, Any] = {
        "role": "model",
        "parts": [{"text": synthetic_model_content}],
    }

    # Step 2: Inject synthetic USER message to start a NEW turn
    synthetic_user: dict[str, Any] = {
        "role": "user",
        "parts": [{"text": "[Continue]"}],
    }

    return stripped_contents + [synthetic_model, synthetic_user]


def needs_thinking_recovery(state: ConversationState) -> bool:
    """
    Checks if conversation state requires tool loop closure for thinking recovery.

    Returns true if:
    - We're in a tool loop (state.in_tool_loop)
    - The turn didn't start with thinking (state.turn_has_thinking === False)

    This is the trigger for the "let it crash and start again" recovery.

    Args:
        state: Analyzed conversation state

    Returns:
        True if thinking recovery is needed
    """
    return state.in_tool_loop and not state.turn_has_thinking
