#!/usr/bin/env python3
"""
Test adaptive thinking configuration for Claude 4.6 Opus.

Validates that ClaudeMessagesRequest accepts thinking.type='adaptive'.
"""

import unittest

from anthropic_proxy.types import (
    ClaudeMessagesRequest,
    ClaudeThinkingConfigAdaptive,
    ClaudeThinkingConfigDisabled,
    ClaudeThinkingConfigEnabled,
    ClaudeTokenCountRequest,
)


class TestAdaptiveThinking(unittest.TestCase):
    """Test adaptive thinking configuration support."""

    def test_adaptive_thinking_config_creation(self):
        """Test creating adaptive thinking config directly."""
        config = ClaudeThinkingConfigAdaptive(type="adaptive")
        self.assertEqual(config.type, "adaptive")

    def test_adaptive_thinking_in_message_request(self):
        """Test ClaudeMessagesRequest accepts adaptive thinking type."""
        request = ClaudeMessagesRequest(
            model="claude-opus-4-6",
            max_tokens=1000,
            messages=[],
            thinking={"type": "adaptive"},
        )
        self.assertIsInstance(request.thinking, ClaudeThinkingConfigAdaptive)
        self.assertEqual(request.thinking.type, "adaptive")

    def test_enabled_thinking_still_works(self):
        """Test enabled thinking still works."""
        request = ClaudeMessagesRequest(
            model="claude-opus-4-5",
            max_tokens=1000,
            messages=[],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
        self.assertIsInstance(request.thinking, ClaudeThinkingConfigEnabled)
        self.assertEqual(request.thinking.type, "enabled")
        self.assertEqual(request.thinking.budget_tokens, 1024)

    def test_disabled_thinking_still_works(self):
        """Test disabled thinking still works."""
        request = ClaudeMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            messages=[],
            thinking={"type": "disabled"},
        )
        self.assertIsInstance(request.thinking, ClaudeThinkingConfigDisabled)
        self.assertEqual(request.thinking.type, "disabled")

    def test_token_count_request_adaptive_thinking(self):
        """Test ClaudeTokenCountRequest accepts adaptive thinking."""
        request = ClaudeTokenCountRequest(
            model="claude-opus-4-6",
            messages=[],
            thinking={"type": "adaptive"},
        )
        self.assertIsInstance(request.thinking, ClaudeThinkingConfigAdaptive)
        self.assertEqual(request.thinking.type, "adaptive")


if __name__ == "__main__":
    unittest.main()
