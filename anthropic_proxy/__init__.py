"""
Anthropic Proxy - A proxy server that translates between Anthropic API and OpenAI-compatible models.

This package provides format conversion between different AI model APIs.
"""

__version__ = "0.1.2"
__author__ = "Claude Code Proxy"

from .config import Config
from .types import ClaudeMessagesRequest, ClaudeMessagesResponse

__all__ = [
    "Config",
    "ClaudeMessagesRequest",
    "ClaudeMessagesResponse",
]
