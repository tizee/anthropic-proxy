"""
Anthropic Proxy - A proxy server that translates between Anthropic API and OpenAI-compatible models.

This package provides intelligent routing and format conversion between different AI model APIs.
"""

__version__ = "0.1.2"
__author__ = "Claude Code Proxy"

# Export main components for easier imports
from .config import Config
from .server import app
from .types import ClaudeMessagesRequest, ClaudeMessagesResponse

__all__ = [
    "Config",
    "app",
    "ClaudeMessagesRequest",
    "ClaudeMessagesResponse",
]
