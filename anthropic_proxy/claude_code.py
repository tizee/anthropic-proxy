"""
Claude Code subscription support.
Handles authentication and request proxying for Claude Code (Pro/Max) subscriptions.

Users provide a setup-token obtained via `claude setup-token` command.
This module handles the required header impersonation, system prompt injection,
and automatic prompt caching.
"""

import getpass
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from .config_manager import load_auth_file, save_auth_file
from .types import ClaudeMessagesRequest

logger = logging.getLogger(__name__)

# Constants
CLAUDE_CODE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_CODE_VERSION = "2.1.2"  # Claude CLI version to impersonate

# Required system prompt prefix for OAuth tokens
CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

# Cache control configuration
# Set CLAUDE_CODE_CACHE_RETENTION=long for 1-hour TTL, otherwise default 5 minutes
CACHE_RETENTION_ENV = "CLAUDE_CODE_CACHE_RETENTION"

# Default models available via Claude Code subscription (Claude 4.5 only).
DEFAULT_CLAUDE_CODE_MODELS = {
    # Claude 4.5 Opus
    "claude-opus-4-5": {
        "model_name": "claude-opus-4-5",
        "description": "Claude Opus 4.5 (latest)",
        "max_tokens": 64000,
    },
    "claude-opus-4-5-20251101": {
        "model_name": "claude-opus-4-5-20251101",
        "description": "Claude Opus 4.5",
        "max_tokens": 64000,
    },
    # Claude 4.5 Sonnet
    "claude-sonnet-4-5": {
        "model_name": "claude-sonnet-4-5",
        "description": "Claude Sonnet 4.5 (latest)",
        "max_tokens": 64000,
    },
    "claude-sonnet-4-5-20250929": {
        "model_name": "claude-sonnet-4-5-20250929",
        "description": "Claude Sonnet 4.5",
        "max_tokens": 64000,
    },
    # Claude 4.5 Haiku
    "claude-haiku-4-5": {
        "model_name": "claude-haiku-4-5",
        "description": "Claude Haiku 4.5 (latest)",
        "max_tokens": 64000,
    },
    "claude-haiku-4-5-20251001": {
        "model_name": "claude-haiku-4-5-20251001",
        "description": "Claude Haiku 4.5",
        "max_tokens": 64000,
    },
}


def is_oauth_token(token: str) -> bool:
    """Check if a token is an OAuth/setup token (vs regular API key)."""
    return "sk-ant-oat" in token


def get_cache_ttl() -> str | None:
    """
    Get cache TTL based on environment variable.

    Returns '1h' for long retention mode, None for default (5 minutes).
    Set CLAUDE_CODE_CACHE_RETENTION=long to enable 1-hour TTL.
    """
    if os.environ.get(CACHE_RETENTION_ENV) == "long":
        return "1h"
    return None


def build_cache_control() -> dict[str, Any]:
    """Build cache_control object with optional TTL."""
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    ttl = get_cache_ttl()
    if ttl:
        cache_control["ttl"] = ttl
    return cache_control


class ClaudeCodeAuth:
    """
    Manages Claude Code authentication tokens.

    Unlike other providers, Claude Code uses a simple token-based auth:
    - User runs `claude setup-token` to get a permanent token
    - User provides this token via CLI
    - Token is stored in auth.json under 'claude-code' key
    """

    PROVIDER_KEY = "claude-code"

    def __init__(self):
        self._auth_data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load auth data from auth.json."""
        full_data = load_auth_file()
        self._auth_data = full_data.get(self.PROVIDER_KEY, {})

    def _save(self) -> None:
        """Save auth data to auth.json."""
        full_data = load_auth_file()
        full_data[self.PROVIDER_KEY] = self._auth_data
        save_auth_file(full_data)

    def has_auth(self) -> bool:
        """Check if we have a stored token."""
        return bool(self._auth_data.get("token"))

    def get_token(self) -> str | None:
        """Get the stored setup token."""
        self._load()
        return self._auth_data.get("token")

    def save_token(self, token: str) -> None:
        """
        Save a setup token.

        Args:
            token: The setup token from `claude setup-token`
        """
        if not token.startswith("sk-ant-"):
            raise ValueError("Invalid token format. Token should start with 'sk-ant-'")

        self._auth_data["token"] = token
        self._save()
        logger.info("Claude Code token saved successfully")

    def clear_token(self) -> None:
        """Remove the stored token."""
        self._auth_data = {}
        self._save()
        logger.info("Claude Code token cleared")

    def login(self) -> None:
        """
        Interactive login flow for Claude Code.

        Prompts user to paste their setup-token (hidden input).
        """
        print("\nClaude Code Authentication")
        print("=" * 40)
        print("\nTo authenticate with Claude Code subscription:")
        print("1. Run: claude setup-token")
        print("2. Copy the generated token (starts with sk-ant-)")
        print("")

        try:
            token = getpass.getpass("Paste your Claude setup-token (hidden): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nLogin cancelled.")
            return

        if not token:
            print("No token provided. Aborting.")
            return

        try:
            self.save_token(token)
            print("\nToken saved successfully!")
            print("Note: This token is permanent until revoked in your Anthropic account.")
        except ValueError as e:
            print(f"\nError: {e}")


# Global auth instance
claude_code_auth = ClaudeCodeAuth()


def build_claude_code_headers(token: str) -> dict[str, str]:
    """
    Build headers required for Claude Code OAuth token requests.

    Must impersonate Claude CLI to avoid token rejection.
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "anthropic-version": "2023-06-01",
        "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
        "x-app": "cli",
        "anthropic-dangerous-direct-browser-access": "true",
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14",
        "accept": "application/json",
    }


def inject_system_prompt_with_cache(request_data: dict[str, Any]) -> dict[str, Any]:
    """
    Inject the required Claude Code system prompt prefix with cache control.

    OAuth tokens require the system prompt to start with
    "You are Claude Code, Anthropic's official CLI for Claude."

    All system prompt blocks get cache_control for prompt caching.
    """
    existing_system = request_data.get("system")
    cache_control = build_cache_control()

    # Build system prompt array with required prefix and cache control
    system_blocks = [
        {
            "type": "text",
            "text": CLAUDE_CODE_SYSTEM_PREFIX,
            "cache_control": cache_control,
        }
    ]

    if existing_system:
        if isinstance(existing_system, str):
            # String format: append as second block with cache
            system_blocks.append({
                "type": "text",
                "text": existing_system,
                "cache_control": cache_control,
            })
        elif isinstance(existing_system, list):
            # Array format: append all existing blocks with cache
            for block in existing_system:
                if isinstance(block, dict):
                    # Add cache_control to existing block
                    block_with_cache = {**block, "cache_control": cache_control}
                    system_blocks.append(block_with_cache)
                else:
                    system_blocks.append(block)

    request_data["system"] = system_blocks
    return request_data


def inject_message_cache_control(request_data: dict[str, Any]) -> dict[str, Any]:
    """
    Add cache_control to the last user message's last content block.

    This enables conversation history caching - only the last user message
    needs cache_control to cache the entire conversation context.
    """
    messages = request_data.get("messages", [])
    if not messages:
        return request_data

    # Find the last user message
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return request_data

    last_message = messages[last_user_idx]
    content = last_message.get("content")

    if content is None:
        return request_data

    cache_control = build_cache_control()

    # Handle string content - convert to array format
    if isinstance(content, str):
        messages[last_user_idx] = {
            **last_message,
            "content": [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": cache_control,
                }
            ],
        }
    elif isinstance(content, list) and content:
        # Add cache_control to the last content block
        last_block = content[-1]
        if isinstance(last_block, dict):
            block_type = last_block.get("type", "")
            # Only add cache to supported types: text, image, tool_result
            if block_type in ("text", "image", "tool_result"):
                content[-1] = {**last_block, "cache_control": cache_control}
                messages[last_user_idx] = {**last_message, "content": content}

    request_data["messages"] = messages
    return request_data


# Keep old function name for backward compatibility
def inject_system_prompt(request_data: dict[str, Any]) -> dict[str, Any]:
    """Alias for inject_system_prompt_with_cache."""
    return inject_system_prompt_with_cache(request_data)


async def handle_claude_code_request(
    request: ClaudeMessagesRequest,
    model_id: str,
    *,
    model_name: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Handle a request to Claude Code subscription.

    Args:
        request: The Claude Messages API request
        model_id: The model ID from the request
        model_name: Optional override for the actual model name

    Yields:
        SSE event strings in Anthropic format
    """
    token = claude_code_auth.get_token()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="No Claude Code token found. Run: anthropic-proxy login --claude-code",
        )

    # Build headers with CLI impersonation
    headers = build_claude_code_headers(token)

    # Prepare request body
    request_data = request.model_dump(exclude_none=True)

    # Use the actual model name for the API call
    target_model = model_name or model_id
    # Strip provider prefix if present
    if target_model.startswith("claude-code/"):
        target_model = target_model[len("claude-code/"):]
    request_data["model"] = target_model

    # Inject required system prompt with cache control
    request_data = inject_system_prompt_with_cache(request_data)

    # Add cache control to last user message for conversation caching
    request_data = inject_message_cache_control(request_data)

    # Force streaming for consistent handling
    request_data["stream"] = True

    cache_ttl = get_cache_ttl() or "5m (default)"
    logger.info(f"Claude Code request: model={target_model}, cache_ttl={cache_ttl}")
    logger.debug(f"Claude Code headers: {list(headers.keys())}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        try:
            async with client.stream(
                "POST",
                CLAUDE_CODE_API_URL,
                json=request_data,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_str = error_text.decode("utf-8", errors="replace")
                    logger.error(
                        f"Claude Code API error {response.status_code}: {error_str}"
                    )

                    # Check for auth errors
                    if response.status_code == 401:
                        raise HTTPException(
                            status_code=401,
                            detail="Claude Code token is invalid or revoked. Run: anthropic-proxy login --claude-code",
                        )
                    elif response.status_code == 403:
                        raise HTTPException(
                            status_code=403,
                            detail="Claude Code access denied. Check your subscription status.",
                        )

                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Claude Code API error: {error_str}",
                    )

                # Stream SSE events directly (already in Anthropic format)
                async for line in response.aiter_lines():
                    if line:
                        yield line + "\n"

        except httpx.RequestError as e:
            logger.error(f"Claude Code network error: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"Claude Code network error: {e}",
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Claude Code unexpected error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Claude Code unexpected error: {e}",
            ) from e
