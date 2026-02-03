"""
Claude Code subscription support.
Handles authentication and request proxying for Claude Code (Pro/Max) subscriptions.

Users provide a setup-token obtained via `claude setup-token` command.
This module handles the required header impersonation, system prompt injection,
and automatic prompt caching.
"""

import getpass
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import HTTPException

from .config_manager import load_auth_file, save_auth_file
from .types import ClaudeMessagesRequest
from .utils import sanitize_anthropic_messages

logger = logging.getLogger(__name__)

# Constants
CLAUDE_CODE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_CODE_VERSION = "2.1.2"  # Claude CLI version to impersonate

# Required system prompt prefix for OAuth tokens
CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

# Cache control configuration
# Set CLAUDE_CODE_CACHE_RETENTION=long for 1-hour TTL, otherwise default 5 minutes
CACHE_RETENTION_ENV = "CLAUDE_CODE_CACHE_RETENTION"

# Beta headers for different features
BETA_FEATURES_BASE = "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14"
BETA_THINKING_FEATURE = "interleaved-thinking-2025-05-14"

# Thinking budget mapping (matches Claude Code /thinking command)
THINKING_BUDGET_MAP = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}

# Models that support thinking/reasoning (Claude 4.5 only)
THINKING_CAPABLE_MODELS = {
    "claude-opus-4-5",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
}

# Minimum output space required beyond thinking budget
MIN_OUTPUT_BUFFER = 1024

# Default models available via Claude Code subscription (Claude 4.5 only).
# max_tokens matches Claude Code behavior (64K for all 4.5 models)
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


def is_thinking_capable_model(model_name: str) -> bool:
    """Check if a model supports thinking/reasoning mode."""
    # Strip prefix if present
    if model_name.startswith("claude-code/"):
        model_name = model_name[len("claude-code/"):]
    return model_name in THINKING_CAPABLE_MODELS


def get_thinking_budget(thinking_config: Any) -> int | None:
    """
    Extract thinking budget from thinking configuration.

    Supports:
    - { type: "enabled", budget_tokens: N }
    - { type: "enabled" } -> uses medium (8192) as default
    - String level: "minimal", "low", "medium", "high"
    """
    if thinking_config is None:
        return None

    if isinstance(thinking_config, dict):
        config_type = thinking_config.get("type")
        if config_type == "disabled":
            return None
        if config_type == "enabled":
            budget = thinking_config.get("budget_tokens")
            if budget is not None:
                return int(budget)
            # Default to medium if no budget specified
            return THINKING_BUDGET_MAP["medium"]

    # Handle object with .type attribute (Pydantic model)
    if hasattr(thinking_config, "type"):
        if thinking_config.type == "disabled":
            return None
        if thinking_config.type == "enabled":
            budget = getattr(thinking_config, "budget_tokens", None)
            if budget is not None:
                return int(budget)
            return THINKING_BUDGET_MAP["medium"]

    # Handle string level (minimal, low, medium, high)
    if isinstance(thinking_config, str):
        level = thinking_config.lower()
        if level in THINKING_BUDGET_MAP:
            return THINKING_BUDGET_MAP[level]

    return None


def build_claude_code_headers(token: str, thinking_enabled: bool = False) -> dict[str, str]:
    """
    Build headers required for Claude Code OAuth token requests.

    Must impersonate Claude CLI to avoid token rejection.

    Args:
        token: The OAuth/setup token
        thinking_enabled: Whether thinking mode is enabled (adds beta header)
    """
    beta_features = BETA_FEATURES_BASE
    if thinking_enabled:
        beta_features = f"{beta_features},{BETA_THINKING_FEATURE}"

    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "anthropic-version": "2023-06-01",
        "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
        "x-app": "cli",
        "anthropic-dangerous-direct-browser-access": "true",
        "anthropic-beta": beta_features,
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


def get_model_max_tokens(model_name: str) -> int:
    """Get the max_tokens limit for a model."""
    # Strip prefix if present
    if model_name.startswith("claude-code/"):
        model_name = model_name[len("claude-code/"):]

    model_config = DEFAULT_CLAUDE_CODE_MODELS.get(model_name)
    if model_config:
        return model_config.get("max_tokens", 64000)

    # Default to 64K for unknown Claude 4.5 models
    return 64000


def process_thinking_config(
    request_data: dict[str, Any],
    model_name: str,
) -> tuple[dict[str, Any], bool]:
    """
    Process thinking configuration in the request.

    - Validates model supports thinking
    - Extracts/normalizes thinking budget
    - Ensures max_tokens > thinking_budget + 1024
    - Caps max_tokens at model's limit (64K for 4.5 models)

    Args:
        request_data: The request data dict
        model_name: The target model name

    Returns:
        Tuple of (modified request_data, thinking_enabled)
    """
    model_max = get_model_max_tokens(model_name)

    # Cap user's max_tokens at model limit
    user_max_tokens = request_data.get("max_tokens")
    if user_max_tokens is not None and user_max_tokens > model_max:
        logger.info(f"Capping max_tokens from {user_max_tokens} to {model_max}")
        request_data["max_tokens"] = model_max

    thinking_config = request_data.get("thinking")
    thinking_budget = get_thinking_budget(thinking_config)

    if thinking_budget is None:
        # No thinking requested, remove thinking field if present
        request_data.pop("thinking", None)
        return request_data, False

    # Check if model supports thinking
    if not is_thinking_capable_model(model_name):
        logger.warning(
            f"Model {model_name} does not support thinking mode. Disabling thinking."
        )
        request_data.pop("thinking", None)
        return request_data, False

    # Normalize thinking configuration to Anthropic API format
    # API expects: { "type": "enabled", "budget_tokens": N }
    # budget_tokens is REQUIRED when type is "enabled"
    request_data["thinking"] = {
        "type": "enabled",
        "budget_tokens": thinking_budget,
    }

    logger.info(f"Thinking enabled: budget_tokens={thinking_budget}")

    # Ensure max_tokens > thinking_budget + 1024
    max_tokens = request_data.get("max_tokens") or model_max
    min_required = thinking_budget + MIN_OUTPUT_BUFFER

    if max_tokens < min_required:
        # Adjust to minimum required, but don't exceed model limit
        adjusted = min(min_required, model_max)
        logger.info(
            f"Adjusting max_tokens from {max_tokens} to {adjusted} "
            f"(thinking_budget={thinking_budget} + buffer={MIN_OUTPUT_BUFFER})"
        )
        request_data["max_tokens"] = adjusted

    return request_data, True


class ClaudeCodeErrorResponse:
    """Represents an error response from Claude Code API with HTTP status preserved."""

    def __init__(self, status_code: int, error_body: dict[str, Any]):
        self.status_code = status_code
        self.error_body = error_body

    def to_json_response(self):
        """Convert to FastAPI JSONResponse with correct status code."""
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=self.status_code,
            content=self.error_body,
        )


async def handle_claude_code_request(
    request: ClaudeMessagesRequest,
    model_id: str,
    *,
    model_name: str | None = None,
) -> AsyncGenerator[str, None] | ClaudeCodeErrorResponse:
    """
    Handle a request to Claude Code subscription.

    Args:
        request: The Claude Messages API request
        model_id: The model ID from the request
        model_name: Optional override for the actual model name

    Returns:
        Either a ClaudeCodeErrorResponse (for non-200 responses) or
        an async generator yielding SSE event strings in Anthropic format.

    Note:
        Caller MUST check if return value is ClaudeCodeErrorResponse and handle accordingly.
    """
    token = claude_code_auth.get_token()
    if not token:
        raise HTTPException(
            status_code=401,
            detail="No Claude Code token found. Run: anthropic-proxy login --claude-code",
        )

    # Prepare request body
    request_data = request.model_dump(exclude_none=True)

    # Sanitize messages to fix orphaned tool_use and empty content issues
    # This handles edge cases from context compaction or interrupted streams
    if "messages" in request_data:
        request_data["messages"] = sanitize_anthropic_messages(request_data["messages"])

    # Use the actual model name for the API call
    target_model = model_name or model_id
    # Strip provider prefix if present
    if target_model.startswith("claude-code/"):
        target_model = target_model[len("claude-code/"):]
    request_data["model"] = target_model

    # Process thinking configuration (validate, normalize, adjust max_tokens)
    request_data, thinking_enabled = process_thinking_config(request_data, target_model)

    # Build headers with CLI impersonation (include thinking beta if enabled)
    headers = build_claude_code_headers(token, thinking_enabled=thinking_enabled)

    # Inject required system prompt with cache control
    request_data = inject_system_prompt_with_cache(request_data)

    # Add cache control to last user message for conversation caching
    request_data = inject_message_cache_control(request_data)

    # Force streaming for consistent handling
    request_data["stream"] = True

    cache_ttl = get_cache_ttl() or "5m (default)"
    thinking_info = f", thinking_budget={request_data['thinking']['budget_tokens']}" if thinking_enabled else ""
    logger.info(f"Claude Code request: model={target_model}, cache_ttl={cache_ttl}{thinking_info}")

    # Debug: log the thinking config being sent
    if thinking_enabled:
        logger.info(f"Thinking config in request: {request_data.get('thinking')}")

    # Make the request and check status before returning
    # This allows us to return proper HTTP status codes for errors
    try:
        client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        response = await client.send(
            client.build_request(
                "POST",
                CLAUDE_CODE_API_URL,
                json=request_data,
                headers=headers,
            ),
            stream=True,
        )

        if response.status_code != 200:
            # Read error body and close response
            error_text = await response.aread()
            error_str = error_text.decode("utf-8", errors="replace")
            await response.aclose()
            await client.aclose()

            logger.error(f"Claude Code API error {response.status_code}: {error_str}")

            # Try to parse upstream error (already in Anthropic format)
            try:
                error_data = json.loads(error_str)
                if error_data.get("type") == "error":
                    return ClaudeCodeErrorResponse(response.status_code, error_data)
            except json.JSONDecodeError:
                pass

            # Construct Anthropic-format error for non-JSON responses
            error_type_map = {
                400: "invalid_request_error",
                401: "authentication_error",
                403: "permission_error",
                404: "not_found_error",
                429: "rate_limit_error",
                500: "api_error",
                502: "api_error",
                503: "overloaded_error",
                529: "overloaded_error",
            }
            error_type = error_type_map.get(response.status_code, "api_error")
            error_body = {
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": error_str or f"HTTP {response.status_code}",
                },
            }
            return ClaudeCodeErrorResponse(response.status_code, error_body)

        # Success - return streaming generator
        return _stream_claude_code_response(client, response)

    except httpx.ConnectError as conn_err:
        logger.error(f"Claude Code connection error: {conn_err}")
        error_body = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"Connection to Claude API failed: {conn_err}",
            },
        }
        return ClaudeCodeErrorResponse(502, error_body)

    except httpx.TimeoutException as timeout_err:
        logger.error(f"Claude Code timeout error: {timeout_err}")
        error_body = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"Request to Claude API timed out: {timeout_err}",
            },
        }
        return ClaudeCodeErrorResponse(504, error_body)

    except httpx.RequestError as e:
        logger.error(f"Claude Code network error: {e}")
        error_body = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"Network error: {e}",
            },
        }
        return ClaudeCodeErrorResponse(502, error_body)

    except Exception as e:
        logger.error(f"Claude Code unexpected error: {e}")
        error_body = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": f"Unexpected error: {e}",
            },
        }
        return ClaudeCodeErrorResponse(500, error_body)


async def _stream_claude_code_response(
    client: httpx.AsyncClient,
    response: httpx.Response,
) -> AsyncGenerator[str, None]:
    """
    Stream SSE events from a successful Claude Code response.

    Args:
        client: The httpx client (will be closed when done)
        response: The httpx response to stream from

    Yields:
        SSE event strings in Anthropic format

    Raises:
        MidStreamAbort: On read errors during streaming to simulate real API
            connection drops that trigger client retry logic.
    """
    from .midstream_abort import MidStreamAbort

    try:
        async for chunk in response.aiter_text():
            if chunk:
                yield chunk
    except httpx.ReadError as e:
        # Error during streaming - abort connection to trigger client retry
        logger.error(f"Claude Code streaming read error: {e}")
        raise MidStreamAbort(f"upstream read error: {e}") from e
    except httpx.ConnectError as e:
        # Connection dropped during streaming
        logger.error(f"Claude Code streaming connection error: {e}")
        raise MidStreamAbort(f"upstream connection error: {e}") from e
    except Exception as e:
        logger.error(f"Claude Code unexpected streaming error: {e}")
        raise MidStreamAbort(f"upstream error: {e}") from e
    finally:
        await response.aclose()
        await client.aclose()
