"""
Tests for error passthrough in streaming responses.

Ensures upstream API errors are forwarded to downstream clients with
correct HTTP status codes and error bodies in Anthropic format,
allowing proper error handling and retry logic at the client level.
"""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from anthropic_proxy.claude_code import ClaudeCodeErrorResponse, handle_claude_code_request
from anthropic_proxy.types import ClaudeMessagesRequest


# Helper to collect async generator results
async def collect_async_generator(gen):
    """Collect all items from an async generator."""
    results = []
    async for item in gen:
        results.append(item)
    return results


def create_mock_response(status_code: int, body: bytes):
    """Create a mock httpx response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code

    async def aread():
        return body

    mock_response.aread = aread
    mock_response.aclose = AsyncMock()
    return mock_response


class TestClaudeCodeErrorPassthrough:
    """Tests for Claude Code error passthrough with correct HTTP status codes."""

    @pytest.fixture
    def basic_request(self):
        """Create a minimal Claude Messages request."""
        return ClaudeMessagesRequest(
            model="claude-code/claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

    @pytest.fixture
    def mock_auth(self):
        """Mock Claude Code authentication."""
        with patch("anthropic_proxy.claude_code.claude_code_auth") as mock:
            mock.get_token.return_value = "sk-ant-oat01-test-token"
            yield mock

    @pytest.mark.asyncio
    async def test_400_error_passthrough(self, basic_request, mock_auth):
        """400 invalid_request_error should be returned with correct HTTP status."""
        upstream_error = {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "messages.130.content.0: unexpected `tool_use_id`"
            }
        }
        upstream_error_str = json.dumps(upstream_error)
        mock_response = create_mock_response(400, upstream_error_str.encode())

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        # Should return ClaudeCodeErrorResponse with correct status
        assert isinstance(result, ClaudeCodeErrorResponse)
        assert result.status_code == 400
        assert result.error_body["type"] == "error"
        assert result.error_body["error"]["type"] == "invalid_request_error"
        assert "tool_use_id" in result.error_body["error"]["message"]

    @pytest.mark.asyncio
    async def test_429_rate_limit_passthrough(self, basic_request, mock_auth):
        """429 rate_limit_error should be returned with correct HTTP status."""
        upstream_error = {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded"
            }
        }
        upstream_error_str = json.dumps(upstream_error)
        mock_response = create_mock_response(429, upstream_error_str.encode())

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        assert isinstance(result, ClaudeCodeErrorResponse)
        assert result.status_code == 429
        assert result.error_body["error"]["type"] == "rate_limit_error"

    @pytest.mark.asyncio
    async def test_500_api_error_passthrough(self, basic_request, mock_auth):
        """500 server error should be returned with correct HTTP status."""
        upstream_error = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Internal server error"
            }
        }
        upstream_error_str = json.dumps(upstream_error)
        mock_response = create_mock_response(500, upstream_error_str.encode())

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        assert isinstance(result, ClaudeCodeErrorResponse)
        assert result.status_code == 500
        assert result.error_body["error"]["type"] == "api_error"

    @pytest.mark.asyncio
    async def test_529_overloaded_passthrough(self, basic_request, mock_auth):
        """529 overloaded_error should be returned with correct HTTP status."""
        upstream_error = {
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "API is overloaded"
            }
        }
        upstream_error_str = json.dumps(upstream_error)
        mock_response = create_mock_response(529, upstream_error_str.encode())

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        assert isinstance(result, ClaudeCodeErrorResponse)
        assert result.status_code == 529
        assert result.error_body["error"]["type"] == "overloaded_error"

    @pytest.mark.asyncio
    async def test_non_json_error_converts_to_anthropic_format(self, basic_request, mock_auth):
        """Non-JSON error response should be converted to Anthropic format."""
        plain_error = "Bad Gateway: upstream server unavailable"
        mock_response = create_mock_response(502, plain_error.encode())

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        assert isinstance(result, ClaudeCodeErrorResponse)
        assert result.status_code == 502
        assert result.error_body["type"] == "error"
        assert result.error_body["error"]["type"] == "api_error"
        assert "Bad Gateway" in result.error_body["error"]["message"]

    @pytest.mark.asyncio
    async def test_connection_error_returns_502(self, basic_request, mock_auth):
        """Connection errors should return 502 with error body."""
        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        assert isinstance(result, ClaudeCodeErrorResponse)
        assert result.status_code == 502
        assert result.error_body["type"] == "error"
        assert result.error_body["error"]["type"] == "api_error"
        assert "Connection" in result.error_body["error"]["message"]

    @pytest.mark.asyncio
    async def test_timeout_error_returns_504(self, basic_request, mock_auth):
        """Timeout errors should return 504 with error body."""
        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(side_effect=httpx.TimeoutException("Read timeout"))

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        assert isinstance(result, ClaudeCodeErrorResponse)
        assert result.status_code == 504
        assert result.error_body["type"] == "error"
        assert result.error_body["error"]["type"] == "api_error"
        assert "timeout" in result.error_body["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_network_error_returns_502(self, basic_request, mock_auth):
        """Network errors should return 502 with error body."""
        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(side_effect=httpx.RequestError("Network unreachable"))

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        assert isinstance(result, ClaudeCodeErrorResponse)
        assert result.status_code == 502
        assert result.error_body["type"] == "error"
        assert result.error_body["error"]["type"] == "api_error"


class TestErrorTypeMapping:
    """Tests for HTTP status code to Anthropic error type mapping."""

    @pytest.fixture
    def basic_request(self):
        return ClaudeMessagesRequest(
            model="claude-code/claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

    @pytest.fixture
    def mock_auth(self):
        with patch("anthropic_proxy.claude_code.claude_code_auth") as mock:
            mock.get_token.return_value = "sk-ant-oat01-test-token"
            yield mock

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code,expected_type", [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (502, "api_error"),
        (503, "overloaded_error"),
        (529, "overloaded_error"),
    ])
    async def test_status_code_to_error_type(
        self, basic_request, mock_auth, status_code, expected_type
    ):
        """Test HTTP status codes map to correct Anthropic error types."""
        # Non-JSON response to force type mapping
        plain_error = f"HTTP {status_code} error"
        mock_response = create_mock_response(status_code, plain_error.encode())

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        assert isinstance(result, ClaudeCodeErrorResponse)
        assert result.status_code == status_code
        assert result.error_body["error"]["type"] == expected_type


class TestErrorResponseFormat:
    """Tests for error response format compliance."""

    @pytest.fixture
    def basic_request(self):
        return ClaudeMessagesRequest(
            model="claude-code/claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )

    @pytest.fixture
    def mock_auth(self):
        with patch("anthropic_proxy.claude_code.claude_code_auth") as mock:
            mock.get_token.return_value = "sk-ant-oat01-test-token"
            yield mock

    @pytest.mark.asyncio
    async def test_error_response_format(self, basic_request, mock_auth):
        """Error responses should be in Anthropic format with correct HTTP status."""
        upstream_error = {
            "type": "error",
            "error": {"type": "api_error", "message": "Test error"}
        }
        mock_response = create_mock_response(500, json.dumps(upstream_error).encode())

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        # Verify it's an error response
        assert isinstance(result, ClaudeCodeErrorResponse)
        assert result.status_code == 500

        # Verify error body format
        assert result.error_body["type"] == "error"
        assert "error" in result.error_body
        assert "type" in result.error_body["error"]
        assert "message" in result.error_body["error"]

    @pytest.mark.asyncio
    async def test_to_json_response(self, basic_request, mock_auth):
        """ClaudeCodeErrorResponse.to_json_response() should produce correct JSONResponse."""
        upstream_error = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "Test error"}
        }
        mock_response = create_mock_response(400, json.dumps(upstream_error).encode())

        mock_client = MagicMock()
        mock_client.build_request = MagicMock(return_value=MagicMock())
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        with patch("anthropic_proxy.claude_code.httpx.AsyncClient", return_value=mock_client):
            result = await handle_claude_code_request(basic_request, "claude-code/claude-sonnet-4-5")

        json_response = result.to_json_response()
        assert json_response.status_code == 400
        assert json.loads(json_response.body) == upstream_error
