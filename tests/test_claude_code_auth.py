"""Tests for Claude Code subscription authentication and caching."""

import pytest

from anthropic_proxy.claude_code import (
    CACHE_RETENTION_ENV,
    CLAUDE_CODE_SYSTEM_PREFIX,
    DEFAULT_CLAUDE_CODE_MODELS,
    THINKING_BUDGET_MAP,
    ClaudeCodeAuth,
    build_cache_control,
    build_claude_code_headers,
    get_cache_ttl,
    get_model_max_tokens,
    get_thinking_budget,
    inject_message_cache_control,
    inject_system_prompt,
    inject_system_prompt_with_cache,
    inject_tool_cache_control,
    is_oauth_token,
    is_opus_4_6_model,
    is_thinking_capable_model,
    process_thinking_config,
)


class TestIsOAuthToken:
    """Tests for is_oauth_token function."""

    def test_oauth_token(self):
        assert is_oauth_token("sk-ant-oat-abc123") is True

    def test_oauth01_token(self):
        assert is_oauth_token("sk-ant-oat01-abc123") is True

    def test_api_key(self):
        assert is_oauth_token("sk-ant-api03-abc123") is False

    def test_random_string(self):
        assert is_oauth_token("random-string") is False


class TestBuildClaudeCodeHeaders:
    """Tests for build_claude_code_headers function."""

    def test_headers_structure(self):
        headers = build_claude_code_headers("sk-ant-oat01-test")

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-ant-oat01-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "claude-cli" in headers["user-agent"]
        assert headers["x-app"] == "cli"
        assert headers["anthropic-dangerous-direct-browser-access"] == "true"
        assert "anthropic-beta" in headers

    def test_beta_features(self):
        headers = build_claude_code_headers("sk-ant-oat01-test")
        beta = headers["anthropic-beta"]

        assert "claude-code" in beta
        assert "oauth" in beta


class TestCacheControl:
    """Tests for cache control functions."""

    def test_get_cache_ttl_default(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        assert get_cache_ttl() is None

    def test_get_cache_ttl_long(self, monkeypatch):
        monkeypatch.setenv(CACHE_RETENTION_ENV, "long")
        assert get_cache_ttl() == "1h"

    def test_get_cache_ttl_other_value(self, monkeypatch):
        monkeypatch.setenv(CACHE_RETENTION_ENV, "short")
        assert get_cache_ttl() is None

    def test_build_cache_control_default(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        result = build_cache_control()
        assert result == {"type": "ephemeral"}
        assert "ttl" not in result

    def test_build_cache_control_long(self, monkeypatch):
        monkeypatch.setenv(CACHE_RETENTION_ENV, "long")
        result = build_cache_control()
        assert result == {"type": "ephemeral", "ttl": "1h"}


class TestInjectSystemPrompt:
    """Tests for inject_system_prompt function."""

    def test_no_existing_system(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {"model": "claude-sonnet-4-5", "messages": []}
        result = inject_system_prompt(request_data)

        assert "system" in result
        assert len(result["system"]) == 1
        assert result["system"][0]["type"] == "text"
        assert result["system"][0]["text"] == CLAUDE_CODE_SYSTEM_PREFIX
        assert result["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_string_system_prompt(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "model": "claude-sonnet-4-5",
            "messages": [],
            "system": "You are a helpful assistant.",
        }
        result = inject_system_prompt(request_data)

        assert len(result["system"]) == 2
        assert result["system"][0]["text"] == CLAUDE_CODE_SYSTEM_PREFIX
        assert "cache_control" not in result["system"][0]
        assert result["system"][1]["text"] == "You are a helpful assistant."
        # Only the last block should have cache_control
        assert result["system"][1]["cache_control"] == {"type": "ephemeral"}

    def test_array_system_prompt(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "model": "claude-sonnet-4-5",
            "messages": [],
            "system": [
                {"type": "text", "text": "First instruction."},
                {"type": "text", "text": "Second instruction."},
            ],
        }
        result = inject_system_prompt(request_data)

        assert len(result["system"]) == 3
        assert result["system"][0]["text"] == CLAUDE_CODE_SYSTEM_PREFIX
        assert result["system"][1]["text"] == "First instruction."
        assert result["system"][2]["text"] == "Second instruction."
        # Only the last block should have cache_control
        assert "cache_control" not in result["system"][0]
        assert "cache_control" not in result["system"][1]
        assert result["system"][2]["cache_control"] == {"type": "ephemeral"}

    def test_array_system_prompt_strips_existing_cache_control(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "model": "claude-sonnet-4-5",
            "messages": [],
            "system": [
                {
                    "type": "text",
                    "text": "First.",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "Second."},
            ],
        }
        result = inject_system_prompt(request_data)

        # Existing cache_control on non-last blocks should be stripped
        assert "cache_control" not in result["system"][0]
        assert "cache_control" not in result["system"][1]
        # Only the last block gets cache_control
        assert result["system"][2]["cache_control"] == {"type": "ephemeral"}

    def test_system_prompt_with_long_ttl(self, monkeypatch):
        monkeypatch.setenv(CACHE_RETENTION_ENV, "long")
        request_data = {"model": "claude-sonnet-4-5", "messages": []}
        result = inject_system_prompt(request_data)

        assert result["system"][0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }


class TestInjectMessageCacheControl:
    """Tests for inject_message_cache_control function."""

    def test_empty_messages(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {"messages": []}
        result = inject_message_cache_control(request_data)
        assert result["messages"] == []

    def test_string_content_last_user(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello!"},
            ]
        }
        result = inject_message_cache_control(request_data)

        last_msg = result["messages"][0]
        assert isinstance(last_msg["content"], list)
        assert last_msg["content"][0]["type"] == "text"
        assert last_msg["content"][0]["text"] == "Hello!"
        assert last_msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_array_content_last_user(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First part"},
                        {"type": "text", "text": "Second part"},
                    ],
                },
            ]
        }
        result = inject_message_cache_control(request_data)

        content = result["messages"][0]["content"]
        # Only last block should have cache_control
        assert "cache_control" not in content[0]
        assert content[1]["cache_control"] == {"type": "ephemeral"}

    def test_multiple_messages_only_last_user(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ]
        }
        result = inject_message_cache_control(request_data)

        # First user message should not have cache_control (string format unchanged)
        assert result["messages"][0]["content"] == "First question"
        # Last user message should have cache_control
        assert result["messages"][2]["content"][0]["cache_control"] == {
            "type": "ephemeral"
        }

    def test_last_message_is_assistant(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        result = inject_message_cache_control(request_data)

        # Should find the last user message (index 0) and add cache
        assert isinstance(result["messages"][0]["content"], list)
        assert result["messages"][0]["content"][0]["cache_control"] == {
            "type": "ephemeral"
        }

    def test_tool_result_content(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "123",
                            "content": "result",
                        },
                    ],
                },
            ]
        }
        result = inject_message_cache_control(request_data)

        assert result["messages"][0]["content"][0]["cache_control"] == {
            "type": "ephemeral"
        }

    def test_image_content(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "data": "..."}},
                    ],
                },
            ]
        }
        result = inject_message_cache_control(request_data)

        assert result["messages"][0]["content"][0]["cache_control"] == {
            "type": "ephemeral"
        }

    def test_long_ttl_message_cache(self, monkeypatch):
        monkeypatch.setenv(CACHE_RETENTION_ENV, "long")
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        result = inject_message_cache_control(request_data)

        assert result["messages"][0]["content"][0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }


class TestInjectToolCacheControl:
    """Tests for inject_tool_cache_control function."""

    def test_no_tools(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {"messages": []}
        result = inject_tool_cache_control(request_data)
        assert "tools" not in result

    def test_empty_tools(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {"tools": []}
        result = inject_tool_cache_control(request_data)
        assert result["tools"] == []

    def test_single_tool(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ]
        }
        result = inject_tool_cache_control(request_data)

        assert result["tools"][0]["cache_control"] == {"type": "ephemeral"}
        assert result["tools"][0]["name"] == "Read"

    def test_multiple_tools_only_last_gets_cache(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object"},
                },
                {
                    "name": "Write",
                    "description": "Write a file",
                    "input_schema": {"type": "object"},
                },
                {
                    "name": "Bash",
                    "description": "Run a command",
                    "input_schema": {"type": "object"},
                },
            ]
        }
        result = inject_tool_cache_control(request_data)

        # Only the last tool should have cache_control
        assert "cache_control" not in result["tools"][0]
        assert "cache_control" not in result["tools"][1]
        assert result["tools"][2]["cache_control"] == {"type": "ephemeral"}

    def test_long_ttl(self, monkeypatch):
        monkeypatch.setenv(CACHE_RETENTION_ENV, "long")
        request_data = {
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object"},
                },
            ]
        }
        result = inject_tool_cache_control(request_data)

        assert result["tools"][0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    def test_preserves_existing_tool_fields(self, monkeypatch):
        monkeypatch.delenv(CACHE_RETENTION_ENV, raising=False)
        request_data = {
            "tools": [
                {
                    "name": "Search",
                    "description": "Search the web",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            ]
        }
        result = inject_tool_cache_control(request_data)

        tool = result["tools"][0]
        assert tool["name"] == "Search"
        assert tool["description"] == "Search the web"
        assert tool["input_schema"]["required"] == ["query"]
        assert tool["cache_control"] == {"type": "ephemeral"}


class TestDefaultModels:
    """Tests for default Claude Code models."""

    def test_models_defined(self):
        assert len(DEFAULT_CLAUDE_CODE_MODELS) > 0

    def test_claude_models(self):
        # Should have Claude 4.6 Opus and 4.5 Sonnet, Haiku
        model_ids = list(DEFAULT_CLAUDE_CODE_MODELS.keys())

        assert "claude-opus-4-6" in model_ids
        assert "claude-opus-4-5" in model_ids
        assert "claude-sonnet-4-5" in model_ids
        assert "claude-haiku-4-5" in model_ids

    def test_model_structure(self):
        for model_id, details in DEFAULT_CLAUDE_CODE_MODELS.items():
            assert "model_name" in details
            assert "description" in details
            assert "max_tokens" in details
            # Opus 4.6 has 128K max output, others have 64K
            if "opus-4-6" in model_id:
                assert details["max_tokens"] == 128000
            else:
                assert details["max_tokens"] == 64000


class TestClaudeCodeAuth:
    """Tests for ClaudeCodeAuth class."""

    def test_has_auth_no_token(self, tmp_path, monkeypatch):
        # Mock config directory to use tmp_path
        monkeypatch.setattr(
            "anthropic_proxy.config_manager.DEFAULT_CONFIG_DIR",
            tmp_path,
        )
        monkeypatch.setattr(
            "anthropic_proxy.config_manager.DEFAULT_AUTH_FILE",
            tmp_path / "auth.json",
        )

        auth = ClaudeCodeAuth()
        assert auth.has_auth() is False

    def test_save_and_get_token(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")

        monkeypatch.setattr(
            "anthropic_proxy.config_manager.DEFAULT_CONFIG_DIR",
            tmp_path,
        )
        monkeypatch.setattr(
            "anthropic_proxy.config_manager.DEFAULT_AUTH_FILE",
            auth_file,
        )

        auth = ClaudeCodeAuth()
        auth.save_token("sk-ant-oat01-test-token")

        assert auth.has_auth() is True
        assert auth.get_token() == "sk-ant-oat01-test-token"

    def test_invalid_token_format(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")

        monkeypatch.setattr(
            "anthropic_proxy.config_manager.DEFAULT_CONFIG_DIR",
            tmp_path,
        )
        monkeypatch.setattr(
            "anthropic_proxy.config_manager.DEFAULT_AUTH_FILE",
            auth_file,
        )

        auth = ClaudeCodeAuth()

        with pytest.raises(ValueError, match="Invalid token format"):
            auth.save_token("invalid-token")

    def test_clear_token(self, tmp_path, monkeypatch):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text("{}")

        monkeypatch.setattr(
            "anthropic_proxy.config_manager.DEFAULT_CONFIG_DIR",
            tmp_path,
        )
        monkeypatch.setattr(
            "anthropic_proxy.config_manager.DEFAULT_AUTH_FILE",
            auth_file,
        )

        auth = ClaudeCodeAuth()
        auth.save_token("sk-ant-oat01-test-token")
        assert auth.has_auth() is True

        auth.clear_token()
        assert auth.has_auth() is False


class TestIsOpus46Model:
    """Tests for Opus 4.6 model detection."""

    def test_opus_4_6(self):
        assert is_opus_4_6_model("claude-opus-4-6") is True

    def test_opus_4_6_with_prefix(self):
        assert is_opus_4_6_model("claude-code/claude-opus-4-6") is True

    def test_opus_4_5_is_not_4_6(self):
        assert is_opus_4_6_model("claude-opus-4-5") is False

    def test_sonnet_is_not_4_6(self):
        assert is_opus_4_6_model("claude-sonnet-4-5") is False

    def test_haiku_is_not_4_6(self):
        assert is_opus_4_6_model("claude-haiku-4-5") is False


class TestOpus46DefaultModels:
    """Tests for Opus 4.6 default model config."""

    def test_opus_4_6_has_reasoning_effort(self):
        config = DEFAULT_CLAUDE_CODE_MODELS["claude-opus-4-6"]
        assert "reasoning_effort" in config

    def test_opus_4_5_no_reasoning_effort(self):
        config = DEFAULT_CLAUDE_CODE_MODELS["claude-opus-4-5"]
        assert "reasoning_effort" not in config


class TestOpus46AdaptiveThinking:
    """Tests for Opus 4.6 adaptive thinking behavior.

    Opus 4.6 uses adaptive thinking by default instead of budget-based.
    Budget-based is only used when explicit budget_tokens is provided.
    """

    def test_opus_4_6_default_uses_adaptive(self):
        """When thinking is enabled without budget, Opus 4.6 uses adaptive."""
        request_data = {
            "model": "claude-opus-4-6",
            "max_tokens": 128000,
            "thinking": {"type": "enabled"},
        }
        result, enabled = process_thinking_config(request_data, "claude-opus-4-6")
        assert enabled is True
        assert result["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in result["thinking"]

    def test_opus_4_6_explicit_budget_uses_budget_based(self):
        """When explicit budget_tokens is provided, Opus 4.6 uses budget-based."""
        request_data = {
            "model": "claude-opus-4-6",
            "max_tokens": 128000,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        result, enabled = process_thinking_config(request_data, "claude-opus-4-6")
        assert enabled is True
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 10000

    def test_opus_4_6_disabled_stays_disabled(self):
        """Disabled thinking remains disabled for Opus 4.6."""
        request_data = {
            "model": "claude-opus-4-6",
            "max_tokens": 128000,
            "thinking": {"type": "disabled"},
        }
        result, enabled = process_thinking_config(request_data, "claude-opus-4-6")
        assert enabled is False
        assert "thinking" not in result

    def test_opus_4_6_no_thinking_field(self):
        """No thinking field means no thinking for Opus 4.6."""
        request_data = {
            "model": "claude-opus-4-6",
            "max_tokens": 128000,
        }
        result, enabled = process_thinking_config(request_data, "claude-opus-4-6")
        assert enabled is False
        assert "thinking" not in result

    def test_opus_4_6_string_effort_uses_adaptive(self):
        """String effort levels (low/medium/high) use adaptive for Opus 4.6."""
        for level in ["low", "medium", "high"]:
            request_data = {
                "model": "claude-opus-4-6",
                "max_tokens": 128000,
                "thinking": level,
            }
            result, enabled = process_thinking_config(request_data, "claude-opus-4-6")
            assert enabled is True, f"Expected thinking enabled for effort={level}"
            assert result["thinking"]["type"] == "adaptive", (
                f"Expected adaptive for effort={level}, got {result['thinking']}"
            )
            assert "budget_tokens" not in result["thinking"]

    def test_opus_4_6_minimal_disables_thinking(self):
        """Minimal effort still disables thinking for Opus 4.6."""
        request_data = {
            "model": "claude-opus-4-6",
            "max_tokens": 128000,
            "thinking": "minimal",
        }
        result, enabled = process_thinking_config(request_data, "claude-opus-4-6")
        assert enabled is False
        assert "thinking" not in result

    def test_opus_4_6_max_effort_uses_adaptive(self):
        """Max effort level uses adaptive thinking for Opus 4.6."""
        request_data = {
            "model": "claude-opus-4-6",
            "max_tokens": 128000,
            "thinking": "max",
        }
        result, enabled = process_thinking_config(request_data, "claude-opus-4-6")
        assert enabled is True
        assert result["thinking"]["type"] == "adaptive"
        assert "budget_tokens" not in result["thinking"]

    def test_non_opus_46_max_falls_back_to_high(self):
        """Max effort falls back to high for non-Opus 4.6 models."""
        request_data = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 64000,
            "thinking": "max",
        }
        result, enabled = process_thinking_config(request_data, "claude-sonnet-4-5")
        assert enabled is True
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == THINKING_BUDGET_MAP["high"]

    def test_non_opus_46_still_uses_budget_based(self):
        """Non-Opus 4.6 models keep using budget-based thinking."""
        request_data = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 64000,
            "thinking": {"type": "enabled"},
        }
        result, enabled = process_thinking_config(request_data, "claude-sonnet-4-5")
        assert enabled is True
        assert result["thinking"]["type"] == "enabled"
        assert "budget_tokens" in result["thinking"]

    def test_opus_4_6_adaptive_no_max_tokens_adjustment(self):
        """Adaptive thinking doesn't need max_tokens > budget + buffer."""
        request_data = {
            "model": "claude-opus-4-6",
            "max_tokens": 4096,
            "thinking": {"type": "enabled"},
        }
        result, enabled = process_thinking_config(request_data, "claude-opus-4-6")
        assert enabled is True
        assert result["thinking"]["type"] == "adaptive"
        # max_tokens should stay as-is (no budget to inflate for)
        assert result["max_tokens"] == 4096
