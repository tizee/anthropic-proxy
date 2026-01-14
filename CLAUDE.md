# AGENTS.md

Guidance for working on this repository.

## Quick Start
- Use `uv` to run Python commands.
- See `Makefile` for common tasks. Run `make help` for the list.
- Dependencies and tool config live in `pyproject.toml`.
- Install with: `uv pip install -e .`
- Uninstall with: `uv pip uninstall anthropic-proxy`

## Project Summary
This proxy converts Anthropic API requests to OpenAI-compatible APIs (and back). It is designed to sit behind **ccproxy**, which provides API keys via request headers.

Key points:
- **API keys** come from the `Authorization` header (ccproxy sets `ANTHROPIC_AUTH_TOKEN`). The proxy does not read API keys from `.env`.
- **Model selection** is driven by the incoming request model. The server does not perform routing or model switching.
- **models.yaml** defines model → API URL mappings and per-model options (no `api_key_name`, no pricing fields).
- **/v1/messages/count_tokens** returns a local tiktoken-based estimate including messages, system, tools, thinking, and tool_choice.
- **Default auth-provider models may drift**: Codex/Gemini/Antigravity default model IDs are best-effort and can become invalid if upstream providers change or disable support. Keep docs/tests in sync when updating.
- **Auth default model IDs are prefixed**: Codex/Gemini/Antigravity defaults use `codex/`, `gemini/`, `antigravity/` prefixes to avoid collisions; user-defined IDs take precedence.

### Format-Based Operation
The proxy routes requests based on the `format` field in models.yaml (with `direct` as a legacy alias):
- **Anthropic format** (`format: anthropic` or legacy `direct: true`): Routes to Anthropic-compatible APIs without format conversion. Providers include Moonshot AI (Kimi), DeepSeek, Zhipu GLM, MiniMax. Uses httpx client.
- **OpenAI format** (`format: openai`): Converts Anthropic format to OpenAI format for third-party providers. Uses AsyncOpenAI SDK client.
- **Gemini format** (`format: gemini`): Routes via Gemini Code Assist subscriptions (Gemini/Antigravity).

## Architecture Reference

### Core Modules
- `anthropic_proxy/server.py`: FastAPI endpoints, request handling, and response conversion.
- `anthropic_proxy/client.py`: Loads `models.yaml` and creates provider-specific clients.
- `anthropic_proxy/openai_converter.py`: Anthropic ↔ OpenAI request/response conversion.
- `anthropic_proxy/gemini_converter.py`: Anthropic ↔ Gemini request/response conversion.
- `anthropic_proxy/converter.py`: Converter facade (re-exports format-specific converters).
- `anthropic_proxy/streaming.py`: Streaming conversion facade.
- `anthropic_proxy/types.py`: Pydantic models and API schemas.
- `anthropic_proxy/utils.py`: Usage tracking and error helpers.

### Auth Providers
- `anthropic_proxy/auth_provider.py`: Shared OAuth PKCE login + refresh base.
- `anthropic_proxy/codex.py`: Codex OAuth + request handling (maps usage-limit 404 -> 429).
- `anthropic_proxy/gemini.py`: Gemini OAuth + project resolution (Code Assist).
- `anthropic_proxy/antigravity.py`: Antigravity OAuth + project resolution (internal gateway).
- Default auth-provider models are defined in each provider module and may become invalid if upstream changes.

### Gemini SDK Integration
- `anthropic_proxy/gemini_sdk.py`: Uses google-genai SDK to send Gemini-style requests.
- `anthropic_proxy/gemini_converter.py`: Ensures Gemini schema/tool/thinking normalization.

### Daemon & CLI
- `anthropic_proxy/daemon.py`: Background daemon management (PID file, process control, start/stop).
- `anthropic_proxy/cli.py`: Command-line interface with subcommands (start/stop/restart/status/init).
- `anthropic_proxy/config_manager.py`: Configuration directory and file management (create/load config).
- `anthropic_proxy/config.py`: Runtime configuration object (log_level, host, port, etc.).
- `anthropic_proxy/__main__.py`: Entry point for the `anthropic-proxy` CLI command.

### Documentation
- `docs/api-response-formats.md`: Reference documentation for API response formats used by different providers (Claude, OpenAI, DeepSeek, VolcEngine).

### Plugin System
- `anthropic_proxy/hook.py`: `HookManager` class that dynamically loads plugins from `anthropic_proxy/plugins/`
- `anthropic_proxy/plugins/filter_tools.py`: Built-in plugin that filters WebSearch, NotebookEdit, NotebookRead tools
- Plugins can define `request_hook(payload)` and/or `response_hook(payload)` functions to modify payloads

### Streaming Architecture
- `AnthropicStreamingConverter` (in `streaming.py`) manages a state machine for SSE chunk conversion
- Handles content blocks, tool calls, thinking blocks during streaming
- Tracks token counts from provider responses (not local estimation)

## Running the Server

The server runs as a background daemon managed by the CLI. All output is logged to `~/.anthropic-proxy/daemon.log`.

```bash
# Initialize config files (first time)
anthropic-proxy --init

# Start server in background
anthropic-proxy start

# Start with custom port
anthropic-proxy start --port 8080

# Check status
anthropic-proxy status

# Stop server
anthropic-proxy stop

# Restart server
anthropic-proxy restart

# View logs for debugging
tail -f ~/.anthropic-proxy/daemon.log
```

### Development Mode

For development, run the server directly in the foreground (not as a daemon):

```bash
# Using make
make dev

# Or directly with uv
uv run -m anthropic_proxy.server --host 0.0.0.0 --port 8082
```

**Note**: Development mode does not support auto-reload. Restart manually after code changes.

## Configuration Files

Located in `~/.config/anthropic-proxy/`:
- `models.yaml`: Model configurations (model_id, api_base, format/direct, etc.)
- `config.json`: Server settings (log_level, log_file_path, host, port)

Runtime files in `~/.anthropic-proxy/`:
- `anthropic-proxy.pid`: PID file for tracking running daemon
- `daemon.log`: All stdout/stderr from daemon process (server startup, configuration, exceptions, uvicorn/fastapi logs)
- `server.log`: Application logs configured by `log_level` in config.json (default: WARNING)

**Log file distinction**: `daemon.log` captures everything including startup errors and FastAPI access logs. `server.log` only contains application-level logs at the configured severity.

## Model Selection

### model_id vs model_name
- `model_id`: Unique key used in incoming requests (what ccproxy specifies as `model`)
- `model_name`: Actual provider model name sent to upstream API
- Multiple `model_id` entries can map to the same `model_name` with different settings

### reasoning_effort
Controls thinking mode behavior. Supports: `minimal`, `low`, `medium`, `high`
- `minimal`: No thinking (thinking block disabled)
- `low`/`medium`/`high`: Progressive thinking intensity

This allows creating "reasoning level" variants of the same base model:
```yaml
- model_id: doubao-nothinking
  model_name: doubao-1-8
  reasoning_effort: minimal

- model_id: doubao-high
  model_name: doubao-1-8
  reasoning_effort: high
```

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `anthropic-proxy start [--host HOST] [--port PORT]` | Start server in background |
| `anthropic-proxy stop` | Stop running server |
| `anthropic-proxy restart [--host HOST] [--port PORT]` | Restart server |
| `anthropic-proxy status` | Show server status |
| `anthropic-proxy --init` | Initialize config files (skip if exist) |
| `anthropic-proxy --init-force` | Force reinitialize config files |
| `anthropic-proxy --print-config [--show-api-keys]` | Print current configuration |

## Testing
- `make test` runs the full test suite.
- `make test-cov` generates coverage report.
- `make test-cov-html` generates HTML coverage report.
- Single test suites: `make test-routing`, `make test-hooks`, `make test-conversion`
- Integration tests that require a live proxy server are intentionally removed.
 - Auth tests: `uv run -m pytest tests/test_codex_auth.py tests/test_gemini_auth.py tests/test_antigravity_auth.py`

### Code Quality
- `make lint` checks and fixes code with ruff.
- `make format` formats code with ruff.

### Test Structure
- `tests/test_conversions.py`: Request/response conversion tests
- `tests/test_routing.py`: Model format routing detection
- `tests/test_hooks.py`: Plugin hook system tests
- `tests/test_insight_conversion.py`: Insight tag preservation tests
- `tests/test_token_parsing.py`: Token count parsing tests
- `tests/test_daemon.py`: Daemon and PID management tests
- `tests/test_config_manager.py`: Configuration management tests
- `tests/test_cli.py`: CLI argument parsing and command tests
- `tests/test_gemini_auth.py`: Gemini OAuth + project resolution
- `tests/test_antigravity_auth.py`: Antigravity OAuth + project resolution
- `tests/test_codex_auth.py`: Codex OAuth + token refresh + usage-limit mapping

## Auth Notes & Gotchas
- OAuth tokens live in `~/.config/anthropic-proxy/auth.json`. Refresh tokens are required for auto-refresh.
- Gemini/Antigravity require a project ID resolved via `v1internal:loadCodeAssist` (and `onboardUser` fallback).
- Antigravity and Gemini default model IDs are best-effort; update docs/tests when upstream changes.
- Codex default models are best-effort; usage-limit errors may be returned as 404 and are mapped to 429.
- Antigravity Claude thinking signatures are cached in-memory per session (see `anthropic_proxy/signature_cache.py`); supply a stable `metadata.session_id` (or similar) for multi-turn recovery.

## Code Design Guidelines
- Keep functions focused and readable (single responsibility).
- Prefer simple, explicit logic over clever abstractions.
- Extract helpers when functions grow too large or mix concerns.
- Maintain testability: small units are easier to verify.
- Use `uv` for all Python package management (no pip, no venv).
- Configuration via files (models.yaml, config.json), not environment variables.
