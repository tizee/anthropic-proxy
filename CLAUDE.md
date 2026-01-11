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

## Architecture Reference

### Core Modules
- `anthropic_proxy/server.py`: FastAPI endpoints, request handling, and response conversion.
- `anthropic_proxy/client.py`: Loads `models.yaml` and creates provider-specific clients.
- `anthropic_proxy/converter.py`: Anthropic ↔ OpenAI request/response conversion.
- `anthropic_proxy/streaming.py`: Streaming conversion logic.
- `anthropic_proxy/types.py`: Pydantic models and API schemas.
- `anthropic_proxy/utils.py`: Usage tracking and error helpers.

### Daemon & CLI
- `anthropic_proxy/daemon.py`: Background daemon management (PID file, process control, start/stop).
- `anthropic_proxy/cli.py`: Command-line interface with subcommands (start/stop/restart/status/init).
- `anthropic_proxy/config_manager.py`: Configuration directory and file management (create/load config).
- `anthropic_proxy/config.py`: Runtime configuration object (log_level, host, port, etc.).
- `anthropic_proxy/__main__.py`: Entry point for the `anthropic-proxy` CLI command.

### Documentation
- `docs/api-response-formats.md`: Reference documentation for API response formats used by different providers (Claude, OpenAI, DeepSeek, VolcEngine).

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
- `models.yaml`: Model configurations (model_id, api_base, direct mode, etc.)
- `config.json`: Server settings (log_level, log_file_path, host, port)

Runtime files in `~/.anthropic-proxy/`:
- `anthropic-proxy.pid`: PID file for tracking running daemon
- `daemon.log`: Server output logs (stdout/stderr from daemon process)
- `server.log`: Application logs (configured in config.json)

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
- Integration tests that require a live proxy server are intentionally removed.

### Test Structure
- `tests/test_conversions.py`: Request/response conversion tests
- `tests/test_routing.py`: Model routing and direct mode detection
- `tests/test_hooks.py`: Plugin hook system tests
- `tests/test_insight_conversion.py`: Insight tag preservation tests
- `tests/test_token_parsing.py`: Token count parsing tests
- `tests/test_daemon.py`: Daemon and PID management tests
- `tests/test_config_manager.py`: Configuration management tests
- `tests/test_cli.py`: CLI argument parsing and command tests

## Code Design Guidelines
- Keep functions focused and readable (single responsibility).
- Prefer simple, explicit logic over clever abstractions.
- Extract helpers when functions grow too large or mix concerns.
- Maintain testability: small units are easier to verify.
- Use `uv` for all Python package management (no pip, no venv).
- Configuration via files (models.yaml, config.json), not environment variables.
