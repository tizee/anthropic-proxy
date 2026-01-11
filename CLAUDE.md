# AGENTS.md

Guidance for working on this repository.

## Quick Start
- Use `uv` to run Python commands.
- See `Makefile` for common tasks. Run `make help` for the list.
- Dependencies and tool config live in `pyproject.toml`.

## Project Summary
This proxy converts Anthropic API requests to OpenAI-compatible APIs (and back). It is designed to sit behind **ccproxy**, which provides API keys via request headers.

Key points:
- **API keys** come from the `Authorization` header (ccproxy sets `ANTHROPIC_AUTH_TOKEN`). The proxy does not read API keys from `.env`.
- **Model selection** is driven by the incoming request model. The server does not perform routing or model switching.
- **models.yaml** defines model → API URL mappings and per-model options (no `api_key_name`, no pricing fields).
- **/v1/messages/count_tokens** returns a local tiktoken-based estimate including messages, system, tools, thinking, and tool_choice.

## Architecture Reference
- `anthropic_proxy/server.py`: FastAPI endpoints, request handling, and response conversion.
- `anthropic_proxy/client.py`: Loads `models.yaml` and creates provider-specific clients.
- `anthropic_proxy/converter.py`: Anthropic ↔ OpenAI request/response conversion.
- `anthropic_proxy/streaming.py`: Streaming conversion logic.
- `anthropic_proxy/types.py`: Pydantic models and API schemas.
- `anthropic_proxy/utils.py`: Usage tracking and error helpers.
- `anthropic_proxy/daemon.py`: Background daemon management (PID file, process control).
- `anthropic_proxy/cli.py`: Command-line interface with subcommands (start/stop/restart/status).
- `docs/api-response-formats.md`: Reference documentation for API response formats used by different providers (Claude, OpenAI, DeepSeek, VolcEngine).

## Running the Server

The server runs as a background daemon. All output is logged to `~/.anthropic-proxy/daemon.log`.

```bash
# Start server
anthropic-proxy start

# Check status
anthropic-proxy status

# Stop server
anthropic-proxy stop

# View logs for debugging
tail -f ~/.anthropic-proxy/daemon.log
```

## Configuration Files

Located in `~/.config/anthropic-proxy/`:
- `models.yaml`: Model configurations (model_id, api_base, direct mode, etc.)
- `config.json`: Server settings (log_level, log_file_path, host, port, env)

Runtime files in `~/.anthropic-proxy/`:
- `anthropic-proxy.pid`: PID file for tracking running daemon
- `daemon.log`: Server output logs
- `server.log`: Application logs (configured in config.json)

## Testing
- `make test` runs the current unit/conversion suite.
- Integration tests that require a live proxy server are intentionally removed.

## Code Design Guidelines
- Keep functions focused and readable (single responsibility).
- Prefer simple, explicit logic over clever abstractions.
- Extract helpers when functions grow too large or mix concerns.
- Maintain testability: small units are easier to verify.
