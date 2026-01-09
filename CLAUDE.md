# CLAUDE.md

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
- **/v1/messages/count_tokens** returns a fixed fallback response for client compatibility; local token counting is removed.

## Architecture Reference
- `anthropic_proxy/server.py`: FastAPI endpoints, request handling, and response conversion.
- `anthropic_proxy/client.py`: Loads `models.yaml` and creates provider-specific clients.
- `anthropic_proxy/converter.py`: Anthropic ↔ OpenAI request/response conversion.
- `anthropic_proxy/streaming.py`: Streaming conversion logic.
- `anthropic_proxy/types.py`: Pydantic models and API schemas.
- `anthropic_proxy/utils.py`: Usage tracking and error helpers.

## Testing
- `make test` runs the current unit/conversion suite.
- Integration tests that require a live proxy server are intentionally removed.

## Code Design Guidelines
- Keep functions focused and readable (single responsibility).
- Prefer simple, explicit logic over clever abstractions.
- Extract helpers when functions grow too large or mix concerns.
- Maintain testability: small units are easier to verify.
