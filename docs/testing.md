# Testing

This document explains how to run the tests for the proxy.

## Running Tests

### Run All Tests

Use `uv run pytest` to run all tests (preferred method for consistency with uv-based development):

```bash
uv run pytest
```

### Run Specific Test Directories

```bash
# Run conversion tests only
uv run pytest tests/conversions/

# Run auth provider tests only
uv run pytest tests/auth/

# Run infrastructure tests only
uv run pytest tests/infrastructure/

# Run unit tests only
uv run pytest tests/unit/

# Run Gemini-specific tests only
uv run pytest tests/gemini/
```

### Run Specific Test Files

```bash
# Run a specific test file
uv run pytest tests/conversions/test_conversions.py

# Run tests matching a pattern
uv run pytest -k "test_conversion"
```

### Run with Coverage

```bash
# Generate coverage report
uv run pytest --cov=anthropic_proxy --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov=anthropic_proxy --cov-report=html
```

## Test Structure

Tests are organized into logical directories based on functionality:

### `tests/conversions/`
Format conversion tests between Anthropic, OpenAI, and Gemini formats.

- `test_conversions.py`: Request/response conversion tests
- `test_converters.py`: Converter package tests
- `test_insight_conversion.py`: Insight tag preservation tests

### `tests/auth/`
Authentication provider tests for OAuth-based providers (Codex, Gemini, Antigravity).

- `test_antigravity_auth.py`: Antigravity OAuth + project resolution
- `test_codex_auth.py`: Codex OAuth + token refresh + usage-limit mapping
- `test_codex_config.py`: Codex configuration tests
- `test_codex_dynamic_models.py`: Codex dynamic model loading
- `test_codex_partial_config.py`: Codex partial configuration handling
- `test_codex_routing.py`: Codex routing tests

### `tests/infrastructure/`
Infrastructure tests for CLI, daemon, and configuration management.

- `test_cli.py`: CLI argument parsing and command tests
- `test_cli_login.py`: CLI login command tests
- `test_daemon.py`: Daemon and PID management tests
- `test_config_manager.py`: Configuration management tests

### `tests/unit/`
Core functionality tests for hooks, routing, tokens, and utilities.

- `test_hooks.py`: Plugin hook system tests
- `test_routing.py`: Model format routing detection
- `test_token_parsing.py`: Token count parsing tests
- `test_signature_cache.py`: Signature caching tests
- `test_image_tokens.py`: Image token counting tests

### `tests/gemini/`
Gemini-specific functionality tests.

- `test_auth.py`: Gemini OAuth + project resolution
- `test_converter.py`: Gemini converter tests
- `test_sdk.py`: Gemini SDK integration tests
- `test_streaming.py`: Gemini streaming tests
- `test_thinking_recovery.py`: Thinking recovery tests
- `test_antigravity_conversion.py`: Antigravity conversion tests
- `test_antigravity_roundtrip.py`: Antigravity roundtrip tests
- `test_full_conversation_cycle.py`: Full conversation cycle tests
- `test_schema_sanitizer.py`: Schema sanitizer tests
- `test_routing.py`: Gemini routing tests
- `test_request.py`: Gemini request tests
- `test_client.py`: Gemini client tests
- `test_cli.py`: Gemini CLI tests

## Make Commands

Alternatively, use make commands for common tasks:

```bash
make test           # Run all tests
make test-cov       # Generate coverage report
make test-cov-html  # Generate HTML coverage report
```

## Notes

- Tests that require a live proxy server are intentionally removed (integration tests)
- Some auth tests may require valid OAuth credentials or mock responses
- Use `-v` flag for verbose output: `uv run pytest -v`
- Use `-s` flag to see print statements: `uv run pytest -s`
