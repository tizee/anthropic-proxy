# Task Plan: Make claude-code-proxy compatible with ccproxy

## Goal
Refactor claude-code-proxy to work seamlessly with ccproxy, allowing the proxy to act as a format conversion server that receives API keys from ccproxy via request headers instead of managing its own API key configuration.

## Current State Analysis

### ccproxy (wrapper script)
- Reads provider config from `~/.config/llm/cc-proxy.json`
- Sets environment variables: `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`, etc.
- Launches `claude` command with these env vars
- Config schema includes: `base_url`, `auth_key`, `model`, `small_model`, `ANTHROPIC_DEFAULT_*_MODEL`

### claude-code-proxy (this project) - AFTER REFACTORING
- API keys are passed via request headers from ccproxy (no .env required for API keys)
- Model configs in `models.yaml` only define model→URL mappings (no `api_key_name`)
- Routes model names to provider APIs with format conversion
- Converts Anthropic API format ↔ OpenAI API format

## Finalized Architecture

```
ccproxy (sets ANTHROPIC_BASE_URL=localhost:8082, provides API key via Authorization header)
    ↓
Claude Code (sends Anthropic-format requests with model name + auth header)
    ↓
claude-code-proxy (extracts key from header, routes model to API URL, converts format)
    ↓
Provider API (receives OpenAI-format request with API key)
```

**How it works:**
1. ccproxy sets `ANTHROPIC_BASE_URL=http://localhost:8082` (this proxy)
2. ccproxy sets `ANTHROPIC_AUTH_TOKEN=<api_key>` which becomes `Authorization: Bearer <api_key>` header
3. Claude Code sends requests to proxy with model name (e.g., `deepseek-chat`) and auth header
4. Proxy extracts API key from incoming `Authorization` header
5. Proxy looks up model→API URL mapping (from simplified models.yaml)
6. Proxy converts Anthropic→OpenAI format, forwards with extracted key
7. Proxy converts response back to Anthropic format

## Phases - ALL COMPLETED ✅

- [x] Phase 1: Analyze current request handling and header extraction
- [x] Phase 2: Refactor client.py - pass API key from request instead of config
- [x] Phase 3: Simplify config.py - remove API key management (custom_api_keys, validate_api_keys, etc.)
- [x] Phase 4: Simplify models.yaml - remove api_key_name field
- [x] Phase 5: Update server.py - extract API key from Authorization header via `extract_api_key()` function
- [x] Phase 6: Clean up tests (removed redundant test_server.py - tested by OpenAI SDK)
- [x] Phase 7: Update documentation (README.md, docs/architecture.md)
- [x] Phase 8: Remove pricing calculation logic (calculate_cost, model_pricing, cost constants)

## Key Changes Made

### client.py
- `create_openai_client(model_id)` → `create_openai_client(model_id, api_key)`
- `create_claude_client(model_id)` → `create_claude_client(model_id, api_key)`
- `initialize_custom_models()` no longer loads API keys from environment
- `load_custom_models()` no longer processes pricing or api_key_name fields

### config.py
- Removed `custom_api_keys` dictionary
- Removed `add_custom_api_key()`, `get_api_key_for_provider()`, `validate_api_keys()` methods
- Removed `model_pricing` attribute

### server.py
- Added `extract_api_key(request)` function to read `Authorization` header
- Updated all client creation calls to pass extracted API key

### types.py
- Removed pricing-related constants from `ModelDefaults`

### utils.py
- Removed `calculate_cost()` function
- Updated `add_session_stats()` to remove cost parameter

### models.yaml
- Removed all `api_key_name` fields
- Removed all pricing fields (`input_cost_per_million_tokens`, `output_cost_per_million_tokens`)

## Status
**COMPLETED** - Refactoring finished successfully
