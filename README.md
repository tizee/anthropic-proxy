# Anthropic API Proxy for Claude Code 🔄

[![GitHub latest commit](https://img.shields.io/github/last-commit/tizee/anthropic-proxy)](https://github.com/tizee/anthropic-proxy)
[![License](https://img.shields.io/github/license/tizee/anthropic-proxy)](https://github.com/tizee/anthropic-proxy/blob/main/LICENSE)

A proxy server that enables Claude Code to work with multiple model providers through **multi-format endpoints** and format-based routing:

1. **OpenAI-Compatible Format** (`format: openai`): Translates Anthropic API requests to OpenAI-compatible endpoints
2. **Anthropic-Compatible Format** (`format: anthropic`): Routes requests directly to official Claude API or compatible endpoints
3. **Gemini Format** (`format: gemini`): Routes requests through Gemini-compatible backends (Gemini/Antigravity subscriptions)

**Multi-Format Support**: The proxy accepts requests in Anthropic, OpenAI, or Gemini format and automatically converts between them as needed. Each format has its own endpoint prefix (`/anthropic/v1/...`, `/openai/v1/...`, `/gemini/v1beta/...`).

- kimi official supports Anthropic API https://api.moonshot.cn/anthropic
- deepseek supports Anthropic https://api-docs.deepseek.com/guides/anthropic_api
- Zhipu GLM supports Anthropic API https://open.bigmodel.cn/api/anthropic
- MiniMax supports Anthropic API: https://api.minimax.io/anthropic (international users), https://api.minimaxi.com/anthropic (China users) - reference: https://platform.minimax.io/docs/api-reference/text-anthropic-api
- VolcEngine 方舟 Coding Plan supports Anthropic API https://ark.cn-beijing.volces.com/api/coding (requires Coding Plan subscription) - reference: https://www.volcengine.com/docs/82379/1928262?lang=zh

This allows you to use Claude Code with OpenAI-compatible models, native Claude API endpoints, and Gemini subscriptions. For third-party models to support Claude Code image files (URL/base64), they must natively support multimodal image understanding.

## Primary Use Case: Claude Code Proxy

The main purpose of this project is to serve as a proxy for **Claude Code**, enabling it to connect to and utilize third-party models that follow the OpenAI API format. This extends the power of Claude Code beyond its native models.

### Recommended Usage Strategy

**1. Primary Choice: Official Claude Models**

If you have a Claude Pro subscription or API access, it is highly recommended to use the official Anthropic models as your default choice. This ensures the best performance, latest features, and full compatibility.

**2. Fallback/Alternative: This Proxy**

Use this proxy in the following scenarios:
- When your official Claude API quota has been exhausted.
- As a cost-effective alternative for less critical tasks.
- To experiment with different models while maintaining the Claude Code workflow.

## Supported Claude Code Versions

| Version | Status | Notes |
|---------|--------|-------|
| 2.1.3 | ✅ Tested | Current latest version (as of January 2026) |

## Third-Party Providers Supporting Anthropic Format

The following providers offer native Anthropic API compatibility, allowing Anthropic format usage without conversion:

| Provider | API Endpoint | Supported Models |
|----------|-------------|------------------|
| **Moonshot AI (Kimi)** | `https://api.moonshot.cn/anthropic` | kimi-k2-0711-preview, kimi-k2-0711-chat |
| **DeepSeek** | `https://api.deepseek.com/anthropic` | deepseek-chat, deepseek-reasoner |
| **Zhipu AI (GLM)** | `https://open.bigmodel.cn/api/anthropic` | glm-4.7, glm-4.6 |
| **MiniMax** | `https://api.minimax.io/anthropic` (international) / `https://api.minimaxi.com/anthropic` (China) | MiniMax-M2.1 |
| **VolcEngine (方舟 Coding Plan)** | `https://ark.cn-beijing.volces.com/api/coding` | ark-code-latest (requires Coding Plan subscription) |


### Configuration Example

For providers with native Anthropic support, configure as Anthropic format:

```yaml
- model_id: kimi-k2-direct
  model_name: kimi-k2-0711-preview
  api_base: https://api.moonshot.cn/anthropic
  format: anthropic
  max_tokens: 16k
  max_input_tokens: 200k

- model_id: volcengine-ark-code
  model_name: ark-code-latest
  api_base: https://ark.cn-beijing.volces.com/api/coding
  format: anthropic
  # Note: This requires a VolcEngine Coding Plan subscription
  max_tokens: 16k
  max_input_tokens: 200k
```

## API Endpoints

The proxy provides multiple endpoint formats to accept requests in different API styles:

### Anthropic Format Endpoints

**`POST /anthropic/v1/messages`**
- Primary endpoint for Anthropic Messages API format
- Accepts requests in Anthropic format, routes to configured providers
- Returns responses in Anthropic format

**`POST /anthropic/v1/messages/count_tokens`**
- Token counting endpoint
- Returns local tiktoken-based estimate including messages, system, tools, thinking, and tool_choice

**`GET /anthropic/v1/stats`**
- Returns comprehensive token usage statistics for the current session

**`POST /anthropic/v1/messages/test_conversion`**
- Test endpoint for direct message format conversion
- Sends requests directly to specified model without server-side model switching

### OpenAI Format Endpoints

**`POST /openai/v1/chat/completions`**
- OpenAI Chat Completions API compatible endpoint
- Accepts requests in OpenAI format
- Converts to appropriate provider format internally
- Returns responses in OpenAI format

**`POST /openai/v1/responses`**
- OpenAI Responses API compatible endpoint (for Codex subscription)
- Direct passthrough to Codex backend without format conversion
- Requires Codex authentication (`anthropic-proxy login --codex`)
- Supports both streaming and non-streaming modes
- **Use case**: Agents/tools that natively use OpenAI Responses API format

### Gemini Format Endpoints

**`POST /gemini/v1beta/models/{model}:generateContent`**
- Gemini GenerateContent API compatible endpoint (non-streaming)
- Accepts requests in Gemini format
- Converts to appropriate provider format internally
- Returns responses in Gemini format

**`POST /gemini/v1beta/models/{model}:streamGenerateContent`**
- Gemini StreamGenerateContent API compatible endpoint
- Same as above but returns streaming responses in Gemini format

### Utility Endpoints

**`GET /test-connection`**
- Test API connectivity to configured providers
- Returns configuration status and model count

## Key Features

### 🔄 Format-Based Operation
- **OpenAI-Compatible Format**: Convert Anthropic API requests to OpenAI format for third-party providers
- **Anthropic-Compatible Format**: Route requests directly to official Anthropic API with native format preservation
- **Gemini Format**: Route requests via Gemini Code Assist (subscription-backed)

**Converter Architecture**: The proxy uses a unified converter package (`anthropic_proxy/converters/`) with:
- Base classes (`BaseConverter`, `BaseStreamingConverter`) for consistent converter interfaces
- Format-specific converters (OpenAI, Anthropic, Gemini) that inherit from base classes
- Factory functions (`get_converter()`, `get_streaming_converter()`) for dynamic converter selection
- Legacy implementation modules (`_openai_impl.py`, `_gemini_impl.py`) for backward compatibility

### 🔗 ccproxy Integration (Recommended)
This proxy is designed to work seamlessly with **ccproxy** (Claude Code wrapper script):
- **API keys managed by ccproxy**: No need to store keys in the proxy's config - they're passed via request headers
- **Simplified configuration**: `models.yaml` only defines model-to-URL mappings
- **Unified key management**: Use ccproxy's `cc-proxy.json` config for all provider keys

### 🎯 Model Selection
- Model choice comes from the incoming request (ccproxy controls this)
- Support for `openai`, `anthropic`, and `gemini` formats in `models.yaml` (with `direct` as a legacy alias)
- Selection is by `model_id` (unique key). You can map multiple `model_id` entries to the same upstream `model_name` with different per-model settings (e.g., `extra_body`, `reasoning_effort`) to expose "reasoning level" variants. `reasoning_effort` supports `minimal|low|medium|high` (where `minimal` means no thinking).

Example:

```yaml
- model_id: doubao-seed-1-8-nothinking
  model_name: doubao-seed-1-8-251228
  api_base: "https://ark.cn-beijing.volces.com/api/v3"
  reasoning_effort: "minimal"

- model_id: doubao-seed-1-8-high
  model_name: doubao-seed-1-8-251228
  api_base: "https://ark.cn-beijing.volces.com/api/v3"
  reasoning_effort: "high"
```

Then select the variant by setting `model` in your ccproxy provider config (or switching providers):

```json
{
  "doubao-high": {
    "base_url": "http://127.0.0.1:8082",
    "model": "doubao-seed-1-8-high",
    "auth_key": "YOUR_KEY"
  }
}
```

### 🔧 Enhanced Error Handling
- Structured error parsing for both OpenAI and Claude API responses
- Detailed logging and debugging information for API failures
- Graceful handling of connection timeouts and rate limits
- Enhanced client reliability with automatic retry mechanisms

### 📊 Advanced Features
- **Multi-format endpoints**: Accept requests in Anthropic, OpenAI, or Gemini format
- **Streaming support**: SSE-based streaming with state machine for chunk conversion
- **Usage statistics tracking**: Provider-reported usage tracking via global stats
- **Custom model configuration**: Per-model settings with `reasoning_effort` support
- **Converter factory pattern**: Dynamic format conversion based on model configuration
- **Thinking mode support**: Configurable thinking/reasoning effort per model

### 🔌 Plugin System (Extensibility)
The proxy includes a plugin system that allows you to modify request and response payloads. Plugins are automatically loaded from the `anthropic_proxy/plugins/` directory.

#### Built-in Plugins
- **`filter_tools.py`**: Filters out specific tools from requests before they're sent to providers. This is useful for removing tools that certain providers don't support or that you want to disable for specific use cases.

**Current filter configuration** (in `filter_tools.py`):
```python
filtered_tool_names = ["WebSearch", "NotebookEdit", "NotebookRead"]
```

**Behavior**:
- Removes `WebSearch`, `NotebookEdit`, and `NotebookRead` tools from all requests
- Works with both Claude format (`{"name": "ToolName"}`) and OpenAI format (`{"function": {"name": "ToolName"}}`)
- Logs removed and remaining tools for debugging
- Applied before requests are sent to upstream providers

**Use cases**:
1. **Provider compatibility**: Some providers may not support certain tools
2. **Security/control**: Disable specific tools for certain deployments
3. **Testing**: Isolate tool-related issues

**To modify the filter**:
1. Edit `anthropic_proxy/plugins/filter_tools.py`
2. Update the `filtered_tool_names` list
3. Restart the proxy server

**To disable the plugin**:
1. Rename or remove `filter_tools.py` from the plugins directory
2. Restart the proxy server

#### Creating Custom Plugins
Create a new `.py` file in the `anthropic_proxy/plugins/` directory with one or both of these functions:

```python
def request_hook(payload):
    """Modify request payload before sending to provider."""
    # Your modification logic here
    return payload

def response_hook(payload):
    """Modify response payload before returning to client."""
    # Your modification logic here
    return payload
```

Plugins are loaded automatically at server startup. Both request and response hooks are optional - include only what you need.

## 🔑 Codex Subscription Support

This proxy supports authentication with OpenAI's Codex subscription plan, allowing you to use Codex models (like `codex/gpt-5.3-codex`) directly.

**Model Reference**: [Introducing GPT-5.3 Codex](https://openai.com/index/introducing-gpt-5-3-codex/)

### ⚠️ Important: Use Anthropic Endpoint for Codex

**Codex models MUST be accessed via the `/anthropic/v1/messages` endpoint**, not `/openai/v1/chat/completions`.

```bash
# CORRECT - Use Anthropic endpoint
ANTHROPIC_BASE_URL=http://localhost:8082/anthropic/v1 claude

# INCORRECT - Do NOT use OpenAI endpoint for Codex
# This will result in 403 Cloudflare challenge errors
```

**Why?** The Codex backend requires specific headers (`originator: codex_cli_rs`, `chatgpt-account-id`, `OpenAI-Beta: responses=experimental`) to be injected. These headers are only added when routing through the Anthropic endpoint (`/anthropic/v1/messages`). Using the OpenAI `/chat/completions` endpoint causes the request to hit Cloudflare's bot protection, returning an HTML challenge page instead of a valid API response.

### Alternative: OpenAI Responses API Endpoint

If your agent/tool uses OpenAI's **Responses API** format (instead of Anthropic format), you can use the dedicated `/openai/v1/responses` endpoint:

```bash
# For agents using OpenAI Responses API format
OPENAI_BASE_URL=http://localhost:8082/openai/v1 claude
```

This endpoint:
- Accepts standard OpenAI Responses API requests
- Directly forwards to Codex backend without format conversion
- Supports both streaming (`stream: true`) and non-streaming modes
- Automatically injects required Codex authentication headers

**Note**: This is different from `/openai/v1/chat/completions` - the `/responses` endpoint is specifically designed for Codex subscription access and does not trigger Cloudflare protection.

### 1. Login

Authenticate via your browser:

```bash
anthropic-proxy login --codex
```
This will open a browser window to log in to OpenAI. Once authenticated, your session tokens are saved securely to `~/.config/anthropic-proxy/auth.json` (or your OS equivalent). Tokens are automatically refreshed in the background.

### 2. Available Models

Once logged in, the following models are **automatically available** without any configuration in `models.yaml` (subject to upstream changes). These IDs are prefixed to avoid collisions:

- `codex/gpt-5.3-codex` (High reasoning) - Newest flagship model
- `codex/gpt-5.2-codex` (High reasoning) - Previous flagship model
- `codex/gpt-5.1-codex-max` (High reasoning)
- `codex/gpt-5.1-codex-mini` (Medium reasoning)
- `codex/gpt-5.2` (High reasoning)

You can use these model IDs directly in your Claude Code setup.
If you define the same `model_id` in `models.yaml`, your configuration takes precedence.

### 3. Customizing Codex Models

To customize parameters (like `reasoning_effort` or `max_tokens`) for a Codex model, add an entry to `models.yaml`. **Note:** You must explicitly set `provider: codex` to inherit the subscription authentication.

```yaml
- model_id: codex/gpt-5.2-codex
  provider: codex
  reasoning_effort: high  # Override default
  max_tokens: 32K
```

### 4. Non-Codex Plan Usage

If you have a Codex model ID but want to use it with a standard OpenAI API key (billing) instead of the subscription plan, provide `api_base`/`api_key` and leave `provider` unset:

```yaml
- model_id: codex/gpt-5.2-codex
  model_name: gpt-5.2-codex
  api_base: https://api.openai.com/v1
  api_key: sk-...
```

## 💎 Gemini Subscription Support (Google)

Use your **Gemini Code Assist plan (including the Free Tier)** directly, bypassing separate API billing.

This authenticates with your Google account and connects to the **Code Assist API** (`cloudcode-pa`), consuming your user quota/subscription rather than per-token Cloud API credits.

### 1. Login

```bash
anthropic-proxy login --gemini
```
This authenticates with your Google account. It will automatically resolve a Google Cloud Project context required for the connection (defaulting to the Free tier if available).

### 2. Available Models

The following models are automatically available (subject to upstream changes). These IDs are prefixed to avoid collisions:
- `gemini/gemini-3-pro-preview`
- `gemini/gemini-3-flash-preview`
- `gemini/gemini-2.5-pro`
- `gemini/gemini-2.5-flash`
- `gemini/gemini-2.5-flash-lite`
If you define the same `model_id` in `models.yaml`, your configuration takes precedence.

### 3. Customizing Gemini Models

To override settings, specify `provider: gemini`:

```yaml
- model_id: gemini/gemini-2.5-flash
  provider: gemini
  reasoning_effort: medium  # if applicable
```

## 🛰️ Antigravity Subscription Support (Google Internal)

Use Google's internal Antigravity service (Cloud Code backend) with your Google account.

### 1. Login

```bash
anthropic-proxy login --antigravity
```

### 2. Available Models

The following models are automatically available (subject to upstream changes). These IDs are prefixed to avoid collisions:
- `antigravity/claude-opus-4-5-thinking`
- `antigravity/claude-sonnet-4-5`
- `antigravity/claude-sonnet-4-5-thinking`
- `antigravity/gemini-3-pro`
- `antigravity/gemini-3-pro-low`
- `antigravity/gemini-3-pro-high`
- `antigravity/gemini-3-pro-preview`
- `antigravity/gemini-3-flash`
- `antigravity/gemini-2.5-pro`
- `antigravity/gemini-2.5-flash`
- `antigravity/gemini-2.5-flash-lite`
- `antigravity/gemini-2.5-flash-thinking`
- `antigravity/gemini-2.0-flash-exp`
- `antigravity/gemini-3-pro-image`
If you define the same `model_id` in `models.yaml`, your configuration takes precedence.

### 3. Customizing Antigravity Models

To override settings, specify `provider: antigravity`:

```yaml
- model_id: antigravity/claude-sonnet-4-5
  provider: antigravity
  reasoning_effort: high
```

## 🤖 Claude Code Subscription Support

Use Claude Code subscription (setup-token based) for access to Claude Opus 4.6, Sonnet 4.5, and Haiku 4.5 models.

**Model Reference**: [Claude Models Documentation](https://platform.claude.com/docs/en/about-claude/models/overview)

### 1. Login

```bash
anthropic-proxy login --claude-code
```

This will prompt you for a setup token from Claude Code. Get your token by running:

```bash
claude setup-token
```

Enter the token when prompted. Unlike OAuth providers, this token is permanent until revoked - no refresh needed.

### 2. Available Models

The following models are automatically available (subject to upstream changes). These IDs are prefixed to avoid collisions:

| Model ID | Max Output | Context Window | Description |
|----------|------------|----------------|-------------|
| `claude-code/claude-opus-4-6` | 128K | 200K (1M with env var) | Most capable model for complex tasks |
| `claude-code/claude-opus-4-5` | 64K | 200K | Claude Opus 4.5 (backward compatibility) |
| `claude-code/claude-opus-4-5-20251101` | 64K | 200K | Claude Opus 4.5 (dated snapshot) |
| `claude-code/claude-sonnet-4-5` | 64K | 200K (1M with env var) | Balanced performance and speed |
| `claude-code/claude-sonnet-4-5-20250929` | 64K | 200K | Claude Sonnet 4.5 (dated snapshot) |
| `claude-code/claude-haiku-4-5` | 64K | 200K | Fast, lightweight model |
| `claude-code/claude-haiku-4-5-20251001` | 64K | 200K | Claude Haiku 4.5 (dated snapshot) |

**1M Context Window**: Opus 4.6 and Sonnet 4.5 support extended 1M token context via environment variable:

```bash
CLAUDE_CODE_1M_CONTEXT=1 anthropic-proxy start
```

**Prompt Caching**: Automatically enabled with system prompt + last user message cached. Set `CLAUDE_CODE_CACHE_RETENTION=long` for 1-hour TTL (default is 5 minutes).

### 3. Customizing Claude Code Models

To override settings, specify `provider: claude-code`:

```yaml
- model_id: claude-code/claude-opus-4-6
  provider: claude-code
  max_tokens: 64k
```

If you define the same `model_id` in `models.yaml`, your configuration takes precedence.

## Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended)
- API keys for desired providers

### Installation

#### Option 1: UV Tool Installation (Recommended)

Install as a global tool using `uv`:

```bash
git clone https://github.com/tizee/anthropic-proxy.git
cd anthropic-proxy
uv tool install .
```

After installation, the `anthropic-proxy` command is available globally:

```bash
anthropic-proxy                    # Start with default config
anthropic-proxy --help             # Show all options
anthropic-proxy --print-config     # View current configuration
```

**Uninstall:**

```bash
uv tool uninstall anthropic-proxy
```

**Upgrade:**

```bash
cd anthropic-proxy
git pull
uv tool install anthropic-proxy --reinstall
```

#### Option 2: Local Installation

Traditional local installation using the existing scripts:

```bash
git clone https://github.com/tizee/anthropic-proxy.git
cd anthropic-proxy
./install.sh
```

Or use `uv run` for development:

```bash
cd anthropic-proxy
uv install
make run
```

### Configuration

Configuration files are stored in `~/.config/anthropic-proxy/`:

```
~/.config/anthropic-proxy/
├── models.yaml      # Model configurations
└── config.json      # Server settings (log level, port, env vars)

~/.anthropic-proxy/
├── anthropic-proxy.pid  # PID file for tracking running daemon
├── daemon.log           # Daemon stdout/stderr output
└── server.log           # Application logs (configured in config.json)
```

**Log Files:**
- **`daemon.log`**: Captures all stdout/stderr output from the daemon process, including:
  - Server startup messages
  - Configuration loading status
  - Python exceptions and tracebacks
  - Uvicorn server output
  - FastAPI access logs (request/response)
- **`server.log`**: Application-level logs configured by `log_level` in `config.json` (default: WARNING)

**First Run**: On first run, config files are auto-created with default values. Use `--init` to reinitialize (skips if files exist):

```bash
anthropic-proxy --init           # Initialize (skips if files exist)
anthropic-proxy --init-force     # Force reinitialize (overwrites existing files)
```

#### models.yaml

Configure your models in `~/.config/anthropic-proxy/models.yaml`:

```yaml
# Required fields: model_id, api_base
# Optional fields: model_name, api_key, can_stream, max_tokens, max_input_tokens,
#                  context, extra_headers, extra_body, format, direct, reasoning_effort, temperature
#
# API Key Configuration:
# - api_key: Optional per-model API key. If set, it takes precedence over request headers.
# - If api_key is not set, the key from the Authorization header (via ccproxy) is used.
# - API keys stored here are in plain text - use caution in shared environments.
#
# reasoning_effort supports: minimal, low, medium, high (minimal = no thinking)
# format: openai | anthropic | gemini (routing format; defaults to openai)
# direct: legacy alias (direct: true -> format=anthropic). If format is set, it wins.

- model_id: deepseek-chat
  model_name: deepseek-chat
  api_base: https://api.deepseek.com/v1
  format: openai
  can_stream: true
  max_tokens: 8K
  context: 128K

- model_id: claude-3-5-sonnet-direct
  model_name: claude-3-5-sonnet-20241022
  api_base: https://api.anthropic.com
  format: anthropic
  can_stream: true
  max_tokens: 8K
  max_input_tokens: 200K
```

#### config.json

Server settings in `~/.config/anthropic-proxy/config.json`:

```json
{
  "log_level": "WARNING",
  "log_file_path": "~/.anthropic-proxy/server.log",
  "host": "0.0.0.0",
  "port": 8082
}
```

### CLI Options

#### Server Control Commands

```bash
anthropic-proxy start               # Start server in background
anthropic-proxy start --port 8080   # Start on custom port
anthropic-proxy stop                # Stop running server
anthropic-proxy restart             # Restart server
anthropic-proxy status              # Show server status (PID, port, uptime)
```

#### Utility Commands

```bash
anthropic-proxy --print-config                # View configuration (API keys redacted)
anthropic-proxy --print-config --show-api-keys  # View with API keys visible
anthropic-proxy --init                         # Initialize config (skips if files exist)
anthropic-proxy --init-force                   # Force reinitialize (overwrites existing files)
anthropic-proxy provider --list                # List auth providers and OAuth status
anthropic-proxy provider --models              # List available model IDs (custom + provider defaults)
anthropic-proxy --models PATH                  # Use custom models file
anthropic-proxy --config PATH                  # Use custom config file
```

#### Global Options

Global options like `--models` and `--config` can be combined with server control commands:

```bash
anthropic-proxy start --models ./my-models.yaml --port 8080
anthropic-proxy restart --config ./my-config.json
```

### Running the Server

The server runs as a background daemon. All output is logged to `~/.anthropic-proxy/daemon.log`.

```bash
# Start with default config
anthropic-proxy start

# Start on custom port
anthropic-proxy start --port 8080

# Check status (shows PID, port, uptime, log file location)
anthropic-proxy status

# Stop the server
anthropic-proxy stop

# Restart the server
anthropic-proxy restart

# View logs for debugging
tail -f ~/.anthropic-proxy/daemon.log
```

### Connecting Claude Code

For Claude Code, use the Anthropic-format endpoint:

```bash
ANTHROPIC_BASE_URL=http://localhost:8082/anthropic/v1 claude
```

**Important Notes**:
- The `/anthropic/v1` prefix is **required** for all requests, including Codex, Gemini, and Antigravity models
- **Do NOT use `/openai/v1`** for Codex models - this will result in 403 Cloudflare errors
- The proxy automatically injects provider-specific headers (like `originator: codex_cli_rs` for Codex) only when using the Anthropic endpoint

The proxy also supports these endpoints for other use cases:
- OpenAI format: `http://localhost:8082/openai/v1` (for standard OpenAI-compatible providers, NOT Codex)
- Gemini format: `http://localhost:8082/gemini/v1beta` (for Gemini SDK clients)

## Debugging & Logging

### Log Files Overview

The proxy uses two separate log files for different purposes:

| File | Location | Purpose | Content |
|------|----------|---------|---------|
| **daemon.log** | `~/.anthropic-proxy/daemon.log` | Daemon process output | Server startup, configuration, exceptions, uvicorn/fastapi logs |
| **server.log** | `~/.anthropic-proxy/server.log` | Application logs | Configured by `log_level` in config.json (default: WARNING) |

### Viewing Logs

```bash
# Follow daemon log in real-time (recommended for debugging)
tail -f ~/.anthropic-proxy/daemon.log

# View last 50 lines of daemon log
tail -n 50 ~/.anthropic-proxy/daemon.log

# Search for errors in daemon log
grep -i "error\|exception\|traceback" ~/.anthropic-proxy/daemon.log

# View server log (application-level)
cat ~/.anthropic-proxy/server.log
```

### Common Debugging Scenarios

**Codex returning 403 / Cloudflare challenge page:**
```bash
# If you see HTML responses with "Enable JavaScript and cookies to continue"
# in the logs, you're using the wrong endpoint.

# SOLUTION 1: Use Anthropic endpoint (for Anthropic-format agents)
ANTHROPIC_BASE_URL=http://localhost:8082/anthropic/v1 claude

# SOLUTION 2: Use OpenAI Responses API endpoint (for OpenAI-format agents)
OPENAI_BASE_URL=http://localhost:8082/openai/v1 claude

# INCORRECT: /chat/completions endpoint doesn't work with Codex
# ANTHROPIC_BASE_URL=http://localhost:8082/openai/v1/chat/completions  # DON'T DO THIS
```
**Why this happens:** The Codex backend requires specific authentication headers (`originator`, `chatgpt-account-id`, `OpenAI-Beta`) that are only injected when using the `/anthropic/v1/messages` endpoint or the `/openai/v1/responses` endpoint. The `/chat/completions` endpoint doesn't add these headers, causing Cloudflare to block the request.

**Server won't start:**
```bash
# Check daemon log for startup errors
cat ~/.anthropic-proxy/daemon.log

# Verify no other process is using the port
lsof -i :8082

# Check if daemon is already running
anthropic-proxy status
```

**Requests failing:**
```bash
# Follow daemon log to see request/response flow
tail -f ~/.anthropic-proxy/daemon.log

# Increase log verbosity in config.json:
# "log_level": "DEBUG"
```

**Configuration issues:**
```bash
# View current configuration
anthropic-proxy --print-config

# Reinitialize config files
anthropic-proxy --init-force
```

### Log Level Configuration

Adjust logging verbosity in `~/.config/anthropic-proxy/config.json`:

```json
{
  "log_level": "DEBUG"   // Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
}
```

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General informational messages
- **WARNING**: Something unexpected happened (default)
- **ERROR**: Serious problem occurred
- **CRITICAL**: Critical error, program may not continue

After changing `log_level`, restart the server:
```bash
anthropic-proxy restart
```

## Development

For detailed information on the architecture, features, and testing of this project, please refer to the documents in the `docs/` directory:

- **[Architecture](./docs/architecture.md)**: A high-level overview of the proxy's architecture.
- **[Features](./docs/features.md)**: A description of the key features of the proxy.
- **[Testing](./docs/testing.md)**: Instructions on how to run the unit and performance tests.
- **[API Response Formats](./docs/api-response-formats.md)**: Reference documentation for API response formats used by different providers.

### Core Module Structure

- **`anthropic_proxy/server.py`**: FastAPI endpoints with prefixed paths (`/anthropic/v1/...`, `/openai/v1/...`, `/gemini/v1beta/...`)
- **`anthropic_proxy/client.py`**: Loads `models.yaml` and creates provider-specific clients
- **`anthropic_proxy/converters/`**: Unified converter package with base classes and format-specific converters
  - `base.py`: Base converter classes (`BaseConverter`, `BaseStreamingConverter`)
  - `anthropic.py`: Anthropic format converter (identity pass-through)
  - `openai.py`: OpenAI format converter
  - `gemini.py`: Gemini format converter
  - `_openai_impl.py`: Legacy OpenAI implementation functions
  - `_gemini_impl.py`: Gemini implementation functions
  - `_gemini_streaming.py`: Gemini streaming implementation
- **`anthropic_proxy/converter.py`**: Converter facade (re-exports format-specific converters)
- **`anthropic_proxy/streaming.py`**: Streaming conversion facade
- **`anthropic_proxy/types.py`**: Pydantic models and API schemas
- **`anthropic_proxy/utils.py`**: Usage tracking and error helpers

Additionally, the `CLAUDE.md` file provides guidance for both developers and AI assistants working with this project:

- **For Developers**: Helps understand the codebase structure, design patterns, and key commands.
- **For AI Assistants**: Contains specific instructions to help AI tools effectively navigate and modify the codebase.

Reading both the documentation in `docs/` and `CLAUDE.md` will give you a comprehensive understanding of the project.

## Credit & Acknowledgment

This project is forked from and based on [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) by [@1rgs](https://github.com/1rgs).
