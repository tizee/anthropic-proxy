# Anthropic API Proxy for Claude Code 🔄

[![GitHub latest commit](https://img.shields.io/github/last-commit/tizee/anthropic-proxy)](https://github.com/tizee/anthropic-proxy)
[![License](https://img.shields.io/github/license/tizee/anthropic-proxy)](https://github.com/tizee/anthropic-proxy/blob/main/LICENSE)

A proxy server that enables Claude Code to work with multiple model providers through two modes:

1. **OpenAI-Compatible Mode**: Translates Anthropic API requests to OpenAI-compatible endpoints
2. **Direct Claude API Mode**: Routes requests directly to official Claude API or compatible endpoints

- kimi official supports Anthropic API https://api.moonshot.cn/anthropic
- deepseek supports Anthropic https://api-docs.deepseek.com/guides/anthropic_api
- Zhipu GLM supports Anthropic API https://open.bigmodel.cn/api/anthropic
- MiniMax supports Anthropic API https://api.minimax.io/anthropic

This allows you to use Claude Code with both OpenAI-compatible models and native Claude API endpoints. For third-party models to support Claude Code image files (URL/base64), they must natively support multimodal image understanding.

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

The following providers offer native Anthropic API compatibility, allowing direct usage without format conversion:

| Provider | API Endpoint | Supported Models |
|----------|-------------|------------------|
| **Moonshot AI (Kimi)** | `https://api.moonshot.cn/anthropic` | kimi-k2-0711-preview, kimi-k2-0711-chat |
| **DeepSeek** | `https://api.deepseek.com/anthropic` | deepseek-chat, deepseek-reasoner |
| **Zhipu AI (GLM)** | `https://open.bigmodel.cn/api/anthropic` | glm-4.7, glm-4.6 |
| **MiniMax** | `https://api.minimax.io/anthropic` | MiniMax-M2.1 |


### Configuration Example

For providers with native Anthropic support, configure as direct mode:

```yaml
- model_id: kimi-k2-direct
  model_name: kimi-k2-0711-preview
  api_base: https://api.moonshot.cn/anthropic
  direct: true
  max_tokens: 16k
  max_input_tokens: 200k
```

## Key Features

### 🔄 Dual-Mode Operation
- **OpenAI-Compatible Mode**: Convert Anthropic API requests to OpenAI format for third-party providers
- **Direct Claude API Mode**: Route requests directly to official Anthropic API with native format preservation

### 🔗 ccproxy Integration (Recommended)
This proxy is designed to work seamlessly with **ccproxy** (Claude Code wrapper script):
- **API keys managed by ccproxy**: No need to store keys in the proxy's config - they're passed via request headers
- **Simplified configuration**: `models.yaml` only defines model-to-URL mappings
- **Unified key management**: Use ccproxy's `cc-proxy.json` config for all provider keys

### 🎯 Model Selection
- Model choice comes from the incoming request (ccproxy controls this)
- Support for both direct and OpenAI-compatible models in `models.yaml`
- Selection is by `model_id` (unique key). You can map multiple `model_id` entries to the same upstream `model_name` with different per-model settings (e.g., `extra_body`, `reasoning_effort`) to expose “reasoning level” variants. `reasoning_effort` supports `minimal|low|medium|high` (where `minimal` means no thinking).

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
- Enhanced client reliability with automatic retry mechanisms. Configure via `MAX_RETRIES` environment variable (default: 2 retries)

### 📊 Advanced Features
- Streaming support for both modes with proper error handling
- Usage statistics tracking (from provider-reported usage)
- Custom model configuration with per-model settings
- Support for thinking mode and reasoning effort parameters

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

## Quick Start

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended)
- [make](https://www.gnu.org/software/make/) (optional but recommended)
- API keys for desired providers

### Installation

#### Option 1: Global Installation (Recommended)
Install once and use from any directory:

```bash
git clone https://github.com/tizee/claude-code-proxy.git
cd claude-code-proxy
./install.sh
```

After installation, you can run the proxy from anywhere:
```bash
claude-proxy        # Production mode
claude-proxy -d     # Development mode
claude-proxy -p 8080  # Custom port
```

#### Option 2: Local Installation
Traditional local installation:

```bash
git clone https://github.com/tizee/claude-code-proxy.git
cd claude-code-proxy
uv install
```

### Configuration

#### ccproxy Integration Mode (Recommended)

When using with ccproxy, API keys are passed via request headers automatically:

1. Configure models in `models.yaml` - only URL mappings needed, no API keys
2. Configure your API keys in ccproxy's `~/.config/llm/cc-proxy.json`
3. ccproxy sets `ANTHROPIC_BASE_URL` to point to this proxy and passes keys via headers

```yaml
# models.yaml - simplified configuration (no api_key_name needed)
- model_id: deepseek-chat
  model_name: deepseek-chat
  api_base: https://api.deepseek.com/v1
  can_stream: true
  max_tokens: 8k
  context: 64k

- model_id: claude-sonnet-4
  model_name: anthropic/claude-sonnet-4
  api_base: https://openrouter.ai/api/v1
  can_stream: true
  max_tokens: 64000
  context: 200k
```

#### Direct Claude API Mode Configuration
To use official Claude API or compatible endpoints directly:

```yaml
- model_id: claude-3-5-sonnet-direct
  model_name: claude-3-5-sonnet-20241022
  api_base: https://api.anthropic.com
  direct: true  # Enable direct Claude API mode
  max_tokens: 8k   # Supports shorthand: 8k, 16k, 32k, etc.
  max_input_tokens: 200k
```

#### OpenAI-Compatible Mode Configuration
For OpenAI-compatible endpoints:

```yaml
- model_id: deepseek-v3
  model_name: deepseek-chat
  api_base: https://api.deepseek.com/v1
  direct: false  # Use OpenAI-compatible mode (default)
  max_tokens: 8k   # Supports shorthand notation
  max_input_tokens: 128k
```

**Note**: Token limits support both numeric values (e.g., `8192`) and shorthand notation (e.g., `8k`, `16K`, `32k`, `128K`, `200k`). The shorthand format is case-insensitive.

### Running the Server
```bash
make run
```
To run in development mode with auto-reload:
```bash
make dev
```

### Connecting Claude Code
```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

## Development

For detailed information on the architecture, features, and testing of this project, please refer to the documents in the `docs/` directory:

- **[Architecture](./docs/architecture.md)**: A high-level overview of the proxy's architecture.
- **[Features](./docs/features.md)**: A description of the key features of the proxy.
- **[Testing](./docs/testing.md)**: Instructions on how to run the unit and performance tests.
- **[API Response Formats](./docs/api-response-formats.md)**: Reference documentation for API response formats used by different providers.

Additionally, the `CLAUDE.md` file provides guidance for both developers and AI assistants working with this project:

- **For Developers**: Helps understand the codebase structure, design patterns, and key commands.
- **For AI Assistants**: Contains specific instructions to help AI tools effectively navigate and modify the codebase.

Reading both the documentation in `docs/` and `CLAUDE.md` will give you a comprehensive understanding of the project.

## Scripts

This repository includes convenient installation and management scripts:

- **`./install.sh`**: Installs the proxy globally so you can run `claude-proxy` from any directory
- **`./uninstall.sh`**: Removes the global installation
- **`claude-proxy`**: Global command after installation (see [Installation](#installation))

Example usage:
```bash
# Installation
./install.sh

# Usage
claude-proxy start
claude-proxy --help

# Removal
./uninstall.sh
```

## Credit & Acknowledgment

This project is forked from and based on [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) by [@1rgs](https://github.com/1rgs).
