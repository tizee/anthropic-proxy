# Anthropic API Proxy for Claude Code 🔄

[![GitHub latest commit](https://img.shields.io/github/last-commit/tizee/anthropic-proxy)](https://github.com/tizee/anthropic-proxy)
[![License](https://img.shields.io/github/license/tizee/anthropic-proxy)](https://github.com/tizee/anthropic-proxy/blob/main/LICENSE)

A proxy server that enables Claude Code to work with multiple model providers through two modes:

1. **OpenAI-Compatible Mode**: Translates Anthropic API requests to OpenAI-compatible endpoints
2. **Direct Claude API Mode**: Routes requests directly to official Claude API or compatible endpoints

- kimi official supports Anthropic API https://api.moonshot.cn/anthropic
- deepseek supports Anthropic https://api-docs.deepseek.com/guides/anthropic_api

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
| 1.0.51 | ✅ Tested | Current supported version |
| 1.0.x   | ⚠️ Likely Compatible | Earlier versions may work but are untested |

## Tested Models

### OpenAI-Compatible Mode
| Provider | Model | Format | Status | Notes |
|----------|-------|--------|--------|-------|
| **DeepSeek** | deepseek-v3 | OpenAI | ✅ Fully Tested | Recommended for background tasks |
| **DeepSeek** | deepseek-r1 | OpenAI | ✅ Fully Tested | Optimized for thinking/reasoning |
| **OpenRouter** | claude models | OpenAI | ✅ Fully Tested | Claude via OpenAI format |
| **ByteDance** | doubao-seed-1.6 | OpenAI | ✅ Fully Tested | Doubao model support |
| **OpenRouter** | Gemini-2.5-pro | OpenAI | ✅ Fully Tested | Google Gemini via OpenRouter |
| **OpenRouter** | gemini-2.5-flash-lite-preview-06-17 | OpenAI | ✅ Fully Tested | Gemini preview via OpenRouter |

### Direct Claude API Mode
| Provider | Model | Format | Status | Notes |
|----------|-------|--------|--------|-------|
| **Anthropic** | claude-3-5-sonnet-20241022 | Direct | ✅ Fully Tested | Official Claude API |
| **Anthropic** | claude-3-5-haiku-20241022 | Direct | ✅ Fully Tested | Official Claude API |
| **Anthropic** | claude-3-opus-20240229 | Direct | ✅ Fully Tested | Official Claude API |

## Third-Party Providers Supporting Anthropic Format

The following providers offer native Anthropic API compatibility, allowing direct usage without format conversion:

| Provider | API Endpoint | Supported Models | Status | Notes |
|----------|-------------|------------------|---------|-------|
| **Moonshot AI (Kimi)** | `https://api.moonshot.cn/anthropic` | kimi-k2-0711-preview, kimi-k2-0711-chat | ✅ Tested | Native Anthropic API compatibility |
| **DeepSeek** | `https://api.deepseek.com/anthropic` | deepseek-chat, deepseek-reasoner | ✅ Tested | Full Anthropic API compatibility |

### Usage Notes

- **✅ Tested**: Provider has been tested and confirmed to work with Anthropic API compatibility

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

## Known Issues

### Provider-Specific Issues

**⚠️ Groq Provider - Tool Call Limitations (as of January 2025)**

The Groq provider has known issues with tool call functionality:
- **Multiple tool calls fail**: When Claude Code requests multiple tools in a single interaction, Groq may not handle them correctly
- **Tool call generation instability**: Even single tool calls may fail intermittently
- **Confirmed across implementations**: This issue affects both JavaScript and Python implementations, confirming it's a provider-side problem

**Root Cause**: This is acknowledged by the Kimi model team on Twitter, where they mentioned bugs in the Kimi-K2-Instruct model's tool call handling, specifically for multi-turn tool calls.

**Discussion Links**:
- **GitHub Issue**: [sst/opencode#1018](https://github.com/sst/opencode/issues/1018#issuecomment-3075860647) - Contains detailed discussion about tool use errors with Groq's Kimi K2
- **Groq Community**: [Tool call issues discussion](https://community.groq.com/discussion-forum-7/groq-kimi-k2-tool-call-issues-213) - Official Groq community thread on Kimi K2 tool call problems

**Workaround**: Use alternative providers like:
- **Moonshot AI** (kimi-k2-0711-preview via direct API)
- **Google Gemini** (via OpenRouter)
- **DeepSeek** models
- **OpenRouter** with other models

**Future**: This issue may be resolved by Groq in future updates, but currently requires using alternative providers for reliable tool call functionality.

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
claude-proxy -d    # Development mode
claude-proxy --help

# Removal
./uninstall.sh
```

## Credit & Acknowledgment

This project is forked from and based on [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) by [@1rgs](https://github.com/1rgs).
