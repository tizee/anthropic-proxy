# Anthropic API Proxy for Claude Code 🔄

[![GitHub latest commit](https://img.shields.io/github/last-commit/tizee/anthropic-proxy)](https://github.com/tizee/anthropic-proxy)
[![License](https://img.shields.io/github/license/tizee/anthropic-proxy)](https://github.com/tizee/anthropic-proxy/blob/main/LICENSE)

A proxy server that enables Claude Code to work with multiple model providers through two modes:

1. **OpenAI-Compatible Mode**: Translates Anthropic API requests to OpenAI-compatible endpoints
2. **Direct Claude API Mode**: Routes requests directly to official Claude API or compatible endpoints

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
| 1.0.35  | ✅ Tested | Current supported version |
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

## Key Features

### 🔄 Dual-Mode Operation
- **OpenAI-Compatible Mode**: Convert Anthropic API requests to OpenAI format for third-party providers
- **Direct Claude API Mode**: Route requests directly to official Anthropic API with native format preservation

### 🧠 Intelligent Routing
- Automatic model selection based on request characteristics (token count, thinking mode, etc.)
- Configurable fallback to long-context models when token limits are exceeded
- Support for both direct and OpenAI-compatible models in routing configuration

### 🔧 Enhanced Error Handling
- Structured error parsing for both OpenAI and Claude API responses
- Detailed logging and debugging information for API failures
- Graceful handling of connection timeouts and rate limits

### 📊 Advanced Features
- Streaming support for both modes with proper error handling
- Token counting and usage statistics tracking
- Custom model configuration with per-model settings
- Support for thinking mode and reasoning effort parameters

## Quick Start

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended)
- [make](https://www.gnu.org/software/make/) (optional but recommended)
- API keys for desired providers

### Installation
```bash
git clone https://github.com/tizee/claude-code-proxy.git
cd claude-code-proxy
uv install
```

### Configuration
1. Copy `.env.example` to `.env`
2. Set your API keys for the desired providers
3. Configure models in `models.yaml` with appropriate mode settings

#### Direct Claude API Mode Configuration
To use official Claude API or compatible endpoints directly:

```yaml
models:
  claude-3-5-sonnet-direct:
    model_name: claude-3-5-sonnet-20241022
    api_base: https://api.anthropic.com
    api_key_name: ANTHROPIC_API_KEY
    direct: true  # Enable direct Claude API mode
    max_tokens: 8k   # Supports shorthand: 8k, 16k, 32k, etc.
    max_input_tokens: 200k
```

#### OpenAI-Compatible Mode Configuration
For OpenAI-compatible endpoints:

```yaml
models:
  deepseek-v3:
    model_name: deepseek-chat
    api_base: https://api.deepseek.com/v1
    api_key_name: DEEPSEEK_API_KEY
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

## Credit & Acknowledgment

This project is forked from and based on [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) by [@1rgs](https://github.com/1rgs). The intelligent routing feature was inspired by [claude-code-router](https://github.com/musistudio/claude-code-router).
