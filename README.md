# Anthropic API Proxy for Claude Code 🔄

A proxy server that translates Anthropic API requests to multiple model providers (OpenAI, Gemini, custom) using native OpenAI SDK. This allows you to use Claude Code with a wide range of OpenAI-compatible models.

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

| Provider | Model | Format | Status | Notes |
|----------|-------|--------|--------|-------|
| **DeepSeek** | deepseek-v3 | OpenAI | ✅ Fully Tested | Recommended for background tasks |
| **DeepSeek** | deepseek-r1 | OpenAI | ✅ Fully Tested | Optimized for thinking/reasoning |
| **OpenRouter** | claude models | OpenAI | ✅ Fully Tested | Claude via OpenAI format |
| **ByteDance** | doubao-seed-1.6 | OpenAI | ✅ Fully Tested | Doubao model support |
| **OpenRouter** | Gemini-2.5-pro | OpenAI | ✅ Fully Tested | Google Gemini via OpenRouter |
| **Google** | gemini-2.5-pro | OpenAI | ✅ Fully Tested | Direct Google API |
| **Google** | gemini-2.5-flash-lite-preview-06-17 | OpenAI | ✅ Fully Tested | Preview model |

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
2. Set your API keys for the desired providers.

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
