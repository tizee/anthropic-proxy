# Anthropic API Proxy for Multiple Model Providers 🔄

**Claude COde Version: 1.0.31**

A proxy server that translates Anthropic API requests to multiple model providers (OpenAI, Gemini, custom) using LiteLLM. Features intelligent routing based on token count, thinking flag, and model name.

## Key Features
- 🚀 Intelligent model routing (background/think/long-context modes)
- 🔄 Supports OpenAI, Gemini, and custom OpenAI-compatible models
- 🛠️ Full tool use (function calling) support
- 📊 Token counting and cost tracking
- 🌉 Seamless Claude API compatibility

## Quick Start

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (recommended)
- API keys for desired providers

### Installation
```bash
git clone https://github.com/tizee/claude-code-proxy.git
cd claude-code-proxy
uv install
```

### Configuration
1. Copy `.env.example` to `.env`
2. Set your API keys:
   - `OPENAI_API_KEY` for OpenAI models
   - `GEMINI_API_KEY` for Gemini models
   - `ANTHROPIC_API_KEY` (optional for proxying to Anthropic)

### Router Configuration
Configure model routing in `.env`:
```
ROUTER_BACKGROUND="custom/deepseek-v3-250324"  # Default for haiku
ROUTER_THINK="custom/deepseek-r1-250528"       # Thinking tasks
ROUTER_LONG_CONTEXT="custom/gemini-2.5-pro"    # Long context
ROUTER_DEFAULT="custom/deepseek-v3-250324"     # Fallback
```

### Running the Server
```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

### Connecting Claude Code
```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

## Custom Models
Add custom OpenAI-compatible models in `custom_models.yaml`:
```yaml
- model_id: my-model
  api_base: https://api.example.com
  api_key_name: MY_API_KEY
  can_stream: true
  max_tokens: 8192
```

## Debugging
Set `LOG_LEVEL=DEBUG` in `.env` for detailed logs.

## Architecture
1. Receives Anthropic API requests
2. Routes to appropriate provider based on:
   - Token count
   - Thinking flag
   - Model name
3. Translates responses back to Anthropic format

## Supported Models
| Provider | Example Models |
|----------|----------------|
| OpenAI   | gpt-4o, gpt-4o-mini |
| Gemini   | gemini-2.5-pro, gemini-2.0-flash |
| Custom   | Any OpenAI-compatible API |
