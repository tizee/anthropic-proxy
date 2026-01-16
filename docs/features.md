# Features

This document describes the key features of the proxy.

## Multi-Format Endpoints

The proxy provides three endpoint formats to accept requests in different API styles:

- **Anthropic Format** (`/anthropic/v1/...`): Accepts Anthropic Messages API format
- **OpenAI Format** (`/openai/v1/...`): Accepts OpenAI Chat Completions API format
- **Gemini Format** (`/gemini/v1beta/...`): Accepts Gemini GenerateContent API format

Each endpoint accepts requests in its respective format, internally routes based on the `format` field in `models.yaml`, and returns responses in the same format as the request. This enables seamless integration with clients that expect different API formats.

## Format-Based Routing

The proxy routes requests based on the `format` field in `models.yaml`:

- **Anthropic format** (`format: anthropic` or legacy `direct: true`): Routes to Anthropic-compatible APIs without format conversion. Providers include Moonshot AI (Kimi), DeepSeek, Zhipu GLM, MiniMax, VolcEngine. Uses httpx client.

- **OpenAI format** (`format: openai`): Converts Anthropic format to OpenAI format for third-party providers. Uses AsyncOpenAI SDK client.

- **Gemini format** (`format: gemini`): Routes via Gemini Code Assist subscriptions (Gemini/Antigravity).

## Converter Factory Pattern

The proxy uses a unified converter package with base classes and factory functions:

- **Base Classes**: `BaseConverter` and `BaseStreamingConverter` provide consistent interfaces for all format converters
- **Format-Specific Converters**: Separate implementations for Anthropic, OpenAI, and Gemini formats
- **Factory Functions**: `get_converter()` and `get_streaming_converter()` for dynamic converter selection based on model configuration
- **Legacy Support**: `_openai_impl.py` and `_gemini_impl.py` provide backward compatibility with existing code

## Model Selection

The proxy does not switch models on its own. The incoming request specifies the model, and the proxy maps that model to a provider entry in `models.yaml`. This keeps selection logic in ccproxy or the caller.

### reasoning_effort

Controls thinking mode behavior. Supports: `minimal`, `low`, `medium`, `high`
- `minimal`: No thinking (thinking block disabled)
- `low`/`medium`/`high`: Progressive thinking intensity

This allows creating "reasoning level" variants of the same base model by mapping multiple `model_id` entries to the same `model_name` with different `reasoning_effort` settings.

## Streaming Support

The proxy fully supports streaming responses from the underlying models:

- **State Machine**: Streaming converters use a state machine to handle SSE chunk conversion
- **Format-Specific Streaming**: Separate streaming converters for OpenAI (`OpenAIToAnthropicStreamingConverter`) and Gemini (`GeminiStreamingConverter`)
- **Content Handling**: Correctly handles content blocks, tool calls, and thinking blocks during streaming
- **Token Tracking**: Tracks token counts from provider responses (not local estimation)
- **Hook Integration**: Response hooks are applied to each streaming event via `hook_streaming_response()`

## Auth Provider Support

The proxy supports OAuth-based authentication for several provider subscriptions:

- **Codex**: OpenAI's Codex subscription plan with automatic token refresh
- **Gemini**: Google's Gemini Code Assist (including Free Tier) with project resolution
- **Antigravity**: Google's internal Antigravity service with signature caching for Claude thinking models

Default auth-provider models are prefixed (`codex/`, `gemini/`, `antigravity/`) to avoid collisions with user-defined models.

## Custom Model Support

The proxy allows you to use custom models from various providers. You can define your own models in the `models.yaml` file with per-model settings including:
- `api_base`: Provider API endpoint
- `model_name`: Actual model name sent to provider
- `format`: Request format (anthropic/openai/gemini)
- `reasoning_effort`: Thinking mode intensity
- `max_tokens`, `max_input_tokens`: Token limits
- `temperature`: Sampling temperature
- `extra_headers`, `extra_body`: Additional request parameters

## Usage Tracking

The proxy records usage statistics from provider responses via the `/anthropic/v1/stats` endpoint. It tracks token counts from provider responses rather than performing local estimation.

## Plugin System

The proxy includes a plugin system that allows you to modify request and response payloads:

- **Dynamic Loading**: Plugins are automatically loaded from `anthropic_proxy/plugins/`
- **Request Hooks**: Modify requests before they're sent to providers via `request_hook(payload)`
- **Response Hooks**: Modify responses before they're returned to clients via `response_hook(payload)`
- **Built-in Plugins**: `filter_tools.py` filters WebSearch, NotebookEdit, and NotebookRead tools

## Error Handling

The proxy includes robust error handling to gracefully manage issues that may arise during API calls:

- **Structured Error Parsing**: Parses errors from both OpenAI and Claude API responses
- **Detailed Logging**: Comprehensive logging and debugging information for API failures
- **Graceful Degradation**: Handles connection timeouts, rate limits, and network errors
- **Automatic Retry**: Enhanced client reliability with automatic retry mechanisms
- **Clear Messages**: Provides clear error messages to clients for debugging and troubleshooting
