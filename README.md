# Anthropic API Proxy for Multiple Model Providers 🔄

**Claude Code version: 1.0.25**

**Use Anthropic clients (like Claude Code) with multiple model providers.** 🤝

A proxy server that lets you use Anthropic clients with various model providers via LiteLLM. 🌉


![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- OpenAI API key (optional) 🔑
- Google AI Studio (Gemini) API key (optional) 🔑
- Custom model API key (if using custom models) 🔑
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/1rgs/claude-code-openai.git
   cd claude-code-openai
   ```

2. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Configure Environment Variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your API keys and model configurations:

   *   `ANTHROPIC_API_KEY`: (Optional) Needed only if proxying *to* Anthropic models.
   *   `OPENAI_API_KEY`: Your OpenAI API key (Required if using OpenAI models as fallback or primary).
   *   `GEMINI_API_KEY`: Your Google AI Studio (Gemini) API key (Required if using the default Gemini preference).
   *   `PREFERRED_PROVIDER` (Optional): Set to `google` (default) or `openai`. This determines the primary backend for mapping `haiku`/`sonnet`.
   *   `BIG_MODEL` (Optional): The model to map `sonnet` requests to. Defaults to `gemini-2.5-pro-preview-03-25` (if `PREFERRED_PROVIDER=google` and model is known) or `gpt-4o`.
   *   `SMALL_MODEL` (Optional): The model to map `haiku` requests to. Defaults to `gemini-2.0-flash` (if `PREFERRED_PROVIDER=google` and model is known) or `gpt-4o-mini`.

   **New Advanced Configuration**:
   *   `HOST`: Server host (default: 0.0.0.0)
   *   `PORT`: Server port (default: 8082)
   *   `LOG_LEVEL`: Log verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL; default: WARNING)
   *   `MAX_TOKENS_LIMIT`: Max tokens per request (default: 16384)
   *   `REQUEST_TIMEOUT`: Request timeout in seconds (default: 120)
   *   `MAX_RETRIES`: Max request retries (default: 2)
   *   `CUSTOM_MODELS_FILE`: Path to custom models config (default: custom_models.yaml)

   **Mapping Logic:**
   - If `PREFERRED_PROVIDER=google` (default), `haiku`/`sonnet` map to `SMALL_MODEL`/`BIG_MODEL` prefixed with `gemini/` *if* those models are in the server's known `GEMINI_MODELS` list.
   - Otherwise (if `PREFERRED_PROVIDER=openai` or the specified Google model isn't known), they map to `SMALL_MODEL`/`BIG_MODEL` prefixed with `openai/`.

4. **Run the server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```
   *(`--reload` is optional, for development)*

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use the configured backend models (defaulting to Gemini) through the proxy. 🎯

## Model Mapping 🗺️

The proxy automatically maps Claude models to either OpenAI or Gemini models based on the configured model:

| Claude Model | Default Mapping | When BIG_MODEL/SMALL_MODEL is a Gemini model |
|--------------|--------------|---------------------------|
| haiku | openai/gpt-4o-mini | gemini/[model-name] |
| sonnet | openai/gpt-4o | gemini/[model-name] |

### Supported Models

#### OpenAI Models
The following OpenAI models are supported with automatic `openai/` prefix handling:
- gpt-4o
- gpt-4o-mini
- gpt-4.5-preview
- gpt-4o-audio-preview
- chatgpt-4o-latest
- o3-mini
- o1
- o1-mini
- o1-pro

#### Gemini Models
The following Gemini models are supported with automatic `gemini/` prefix handling:
- gemini-2.5-pro-preview-03-25
- gemini-2.0-flash
- gemini-1.5-pro-latest
- gemini-1.5-pro-preview-0514
- gemini-1.5-flash-latest
- gemini-1.5-flash-preview-0514
- gemini-pro
- gemini-2.5-pro-preview-05-06
- gemini-2.5-flash-preview-05-20

### Model Prefix Handling
The proxy automatically adds the appropriate prefix to model names:
- OpenAI models get the `openai/` prefix
- Gemini models get the `gemini/` prefix
- Custom models get the `custom/` prefix

For example:
- `gpt-4o` becomes `openai/gpt-4o`
- `gemini-2.5-pro-preview-03-25` becomes `gemini/gemini-2.5-pro-preview-03-25`
- When BIG_MODEL is set to a Gemini model, Claude Sonnet will map to `gemini/[model-name]`

### Customizing Model Mapping

You can customize which models are used via environment variables:

- `BIG_MODEL`: The model to use for Claude Sonnet models (default: "gpt-4o")
- `SMALL_MODEL`: The model to use for Claude Haiku models (default: "gpt-4o-mini")
- `PREFERRED_PROVIDER`: Set to "google" (default), "openai" or "custom" to choose the primary backend

Add these to your `.env` file to customize:
```
OPENAI_API_KEY=your-openai-key
# For OpenAI models
PREFERRED_PROVIDER=openai
BIG_MODEL=gpt-4o
SMALL_MODEL=gpt-4o-mini

# For Gemini models
PREFERRED_PROVIDER=google
BIG_MODEL=gemini-2.5-pro-preview-03-25
SMALL_MODEL=gemini-2.0-flash

# For custom OpenAI-compatible models
PREFERRED_PROVIDER=custom
BIG_MODEL=my-large-model
SMALL_MODEL=my-small-model
```

### Custom OpenAI-Compatible Models

You can add support for custom OpenAI-compatible models by creating a `custom_models.yaml` file. Alternatively, check [LiteLLM's supported providers](https://docs.litellm.ai/docs/providers) if you want to use specific provider's models without writing custom configs:

```yaml
- model_id: "my-model"
  api_base: "https://api.example.com"
  api_key_name: "MY_API_KEY"
  can_stream: true
  max_tokens: 8192
  model_name: "actual-model-name"  # Optional - if different from model_id
```

Key features of custom models:
- Supports any OpenAI-compatible API endpoint
- Handles streaming responses if `can_stream: true`
- Automatically adds `custom/` prefix to model names
- Uses specified API key from environment variables
- Supports tool use (function calling) if the backend supports it
- Maintains Anthropic API compatibility while using custom backends
- Automatically loads from `custom_models.yaml` on server startup
- Supports multiple custom models in the same configuration file

Then set these environment variables:
```
PREFERRED_PROVIDER=custom
BIG_MODEL=my-model
MY_API_KEY=your-api-key
```

Or set them directly when running the server:
```bash
# Using OpenAI models (with uv)
BIG_MODEL=gpt-4o SMALL_MODEL=gpt-4o-mini uv run uvicorn server:app --host 0.0.0.0 --port 8082

# Using custom models (with uv)
PREFERRED_PROVIDER=custom BIG_MODEL=my-large-model SMALL_MODEL=my-small-model uv run uvicorn server:app --host 0.0.0.0 --port 8082

# Using Gemini models (with uv)
BIG_MODEL=gemini-2.5-pro-preview-03-25 SMALL_MODEL=gemini-2.0-flash uv run uvicorn server:app --host 0.0.0.0 --port 8082

# Using custom models (with uv)
PREFERRED_PROVIDER=custom BIG_MODEL=my-large-model SMALL_MODEL=my-small-model uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

## How It Works 🧩

This proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Translating** the requests to OpenAI format via LiteLLM 🔄
3. **Sending** the translated request to OpenAI 📤
4. **Converting** the response back to Anthropic format 🔄
5. **Returning** the formatted response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. 🌊

## Debugging 🐛

### Enabling Debug Logging
To enable detailed debug logging, set the logging level to `DEBUG` in `server.py` by modifying the `logging.basicConfig` call:
```python
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
)
```

### Logging Levels
- `INFO`: Basic request/response logging (default)
- `DEBUG`: Detailed logs including model mappings and request processing

### Filtering Logs
The server filters out certain verbose logs by default. To see all logs, remove or modify the `MessageFilter` class in `server.py`.

### Troubleshooting
- **Server not starting**: Check logs for errors related to missing API keys or invalid model configurations.
- **Connection issues**: Verify the proxy URL (`http://localhost:8082`) is accessible.
- **Model mapping failures**: Ensure `BIG_MODEL` and `SMALL_MODEL` are correctly set in `.env`.

### Common Issues
- **Invalid API keys**: Double-check your API keys in `.env`.
- **Unsupported models**: Confirm model names match the supported lists in the "Model Mapping" section.
- **Streaming errors**: Disable streaming with `--no-streaming` in tests to isolate issues.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁
