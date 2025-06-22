# Anthropic API Proxy for Multiple Model Providers 🔄

**Claude Code Version: 1.0.31**

Tested Models: DeepSeek/Gemini

A proxy server that translates Anthropic API requests to multiple model providers (OpenAI, Gemini, custom) using native OpenAI SDK. Features intelligent routing based on token count, thinking flag, and model name.

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
make run
```

Or manually:
```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

### Connecting Claude Code
```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

### Testing

The project includes a comprehensive test suite with multiple testing modes:

```bash
make test                    # Run all tests
uv run tests.py             # Direct execution
```

#### Test Script Features (`tests.py`)

**Test Categories:**
- **Simple Tests**: Basic request/response validation
- **Tool Tests**: Function calling and tool use scenarios
- **Thinking Tests**: Reasoning mode with `<thinking>` blocks
- **Streaming Tests**: Real-time response processing
- **Multi-turn Tests**: Conversation continuity

**Usage Examples:**
```bash
# Run specific test categories
python tests.py --simple                     # Basic tests only
python tests.py --tools-only                 # Tool use tests
python tests.py --thinking-only              # Thinking mode tests
python tests.py --streaming-only             # Streaming tests only
python tests.py --no-streaming               # Skip streaming tests

# Run specific tests
python tests.py --test calculator            # Single test
python tests.py --test calculator --streaming-only  # Streaming version
python tests.py --list-tests                 # Show all available tests

# Custom model testing
python tests.py --model deepseek-v3-250324   # Test custom model
python tests.py --model deepseek-v3-250324 --compare  # Compare with official
python tests.py --test calculator --model deepseek-v3-250324 --compare
```

#### Available Test Cases

**Simple Tests:**
- `hello`: Basic greeting response
- `simple`: Text-only conversation
- `multi_turn`: Conversation continuity

**Tool Tests:**
- `calculator`: Mathematical calculations
- `weather`: Weather API simulation
- `bash`: Shell command execution
- `websearch`: Web search simulation
- `parallel_tools`: Multiple tools in one request

**Thinking Tests:**
- `reasoning`: Complex problem solving
- `analysis`: Text analysis with reasoning
- `math_problem`: Mathematical reasoning

#### Test Output Features

- **Colored output** for better readability
- **Token counting** and cost estimation
- **Response time tracking**
- **Error handling validation**
- **Streaming chunk validation**
- **Tool execution verification**

### Linting
```bash
make lint
```

### Formatting
```bash
make format
```

## Project Structure

```
claude-code-proxy/
├── server.py                 # Main FastAPI proxy server
├── models.py                 # Pydantic models and format conversion
├── tests.py                  # Comprehensive test suite
├── test_conversions.py       # Format conversion unit tests
├── custom_models.yaml.example # Example custom model config
├── Makefile                  # Development commands
├── pyproject.toml           # Python project configuration
├── uv.lock                  # Dependency lock file
├── .env.example             # Environment variables template
└── README.md                # This file
```

### Key Files

#### **server.py**
- FastAPI application setup and routing
- Intelligent model selection logic
- Request/response processing pipeline
- Streaming and non-streaming handlers
- Session statistics tracking
- Error handling with provider-specific mapping

#### **models.py**
- Pydantic models for type safety
- Anthropic ↔ OpenAI format conversion utilities
- Tool choice handling and parallel execution
- Content block processing (text, images, tool use, thinking)
- Input validation and sanitization

#### **tests.py**
- Comprehensive test suite with 15+ test scenarios
- Support for streaming/non-streaming modes
- Custom model testing capabilities
- Tool use validation and parallel execution tests
- Thinking mode and reasoning tests
- Multi-turn conversation testing

#### **custom_models.yaml**
- YAML configuration for custom OpenAI-compatible models
- Includes API endpoints, authentication, and pricing
- Supports DeepSeek, local models, and other providers
- Cost tracking and token limit configuration

## Custom Models
Add custom OpenAI-compatible models in `custom_models.yaml`:
```yaml
- model_id: my-model
  model_name: model-name-used-in-request
  api_base: https://api.example.com
  api_key_name: MY_API_KEY
  can_stream: true
  max_tokens: 8192
```

## Development Workflow

### Getting Started
1. **Clone and Setup**:
   ```bash
   git clone https://github.com/tizee/claude-code-proxy.git
   cd claude-code-proxy
   uv install
   cp .env.example .env
   ```

2. **Configure Environment**:
   - Edit `.env` with your API keys
   - Configure router models in `.env`
   - Customize `custom_models.yaml` if needed

3. **Development Cycle**:
   ```bash
   make run        # Start development server
   make test       # Run test suite
   make lint       # Check code quality
   make format     # Format code
   ```

### Adding New Models

1. **Add to custom_models.yaml**:
   ```yaml
   - model_id: my-new-model
     api_base: https://api.example.com/v1
     api_key_name: MY_API_KEY
     can_stream: true
     max_tokens: 8192
   ```

2. **Set API key in .env**:
   ```
   MY_API_KEY=your_api_key_here
   ```

3. **Test the model**:
   ```bash
   python tests.py --model my-new-model --test simple
   ```

### Testing Workflow

- **Development**: Use `--test <name>` for quick single test runs
- **Pre-commit**: Run `make test` to validate all functionality
- **Model validation**: Use `--compare` flag to test against official models
- **Performance**: Monitor token counts and response times in test output

### Debugging
Set `LOG_LEVEL=DEBUG` in `.env` for detailed logs.

### Log Files
The server generates several log files for debugging:
- `server.log`: General server operations
- `simple.log`: Simple test executions
- `streaming.log`: Streaming response logs
- `thinking.log`: Thinking mode operations
- `tools.log`: Tool use and function calling logs

## Project Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────┐    ┌────────────────┐
│   Claude Code   │───▶│ Proxy Server │───▶│ Model Provider │
│   (Client)      │    │ (server.py)  │    │ (OpenAI/etc)   │
└─────────────────┘    └──────────────┘    └────────────────┘
```

#### 1. **server.py** - Main Proxy Server
- **FastAPI-based HTTP server** handling Anthropic API format
- **Intelligent routing** based on token count, thinking flag, and model name
- **Request/Response translation** between Anthropic and OpenAI formats
- **Session statistics** tracking for cost monitoring
- **Streaming support** for real-time responses

#### 2. **models.py** - Data Models & Conversion
- **Pydantic models** for type-safe API contracts
- **Format conversion utilities** (Anthropic ↔ OpenAI)
- **Tool choice handling** with parallel execution support
- **Content block processing** (text, images, tool use, thinking)

#### 3. **Configuration System**
- **Environment-based config** (.env file)
- **Custom model definitions** (custom_models.yaml)
- **Router configuration** for model selection logic

### Intelligent Routing Logic

The proxy uses a sophisticated routing system to select the optimal model:

```python
# Router modes (configured via environment variables)
ROUTER_BACKGROUND  = "deepseek-v3-250324"    # Fast tasks (claude-haiku)
ROUTER_THINK       = "deepseek-r1-250528"    # Reasoning tasks
ROUTER_LONG_CONTEXT = "gemini-2.5-pro"      # Long context (>128k tokens)
ROUTER_DEFAULT     = "deepseek-v3-250324"    # Fallback
```

**Decision Flow:**
1. **Long Context Check**: If input > 128k tokens → use `ROUTER_LONG_CONTEXT`
2. **Thinking Mode**: If `thinking` enabled → use `ROUTER_THINK`
3. **Model Override**: If specific model requested → respect user choice
4. **Background Mode**: Default fast processing → use `ROUTER_BACKGROUND`

### Request Processing Pipeline

```
Anthropic Request → Validation → Model Selection → Format Conversion →
OpenAI API Call → Response Processing → Anthropic Format → Client
```

1. **Request Validation**: Pydantic model validation
2. **Token Counting**: Estimate input tokens using tiktoken
3. **Model Resolution**: Apply routing logic + custom model lookup
4. **API Key Setup**: Configure appropriate credentials
5. **Format Translation**: Convert messages, tools, and content blocks
6. **Provider Call**: Use OpenAI SDK with custom endpoints
7. **Response Processing**: Handle streaming/non-streaming responses
8. **Statistics Tracking**: Update session cost/token counters

### Custom Model Integration

Models are defined in `custom_models.yaml`:

```yaml
- model_id: deepseek-v3-250324
  api_base: https://api.deepseek.com
  api_key_name: DEEPSEEK_API_KEY
  can_stream: true
  max_tokens: 8192
  input_cost_per_token: 0.00000027
  output_cost_per_token: 0.00000055
```

### Error Handling Strategy

- **Centralized error extraction** with detailed logging
- **Provider-specific error mapping** to Anthropic format
- **Retry logic** with exponential backoff
- **Graceful degradation** for partial failures

## Supported Models
| Provider | Example Models |
|----------|----------------|
| OpenAI   | gpt-4o, gpt-4o-mini |
| Gemini   | gemini-2.5-pro, gemini-2.0-flash |
| Custom   | Any OpenAI-compatible API |

## API Endpoints
- `/v1/messages`: Main endpoint for message requests
- `/v1/messages/count_tokens`: Count tokens in messages
- `/v1/stats`: Get session statistics
- `/health`: Health check endpoint
- `/test-connection`: Test API connectivity

## Error Handling
- Centralized error extraction and formatting
- Detailed logging for debugging

## Performance
- Batch operations for efficiency
- Chunked reading for large files
- Concurrent processing where possible

## Code Quality
Follows SOLID principles:
- Single Responsibility Principle (SRP)
- Open/Closed Principle (OCP)
- KISS and DRY principles

## Security
- Input validation
- Secure API key handling
- Error handling without exposing sensitive data

## Credit & Acknowledgment

This project is forked from and based on [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) by [@1rgs](https://github.com/1rgs). The original project provided the foundation for proxying Anthropic API requests to other model providers.

The intelligent routing feature that distributes requests to different models based on token count and thinking flag was inspired by [claude-code-router](https://github.com/musistudio/claude-code-router).

### Key Enhancements in This Fork
- **Removed LiteLLM dependency**: Replaced with native OpenAI SDK for better control and simplified architecture
- Enhanced intelligent routing with background/think/long-context modes
- Improved DeepSeek and Gemini integration
- Extended custom model support via YAML configuration
- Better error handling and logging
- Additional API endpoints for token counting and statistics
- Comprehensive testing and development tooling
- Streamlined request processing using OpenAI SDK for all third-party model communications

Special thanks to the original author for creating the initial proxy architecture that made this enhanced version possible.
