# Anthropic API Proxy for Multiple Model Providers 🔄

## Supported Claude Code Versions

| Version | Status | Notes |
|---------|--------|-------|
| 1.0.35  | ✅ Tested | Current supported version |
| 1.0.x   | ⚠️ Likely Compatible | Earlier versions may work but untested |

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

A proxy server that translates Anthropic API requests to multiple model providers (OpenAI, Gemini, custom) using native OpenAI SDK. Features intelligent routing based on token count, thinking flag, and model name.

## Key Features
- 🚀 Intelligent model routing (background/think/long-context modes)
- 🔄 Supports OpenAI, Gemini, and custom OpenAI-compatible models
- 🛠️ Full tool use (function calling) support
- 📊 Token counting and cost tracking
- 🌉 Seamless Claude API compatibility
- 🔄 Automatic server restart on .env file changes

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

**Note**: The server automatically monitors the `.env` file for changes and restarts itself when modifications are detected, ensuring new configuration takes effect immediately.

### Router Configuration
Configure model routing in `.env`:
```
ROUTER_BACKGROUND="custom/deepseek-v3-250324"  # Default for haiku
ROUTER_THINK="custom/deepseek-r1-250528"       # Thinking tasks
ROUTER_LONG_CONTEXT="custom/gemini-2.5-pro"    # Long context
ROUTER_DEFAULT="custom/deepseek-v3-250324"     # Fallback
```

### Running the Server

There are several ways to run the server, depending on your development needs.

#### Development (Foreground)
For most development work, run the server in the foreground with auto-reload. This is the recommended way to run the server while you are making changes.

```bash
make dev
```
This command will stream logs directly to your terminal. Press `Ctrl+C` to stop the server.

This is equivalent to running manually:
```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

#### Production / Background
To run the server as a background process, use:
```bash
make run
```
This will start the server and write its logs to `uvicorn.log`. The process ID (PID) will be saved in `uvicorn.pid`.

You can manage the background process with:
- `make stop`: Stops the background server process.
- `make restart`: Restarts the background server process.

### Connecting Claude Code
```bash
ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

## Recommended Usage Strategy

### When to Use Claude Official Models vs Proxy

**Primary Recommendation: Use Claude Official Models First**

If you have access to Claude API or a Claude Pro subscription, we recommend using Claude's official models as your primary choice for the best experience:

- ✅ **Use Claude Official**: Full feature compatibility, latest model updates, optimal performance
- 🔄 **Use This Proxy**: When Claude API quota is exhausted or as a cost-effective alternative

### Setup Strategy

1. **Default Setup** (Official Claude):
   ```bash
   # Use Claude normally
   claude
   ```

2. **Fallback Setup** (This Proxy):
   ```bash
   # Switch to proxy when official API limit reached
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

### Benefits of This Approach

- **Best of Both Worlds**: Official Claude for primary work, proxy for extended usage
- **Cost Management**: Use official Claude for critical tasks, proxy for background/testing work
- **Quota Extension**: Continue working seamlessly when daily limits are reached
- **Development Flexibility**: Test with different models while maintaining Claude compatibility

### When the Proxy Excels

This proxy is particularly valuable for:
- 🧠 **Reasoning Tasks**: Third-party models optimized for complex problem-solving
- 💰 **Cost-Sensitive Workflows**: Significant cost savings with alternative providers
- 📊 **High-Volume Usage**: Extended usage beyond API quotas
- 🔬 **Model Experimentation**: Testing different providers with consistent interface

### Testing

The project uses `pytest` as the primary testing framework and includes a comprehensive test suite. You can run tests using the provided `make` commands.

#### Running Test Suites

- **Run the main test suite (recommended)**:
  ```bash
  make test
  ```

- **Run the `unittest`-based suite**:
  ```bash
  make test-unittest
  ```

- **Generate a test coverage report**:
  ```bash
  make test-cov
  ```
  To view the report in HTML format, run `make test-cov-html` and open `htmlcov/index.html`.

#### Quick Development Tests

For faster feedback during development, you can run specific sets of tests:

- `make test-basic`: Runs basic request validation tests.
- `make test-tools`: Tests tool usage scenarios.
- `make test-custom`: Focuses on custom model integrations.
- `make test-comparison`: Compares proxy output with the official Anthropic API.
- `make test-performance`: Runs performance tests (requires running server)

### Performance Testing

The `performance_test.py` script measures proxy overhead when routing requests to third-party models (DeepSeek, Gemini, etc.) vs direct API calls.

#### Key Metrics Measured:
- Time to First Chunk (TTFC)
- Total request duration
- Token throughput (tokens/sec)
- Response format overhead
- Token consistency between direct and proxy calls
- Cost impact analysis

#### Usage:
```bash
python performance_test.py --model_id <model_id_from_models.yaml>
```

Example output analysis:
```
PROXY EFFICIENCY RATING: 🟢 EXCELLENT
  Proxy adds minimal overhead.
  - Criteria: TTFC Overhead < 50ms AND Throughput Loss < 5%
  - Actuals: TTFC Overhead = 32.14ms, Throughput Loss = 2.1%
```

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
├── performance_test.py        # Proxy performance analysis tool
├── tests/                    # Test directory
│   ├── test_unittest.py      # Modern unittest framework with API comparison ⭐
│   └── test_conversions.py   # Format conversion unit tests
├── models.yaml.example       # Example custom model config
├── Makefile                  # Development commands
├── pyproject.toml           # Python project configuration (pytest & ruff)
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

#### **tests/test_unittest.py** ⭐ (Recommended)
- Modern unittest framework with `IsolatedAsyncioTestCase`
- 13 test classes with 33+ async test methods
- **Ground truth validation** against official Anthropic API
- **Behavioral difference analysis** for expected variations
- **Custom model conversion testing** (Gemini, DeepSeek, etc.)
- **Tool use validation** with 13 different tools
- **Advanced streaming tests** with event validation
- **Claude Code workflow testing** for development scenarios

#### **tests/test_conversions.py**
- Format conversion unit tests between Claude and OpenAI formats
- Message processing validation
- Content block conversion testing
- Tool use and result conversion validation

#### **models.yaml**
- YAML configuration for custom OpenAI-compatible models
- Includes API endpoints, authentication, and pricing
- Supports DeepSeek, local models, and other providers
- Cost tracking and token limit configuration

## Custom Models
Add custom OpenAI-compatible models in `models.yaml`:
```yaml
- model_id: my-model
  model_name: model-name-used-in-request
  api_base: https://api.example.com
  api_key_name: MY_API_KEY
  can_stream: true
  max_tokens: 8192
  input_cost_per_token: 0.00000027  # Optional pricing info for cost tracking
  output_cost_per_token: 0.00000055 # Optional pricing info for cost tracking
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
   - Customize `models.yaml` if needed

3. **Development Cycle**:
   ```bash
   make dev            # Start development server
   make test           # Run test suite
   make lint           # Check code quality
   make format         # Format code
   ```

### Adding New Models

1. **Add to models.yaml**:
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
   # Modern pytest suite (recommended)
   pytest tests/test_unittest.py::TestCustomModels::test_gemini_tool_conversion -v
   # Or unittest
   python -m unittest tests.test_unittest.TestCustomModels.test_gemini_tool_conversion -v
   ```

### Testing Workflow

#### Development Testing
- **Quick validation**: `python -m unittest tests.test_unittest.TestBasicRequests -v`
- **Model-specific testing**: `python -m unittest tests.test_unittest.TestCustomModels -v`
- **API comparison**: `python -m unittest tests.test_unittest.TestAnthropicComparison -v`

#### Pre-commit Testing
- **Full validation**: `make test` (pytest, recommended)
- **Alternative**: `make test-unittest` (unittest framework)
- **Code quality**: `make lint && make format`

#### Model Validation
- **Ground truth comparison**: unittest framework automatically compares with Anthropic API
- **Behavioral analysis**: unittest distinguishes between bugs and expected differences
- **Custom model testing**: Dedicated test classes for specific providers
- **Performance monitoring**: Token counts and response times in test output

### Debugging
Set `LOG_LEVEL=DEBUG` in `.env` for detailed logs.

Performance metrics from `performance_test.py` can help identify:
- Network latency issues (high TTFC overhead)
- Processing bottlenecks (throughput loss)
- Format conversion inefficiencies (response size overhead)

### Log Files
The default log file path can be configured using the `LOG_FILE_PATH` environment variable in your `.env` file, defaulting to the directory containing `server.py` if not specified.

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
- **Enhanced logging** with ANSI color formatting via Colors class for better visibility
- **Automatic configuration reload** with .env file monitoring and restart capability

#### 2. **models.py** - Data Models & Conversion
- **Pydantic models** for type-safe API contracts
- **Format conversion utilities** (Anthropic ↔ OpenAI)
- **Tool choice handling** with parallel execution support
- **Content block processing** (text, images, tool use, thinking)

#### 3. **Configuration System**
- **Environment-based config** (.env file)
- **Custom model definitions** (models.yaml)
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
1. **Long Context Check**: If input > configured threshold → use `ROUTER_LONG_CONTEXT` (configurable via `LONG_CONTEXT_THRESHOLD` in .env)
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

Models are defined in `models.yaml`:

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
- `/v1/messages`: The main endpoint for handling chat completion requests. It supports streaming and intelligent model routing.
- `/v1/messages/count_tokens`: Calculates the number of tokens for a given set of messages, which is useful for estimating costs.
- `/v1/messages/test_conversion`: A testing endpoint to directly convert and proxy a request to a specific model, bypassing the router.
- `/v1/stats`: Returns token usage statistics for the current session, including costs and tokens for both input and output.
- `/test-connection`: Checks the connection to the configured API providers.

## Error Handling
- Centralized error extraction and formatting
- Detailed logging for debugging

## Performance
- Batch operations for efficiency
- Chunked reading for large files
- Concurrent processing where possible

## Code Design Principles

When refactoring or modifying the codebase, follow these established design principles:

### SOLID Principles
- **Single Responsibility Principle (SRP)**: Each function/class should have one reason to change. Keep functions focused on a single task.
- **Open/Closed Principle (OCP)**: Code should be open for extension but closed for modification. Use function extraction to make code more extensible.

### Code Quality Guidelines
- **KISS Principle**: Keep it simple. Break complex logic into smaller, understandable functions.
- **DRY Principle**: Don't repeat yourself. Extract common logic into reusable functions.
- **Function Length**: Keep functions short and focused (ideally under 50 lines following Code Complete 2 guidelines).

### Codebase Patterns
- **Error Handling**: Use centralized error extraction and formatting functions (`_extract_error_details`, `_format_error_message`).
- **Model Validation**: Use the shared `validate_and_map_model` function for consistent model handling.
- **API Integration**: Separate concerns with dedicated setup functions (`setup_api_key_for_request`, `process_openai_message_format`).
- **Streaming**: Use event helper functions for consistent streaming response formatting.

### Refactoring Guidelines
- **Extract Functions**: When a function exceeds 50 lines or handles multiple concerns, extract subfunctions.
- **Eliminate Duplication**: Look for repeated code patterns and extract them into shared utilities.
- **Improve Readability**: Use descriptive function names that clearly indicate their purpose.
- **Maintain Testability**: Small, focused functions are easier to test and debug.

## Security
- Input validation
- Secure API key handling
- Error handling without exposing sensitive data

## Testing Frameworks Summary

### pytest vs unittest Frameworks

| Feature | `pytest` ⭐ (Recommended) | `unittest` |
|---------|-------------------------|------------|
| **Framework** | Modern pytest with auto-discovery | `unittest.IsolatedAsyncioTestCase` |
| **Test Count** | 33+ async test methods | 33+ async test methods |
| **Organization** | Automatic test discovery in `tests/` | 13 organized test classes |
| **Configuration** | `pyproject.toml` integration | Standard unittest patterns |
| **API Comparison** | ✅ Proxy vs Anthropic ground truth | ✅ Proxy vs Anthropic ground truth |
| **Behavioral Analysis** | ✅ Distinguishes bugs vs differences | ✅ Distinguishes bugs vs differences |
| **Custom Models** | ✅ Dedicated test classes | ✅ Dedicated test classes |
| **Development** | ✅ Flexible test running | ✅ Targeted test running |
| **Async Support** | ✅ pytest-asyncio integration | ✅ Native async/await |
| **Maintenance** | ✅ Industry standard | ✅ Python standard library |

**Recommendation**: Use `pytest` for development and CI/CD as it provides better test discovery, reporting, and integration with modern Python tooling.

### Quick Testing Commands

```bash
# Development workflow
make test-basic          # Quick validation (3 tests)
make test-tools          # Tool functionality (6 tests)
make test-custom         # Custom models (3 tests)
make test-comparison     # API comparison (2 tests)

# Specific tests
make test-gemini         # Gemini conversion test
make test-calculator     # Calculator tool test

# Full suites
make test               # Modern pytest (recommended)
```

## Credit & Acknowledgment

This project is forked from and based on [claude-code-proxy](https://github.com/1rgs/claude-code-proxy) by [@1rgs](https://github.com/1rgs). The original project provided the foundation for proxying Anthropic API requests to other model providers.

The intelligent routing feature that distributes requests to different models based on token count and thinking flag was inspired by [claude-code-router](https://github.com/musistudio/claude-code-router).

### Key Enhancements in This Fork
- **Removed LiteLLM dependency**: Replaced with native OpenAI SDK for better control and simplified architecture
- **Enhanced testing framework**: Added modern unittest suite with ground truth API comparison
- **Improved model support**: Enhanced DeepSeek and Gemini integration with dedicated test validation
- **Intelligent routing**: background/think/long-context modes with comprehensive testing
- **Extended custom model support**: YAML configuration with validation testing
- **Better error handling**: Centralized error extraction with behavioral difference analysis
- **Additional API endpoints**: Token counting, statistics, and test conversion endpoints
- **Comprehensive development tooling**: Multiple test frameworks and development commands
- **Streamlined architecture**: OpenAI SDK for all third-party model communications

Special thanks to the original author for creating the initial proxy architecture that made this enhanced version possible.
