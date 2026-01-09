# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Commands
To understand the available commands and project dependencies, please refer to the following files as the single source of truth:

- **`Makefile`**: Contains a list of common tasks and scripts. Run `make help` to see a full list of available commands for running the server, executing tests, and more.
- **`pyproject.toml`**: Defines the project's dependencies and tool configurations.
- use `uv` instead of `python` to run python scripts

## Documentation

This project includes a `docs/` directory with detailed information about the architecture, features, and testing. When you need to understand the project, please refer to these documents first:

- **`docs/architecture.md`**: Provides a high-level overview of the proxy's architecture, including the core components and the request flow.
- **`docs/features.md`**: Describes the key features of the proxy, such as model mapping, streaming support, and custom model integration.
- **`docs/testing.md`**: Explains how to run the unit and performance tests.

By using these documents, you can quickly get up to speed on the project without having to read through all the code.

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
- **Configuration Management (`config.py`)**: Centralized configuration handling using a `Config` class that loads from `.env` files.
- **Model & Client Management (`client.py`)**: Handles loading custom model definitions from `models.yaml` and creating provider-specific `AsyncOpenAI` clients (`create_openai_client`).
- **Request/Response Conversion (`converter.py`)**: Contains the core logic for converting message formats between Anthropic and OpenAI. Key functions include `ClaudeMessagesRequest.to_openai_request()` for outgoing requests and `convert_openai_response_to_anthropic()` for incoming responses.
- **Streaming Logic (`streaming.py`)**: All streaming-related processing is encapsulated in the `AnthropicStreamingConverter` class. This class manages the state of the streaming connection and converts OpenAI's `ChatCompletionChunk` events into the appropriate Anthropic streaming format.
- **Error Handling (`utils.py`)**: Utility functions like `_extract_error_details` and `_format_error_message` provide a consistent way to process and format exceptions throughout the application.
- **Type Definitions (`types.py`)**: Pydantic models define the data structures for API requests and responses (e.g., `ClaudeMessagesRequest`, `ClaudeMessagesResponse`), ensuring type safety and clear data contracts.

### Refactoring Guidelines
- **Extract Functions**: When a function exceeds 50 lines or handles multiple concerns, extract subfunctions.
- **Eliminate Duplication**: Look for repeated code patterns and extract them into shared utilities.
- **Improve Readability**: Use descriptive function names that clearly indicate their purpose.
- **Maintain Testability**: Small, focused functions are easier to test and debug.
