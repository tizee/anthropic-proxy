# Architecture

This document provides a high-level overview of the proxy's architecture.

## Core Components

The proxy is composed of several key components that work together to process and route API requests.

- **Proxy Server (`anthropic_proxy/server.py`)**: The main entry point of the application. It uses FastAPI to create a web server that listens for incoming Anthropic API requests. It handles request validation, sets up API keys, and delegates processing to other components.

- **Client (`anthropic_proxy/client.py`)**: Responsible for making outgoing requests to the underlying LLM APIs (e.g., OpenAI, Gemini). It uses `httpx` for asynchronous HTTP requests and handles the communication with the final API endpoint.

- **Converter (`anthropic_proxy/converter.py`)**: This component is responsible for translating the Anthropic API request format into the format expected by the target model (e.g., OpenAI or Gemini). It also converts the response back into the Anthropic format. This ensures seamless interoperability between different API standards.

- **Streaming (`anthropic_proxy/streaming.py`)**: Handles the complexities of streaming responses. It processes Server-Sent Events (SSE) from the target API and transforms them into the format expected by the Anthropic client.

- **Configuration (`anthropic_proxy/config.py`)**: Manages the application's configuration, including server settings, model mappings, and routing rules. It loads settings from environment variables and `models.yaml`. Note: API keys are no longer stored in config - they're passed via request headers from ccproxy.

- **Utilities (`anthropic_proxy/utils.py`)**: A collection of helper functions used across the application, such as token counting and error handling.

- **Types (`anthropic_proxy/types.py`)**: Defines the data structures and types used throughout the application, ensuring type safety and clarity.

## ccproxy Integration

This proxy is designed to work with **ccproxy** (Claude Code wrapper script), which handles API key management externally:

```
ccproxy (sets ANTHROPIC_BASE_URL=localhost:8082, passes API key via Authorization header)
    ↓
Claude Code (sends Anthropic-format requests with model name + auth header)
    ↓
claude-code-proxy (extracts key from header, routes model to API URL, converts format)
    ↓
Provider API (receives OpenAI-format request with API key)
```

**Key design decisions:**
- API keys are extracted from the `Authorization: Bearer <key>` header on each request
- Model-to-URL routing is defined in `models.yaml` without storing any API keys
- This allows unified key management through ccproxy's configuration

## Request Flow

The diagram below illustrates the request/response lifecycle within the proxy.

```mermaid
sequenceDiagram
    participant C as Client
    participant P as Proxy Server
    participant Conv as Converter
    participant LLMC as LLM Client
    participant S as Streaming
    participant T as Target Model

    C-)P: Anthropic API Request
    activate P

    P-)P: Validate & Route Request
    P-)Conv: Translate to Target Format
    activate Conv
    Conv--)P: Converted Request
    deactivate Conv

    P-)LLMC: Send Request to Model
    activate LLMC
    LLMC-)T: Forward to Model API
    activate T
    T--)LLMC: Model Response
    deactivate T
    LLMC--)P: Receive Response
    deactivate LLMC

    alt Streaming Response
        P-)S: Process SSE Stream
        activate S
        S--)P: Anthropic-formatted Chunks
        deactivate S
    else Non-Streaming Response
        P-)Conv: Convert Full Response
        activate Conv
        Conv--)P: Converted Response
        deactivate Conv
    end

    P--)C: Final Anthropic Response
    deactivate P
```

**Lifecycle Steps:**

1.  **Request Initiation**: A **Client** sends an Anthropic-formatted API request to the **Proxy Server**.
2.  **Internal Processing**: The server validates the request, applies routing rules, and uses the **Converter** to translate it into the format expected by the **Target Model**.
3.  **Model Communication**: The **LLM Client** sends the converted request to the target model's API and receives a response.
4.  **Response Handling**:
    - For **streaming** responses, the **Streaming** component processes the Server-Sent Events (SSE) stream.
    - For **non-streaming** responses, the **Converter** translates the entire payload.
5.  **Final Response**: The **Proxy Server** sends the final, Anthropic-formatted response back to the **Client**.
