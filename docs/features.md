# Features

This document describes the key features of the proxy.

## Model Selection

The proxy does not switch models on its own. The incoming request specifies the model, and the proxy maps that model to a provider entry in `models.yaml`. This keeps selection logic in ccproxy or the caller.

## Streaming Support

The proxy fully supports streaming responses from the underlying models. It correctly handles Server-Sent Events (SSE) and ensures that the data is streamed back to the client in the proper Anthropic API format. This is crucial for real-time applications and for providing a responsive user experience.

## Custom Model Support

The proxy allows you to use custom OpenAI-compatible models. You can define your own models in the `models.yaml` file, and the proxy will handle the necessary API calls. This is useful for integrating with custom or fine-tuned models.

## Usage Tracking

The proxy records usage statistics from provider responses. It does not perform local token counting.

## Error Handling

The proxy includes robust error handling to gracefully manage issues that may arise during API calls. It provides clear error messages to the client, which helps with debugging and troubleshooting.
