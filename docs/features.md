# Features

This document describes the key features of the proxy.

## Intelligent Routing

The proxy can intelligently route requests to different models based on a set of rules. This allows for cost optimization and performance improvements by using the most appropriate model for each task.

- **Token-Based Routing**: Requests with a high token count can be routed to models that support a larger context window.
- **Flag-Based Routing**: A `thinking` flag in the request can be used to route the request to a more powerful model for complex tasks.
- **Default Routing**: A default model is used when no other routing rules match.

Routing rules are configured through environment variables (e.g., `ROUTER_DEFAULT`, `ROUTER_THINK`).

## Streaming Support

The proxy fully supports streaming responses from the underlying models. It correctly handles Server-Sent Events (SSE) and ensures that the data is streamed back to the client in the proper Anthropic API format. This is crucial for real-time applications and for providing a responsive user experience.

## Custom Model Support

The proxy allows you to use custom OpenAI-compatible models. You can define your own models in the `models.yaml` file, and the proxy will handle the necessary API calls. This is useful for integrating with custom or fine-tuned models.

## Token Counting and Cost Calculation

The proxy uses the `tiktoken` library to accurately count the number of tokens in both the prompt and the response. This information is used for:

- **Intelligent Routing**: Making decisions based on the token count.
- **Cost Calculation**: The proxy can be extended to calculate the cost of each API call based on the token count, which is useful for monitoring and billing.

## Error Handling

The proxy includes robust error handling to gracefully manage issues that may arise during API calls. It provides clear error messages to the client, which helps with debugging and troubleshooting.
