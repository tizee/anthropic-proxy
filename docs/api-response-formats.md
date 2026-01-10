# API Response Formats Reference

This document provides reference information about the API response formats used by different model providers that this proxy server supports. Understanding these formats is essential for developing and maintaining the conversion logic between different API standards.

## Table of Contents
- [Claude API Streaming Format](#claude-api-streaming-format)
- [OpenAI API Streaming Format](#openai-api-streaming-format)
- [DeepSeek API Response Format](#deepseek-api-response-format)
- [VolcEngine API Response Format](#volcengine-api-response-format)

## Claude API Streaming Format

**Source**: [https://platform.claude.com/docs/en/build-with-claude/streaming.md](https://platform.claude.com/docs/en/build-with-claude/streaming.md)

### Event Types
Claude API uses server-sent events (SSE) for streaming with the following event flow:

1. **`message_start`**: Initial message with empty content
2. **Content blocks**:
   - `content_block_start`
   - `content_block_delta`
   - `content_block_stop`
3. **`message_delta`**: Top-level message changes
4. **`message_stop`**: Final event
5. **`ping` events**: Keep-alive events
6. **Error events**: Stream errors like `overloaded_error`

### Content Block Delta Types
- **Text deltas**: `text_delta` events for text content
- **Input JSON deltas**: `input_json_delta` for tool use parameters (partial JSON strings)
- **Thinking deltas**: `thinking_delta` and `signature_delta` for extended thinking

### Technical Details
- Token counts in `message_delta` events are cumulative
- Tool use streaming supports fine-grained parameter value streaming (beta feature)
- Extended thinking with streaming shows Claude's step-by-step reasoning
- Each content block has an `index` corresponding to its position in the final message `content` array
- For tool use blocks, deltas are partial JSON strings that need to be accumulated and parsed
- The signature delta verifies integrity of thinking blocks

## OpenAI API Streaming Format

**Source**: [https://platform.openai.com/docs/api-reference/chat-streaming](https://platform.openai.com/docs/api-reference/chat-streaming)

### Streaming Configuration
- Set `"stream": true` in the request
- Responses follow the Server-Sent Events (SSE) protocol

### Response Format
OpenAI streaming responses consist of chunks with the following structure:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1694268190,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "Hello"
      },
      "finish_reason": null
    }
  ]
}
```

### Key Features
- Each chunk contains a `delta` field with incremental content
- The stream ends with a `data: [DONE]` message
- Final chunk may include usage statistics if configured
- Supports tool calls in streaming mode

## DeepSeek API Response Format

**Source**: [https://api-docs.deepseek.com/api/create-chat-completion](https://api-docs.deepseek.com/api/create-chat-completion)

### Non-Streaming Response Format

```json
{
  "id": "string",
  "choices": [
    {
      "finish_reason": "stop|length|content_filter|tool_calls|insufficient_system_resource",
      "index": 0,
      "message": {
        "content": "string",
        "reasoning_content": "string",  // For deepseek-reasoner model only
        "tool_calls": [
          {
            "id": "string",
            "type": "function",
            "function": {
              "name": "string",
              "arguments": "string"
            }
          }
        ],
        "role": "assistant"
      },
      "logprobs": {
        "content": [
          {
            "token": "string",
            "logprob": 0,
            "bytes": [0],
            "top_logprobs": [
              {
                "token": "string",
                "logprob": 0,
            "bytes": [0]
              }
            ]
          }
        ],
        "reasoning_content": [
          {
            "token": "string",
            "logprob": 0,
            "bytes": [0],
            "top_logprobs": [
              {
                "token": "string",
                "logprob": 0,
                "bytes": [0]
              }
            ]
          }
        ]
      }
    }
  ],
  "created": 0,
  "model": "string",
  "system_fingerprint": "string",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 0,
    "prompt_tokens": 0,
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 0,
    "total_tokens": 0,
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  }
}
```

### Streaming Response
- **Content-Type**: `text/event-stream`
- **Object Type**: `chat.completion.chunk`
- Use `"stream_options": {"include_usage": true}` to include token usage in final chunk
- Stream terminated by `data: [DONE]` message

### Key Response Fields
1. **`finish_reason` Values**:
   - `stop`: Model hit natural stop point or provided stop sequence
   - `length`: Maximum tokens reached
   - `content_filter`: Content omitted due to content filters
   - `tool_calls`: Model called a tool
   - `insufficient_system_resource`: Request interrupted due to insufficient system resources

2. **Special Features**:
   - `reasoning_content`: For deepseek-reasoner model only - contains reasoning before final answer
   - `thinking` parameter: Controls switch between thinking (`enabled`) and non-thinking (`disabled`) modes

## VolcEngine API Response Format

**Source**: [https://www.volcengine.com/docs/82379/1494384?lang=zh](https://www.volcengine.com/docs/82379/1494384?lang=zh)

### Streaming Control
- **`stream`** (boolean/null, default: `false`):
  - `false`: Model generates all content first, returns complete result in one response
  - `true`: Returns content in chunks following SSE protocol

### Streaming Options (`stream_options`)
- **`stream_options.include_usage`** (boolean/null, default: `false`):
  - `true`: Before `data: [DONE]`, an additional chunk with token usage in `usage` field
  - `false`: No token usage information before end of output

- **`stream_options.chunk_include_usage`** (boolean/null, default: `false`):
  - `true`: Each chunk includes cumulative token usage up to that point
  - `false`: Token usage not included in every chunk

### Non-Streaming Response Format

```json
{
  "id": "0217426318107460cfa43dc3f3683b1de1c09624ff49085a456ac",
  "model": "doubao-1-5-pro-32k-250115",
  "service_tier": "default",
  "created": 1742631811,
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?",
        "reasoning_content": null,
        "tool_calls": null
      },
      "logprobs": null,
      "moderation_hit_type": null
    }
  ],
  "usage": {
    "total_tokens": 28,
    "prompt_tokens": 19,
    "completion_tokens": 9,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  }
}
```

### Streaming Response Format
- **Object field**: `"object": "chat.completion.chunk"` (instead of `"chat.completion"`)
- **Usage field behavior**:
  - Default: `null` (no token usage statistics)
  - With `include_usage: true`: Final chunk contains complete usage data with empty `choices` array
  - With `chunk_include_usage: true`: Each chunk contains cumulative token usage

### Key Response Fields
- **`choices` array elements**:
  - `index`: Element index in choices list
  - `finish_reason`: Why model stopped generating tokens:
    - `stop`: Natural end or truncated by `stop` parameter
    - `length`: Reached output limits
    - `content_filter`: Blocked by content moderation
    - `tool_calls`: Model called tools

- **`service_tier`**: Shows if TPM保障包 was used
  - `scale`: Used TPM保障包额度 (higher service level)
  - `default`: Did not use TPM保障包额度 (default service level)

## Comparison Summary

| Provider | Streaming Protocol | Special Features | Key Differentiators |
|----------|-------------------|------------------|---------------------|
| **Claude** | SSE with custom events | Thinking blocks, tool use parameter streaming | Fine-grained content block events, signature verification |
| **OpenAI** | Standard SSE | Tool calls in streaming | Mature API, extensive documentation |
| **DeepSeek** | SSE with `chat.completion.chunk` | Reasoning content, thinking modes | Detailed token usage, cache statistics |
| **VolcEngine** | SSE with service tier indicator | TPM保障包 tracking, chunk usage options | Chinese documentation, enterprise features |

## Implementation Notes

1. **Claude → OpenAI Conversion**:
   - Map Claude's `content_block_delta` events to OpenAI's `delta.content`
   - Handle tool use parameter accumulation for OpenAI format
   - Convert thinking blocks to appropriate reasoning content

2. **OpenAI → Claude Conversion**:
   - Map OpenAI's `delta.content` to Claude's text deltas
   - Handle tool calls with appropriate parameter formatting
   - Preserve usage statistics where possible

3. **Common Patterns**:
   - All providers use SSE for streaming
   - Most support tool calls with similar structure
   - Token usage tracking varies in detail
   - Error handling follows similar patterns

This reference document should be updated as API specifications evolve and new providers are added to the proxy server.
