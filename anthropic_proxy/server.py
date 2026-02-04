"""
FastAPI server for the anthropic proxy.
This module contains the FastAPI application and API endpoints.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import (
    AsyncStream,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
)

from .antigravity import handle_antigravity_request
from .claude_code import handle_claude_code_request
from .client import (
    CUSTOM_OPENAI_MODELS,
    create_claude_client,
    create_openai_client,
    get_model_format,
    initialize_custom_models,
    is_antigravity_model,
    is_claude_code_model,
    is_codex_model,
    is_gemini_model,
)
from .codex import handle_codex_request
from .config import config, setup_logging
from .converters import (
    GeminiConverter,
    GeminiStreamingConverter,
    OpenAIConverter,
    OpenAIToAnthropicStreamingConverter,
    convert_anthropic_streaming_with_usage_tracking,
    convert_gemini_streaming_response_to_anthropic,
    convert_openai_response_to_anthropic,
    convert_openai_streaming_response_to_anthropic,
)
from .gemini import handle_gemini_request
from .hook import hook_manager, load_all_plugins
from .types import (
    ClaudeMessagesRequest,
    ClaudeTokenCountRequest,
    ClaudeTokenCountResponse,
    global_usage_stats,
)
from .utils import (
    _extract_error_details,
    _format_error_message,
    log_openai_api_error,
    sanitize_anthropic_messages,
    sanitize_openai_messages,
    sanitize_openai_request,
    update_global_usage_stats,
)

logger = logging.getLogger(__name__)


def extract_api_key(request: Request) -> str | None:
    """Extract API key from request Authorization header.

    Claude Code sends the API key via Authorization header when ANTHROPIC_AUTH_TOKEN is set.
    The format is: "Bearer <api_key>" or just the key directly.

    Args:
        request: FastAPI Request object

    Returns:
        The API key extracted from the Authorization header, or None if not found.
        When None is returned, the caller should check if the model has its own API key.
    """
    auth_header = request.headers.get("Authorization") or request.headers.get("x-api-key")

    if not auth_header:
        return None

    # Handle "Bearer <token>" format
    if auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix

    # Handle plain token
    return auth_header



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan event handler."""
    # Startup
    initialize_custom_models()
    load_all_plugins()

    yield
    # Shutdown (if needed)


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path

    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")

    # Process the request and get the response
    response = await call_next(request)

    return response


@app.middleware("http")
async def handle_midstream_abort(request: Request, call_next):
    """
    Handle MidStreamAbort exceptions to simulate real API connection drops.

    When upstream API errors occur mid-stream, the real API drops the TCP
    connection. This middleware catches MidStreamAbort and returns a response
    that closes the connection cleanly without logging scary tracebacks.

    This triggers client-side retry logic as expected by downstream agents.
    """
    from .midstream_abort import MidStreamAbort

    try:
        response = await call_next(request)
        return response
    except MidStreamAbort as e:
        # Log clean one-liner instead of full traceback
        logger.info(f"[mid-stream abort] {e}")
        # Re-raise to abort the connection (triggers client retry)
        raise


async def hook_streaming_response(response_generator, request, model_id):
    """Wraps a streaming response generator to apply response hooks to each event.

    Handles raw chunks from aiter_text() that may contain multiple SSE events
    or partial events split across chunks.
    """
    buffer = ""

    async for chunk in response_generator:
        # Accumulate chunks in buffer
        buffer += chunk

        # Process complete events (separated by double newline)
        while "\n\n" in buffer:
            event_str, buffer = buffer.split("\n\n", 1)
            event_str += "\n\n"  # Add back the delimiter

            # Process this complete event
            processed = _process_single_sse_event(event_str)
            yield processed

    # Yield any remaining buffer content
    if buffer.strip():
        yield buffer


def _process_single_sse_event(event_str: str) -> str:
    """Process a single SSE event string through hooks."""
    try:
        # Skip events without data
        if "data: " not in event_str:
            return event_str

        # Split event string to process data part
        header, _, data_part = event_str.partition("data: ")

        # Handle [DONE] marker
        if data_part.strip() == "[DONE]":
            return event_str

        # Skip empty data
        if not data_part.strip():
            return event_str

        # Parse JSON, apply hooks, and reconstruct
        data_dict = json.loads(data_part.strip())
        hooked_data_dict = hook_manager.trigger_response_hooks(data_dict)
        new_data_str = json.dumps(hooked_data_dict)

        return f"{header}data: {new_data_str}\n\n"

    except json.JSONDecodeError:
        # Pass through unparseable events unchanged
        return event_str


def parse_claude_api_error(response: httpx.Response) -> dict:
    """Parse Claude API error response and extract structured error details."""
    try:
        # Try to parse the JSON response
        error_data = response.json()

        # Extract Claude API error structure
        if isinstance(error_data, dict) and "error" in error_data:
            claude_error = error_data["error"]
            if isinstance(claude_error, dict):
                error_type = claude_error.get("type", "unknown_error")
                error_message = claude_error.get("message", "Unknown error occurred")

                return {
                    "status_code": response.status_code,
                    "error_type": error_type,
                    "error_message": error_message,
                    "full_response": error_data
                }

        # Fallback for non-standard error format
        return {
            "status_code": response.status_code,
            "error_type": "http_error",
            "error_message": f"HTTP {response.status_code}: {response.reason_phrase}",
            "full_response": error_data if isinstance(error_data, dict) else {"raw_response": str(error_data)}
        }

    except (json.JSONDecodeError, ValueError):
        # Handle non-JSON responses
        try:
            response_text = response.text
        except Exception:
            response_text = "Unable to read response"

        return {
            "status_code": response.status_code,
            "error_type": "http_error",
            "error_message": f"HTTP {response.status_code}: {response.reason_phrase}",
            "full_response": {"raw_response": response_text}
        }


async def handle_direct_claude_request(
    request: ClaudeMessagesRequest,
    model_id: str,
    model_config: dict,
    raw_request: Request
):
    """Handle direct Claude API requests without OpenAI conversion."""
    try:
        # Extract API key from request header (may be None if model has its own key)
        api_key = extract_api_key(raw_request)

        # Create Claude client - uses model-specific key if available, otherwise header key
        claude_client = create_claude_client(model_id, api_key)

        # Prepare request payload - use original Claude format
        claude_request_data = request.model_dump(exclude_none=True)

        # Sanitize messages to fix orphaned tool_use and empty content issues
        if "messages" in claude_request_data:
            claude_request_data["messages"] = sanitize_anthropic_messages(
                claude_request_data["messages"]
            )

        # Update model name to the actual model name for the API call
        claude_request_data["model"] = model_config.get("model_name", model_id)

        # Handle max_tokens limits
        max_tokens = min(model_config.get("max_tokens", 8192), request.max_tokens)
        claude_request_data["max_tokens"] = max_tokens

        # Log the request
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            f"{model_id} (DIRECT)",
            claude_request_data.get("model"),
            len(claude_request_data["messages"]),
            num_tools,
            200,
        )

        # Handle streaming mode
        if request.stream:
            # Determine message type for consistent logging
            message_type = determine_message_type(request)
            logger.info(f"🔗 DIRECT STREAMING: Type={message_type}, Source-Model={model_id}, Target-Model={model_config.get('model_name')}")

            async def direct_streaming_generator():
                try:
                    async with claude_client.stream("POST", "/messages", json=claude_request_data) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_text():
                            if chunk:
                                yield chunk

                except httpx.HTTPStatusError as http_err:
                    # Parse Claude API error response for streaming
                    error_details = parse_claude_api_error(http_err.response)
                    logger.error(f"Claude API streaming error: {json.dumps(error_details, indent=2)}")

                    # Send structured error event
                    error_event = {
                        "type": "error",
                        "error": {
                            "type": error_details["error_type"],
                            "message": error_details["error_message"]
                        }
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

                except httpx.ConnectError as conn_err:
                    logger.error(f"Connection error in streaming: {conn_err}")
                    error_event = {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"Unable to connect to Claude API: {conn_err}"
                        }
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

                except httpx.TimeoutException as timeout_err:
                    logger.error(f"Timeout error in streaming: {timeout_err}")
                    error_event = {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"Request to Claude API timed out: {timeout_err}"
                        }
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

                except Exception as e:
                    logger.error(f"Unexpected streaming error: {e}")
                    error_event = {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": f"Unexpected error: {str(e)}"
                        }
                    }
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                finally:
                    await claude_client.aclose()

            # Wrap with usage tracking
            tracked_stream = convert_anthropic_streaming_with_usage_tracking(
                direct_streaming_generator(),
                request,
                model_id,
            )

            return StreamingResponse(
                tracked_stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        else:
            # Non-streaming mode
            # Determine message type for consistent logging
            message_type = determine_message_type(request)
            logger.info(f"🔗 DIRECT REQUEST: Type={message_type}, Source-Model={model_id}, Target-Model={model_config.get('model_name')}")
            start_time = time.time()

            try:
                response = await claude_client.post("/messages", json=claude_request_data)
                response.raise_for_status()

                logger.debug(
                    f"✅ DIRECT RESPONSE RECEIVED: Model={claude_request_data.get('model')}, Time={time.time() - start_time:.2f}s"
                )

                # Safely parse response JSON
                try:
                    response_data = response.json()
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse Claude API response JSON: {json_err}")
                    raise HTTPException(
                        status_code=502,
                        detail="Invalid JSON response from Claude API"
                    ) from json_err

                # Apply response hooks if needed
                hooked_response_data = hook_manager.trigger_response_hooks(response_data)

                # Update global usage statistics if usage data is present
                if "usage" in hooked_response_data:
                    from .types import ClaudeUsage
                    usage = ClaudeUsage(**hooked_response_data["usage"])
                    update_global_usage_stats(usage, model_id, "Direct Mode")

                return JSONResponse(content=hooked_response_data)

            except httpx.HTTPStatusError as http_err:
                # Parse Claude API error response
                error_details = parse_claude_api_error(http_err.response)

                # Log detailed error information for debugging
                debug_info = {
                    "model_id": model_id,
                    "api_base": model_config.get("api_base", "unknown"),
                    "status_code": error_details["status_code"],
                    "error_type": error_details["error_type"],
                    "error_message": error_details["error_message"]
                }
                logger.error(f"Claude API error: {json.dumps(debug_info, indent=2)}")

                # Create user-friendly error message
                error_type_prefixes = {
                    "rate_limit_error": "Rate limit exceeded",
                    "authentication_error": "Authentication failed",
                    "permission_error": "Permission denied",
                    "invalid_request_error": "Invalid request",
                }
                prefix = error_type_prefixes.get(error_details["error_type"], "Claude API error")
                error_message = f"{prefix}: {error_details['error_message']}"

                raise HTTPException(
                    status_code=error_details["status_code"],
                    detail=error_message
                ) from http_err

            except httpx.ConnectError as conn_err:
                logger.error(f"Connection error to Claude API: {conn_err}")
                raise HTTPException(
                    status_code=502,
                    detail="Unable to connect to Claude API. Please check your network connection and try again."
                ) from conn_err

            except httpx.TimeoutException as timeout_err:
                logger.error(f"Timeout error to Claude API: {timeout_err}")
                raise HTTPException(
                    status_code=504,
                    detail="Request to Claude API timed out. Please try again."
                ) from timeout_err

            except HTTPException:
                # Re-raise HTTPExceptions (from JSON parsing error above)
                raise

            except Exception as e:
                # Generic error fallback
                logger.error(f"Unexpected error in direct Claude request: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected error occurred: {str(e)}"
                ) from e
            finally:
                await claude_client.aclose()

    except HTTPException:
        # Re-raise HTTPExceptions that we've already handled properly
        raise
    except Exception as e:
        # Fallback for any unhandled exceptions in the direct Claude request setup
        logger.error(f"Unhandled error in direct Claude request setup: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        ) from e


@app.post("/anthropic/v1/messages")
async def create_message(raw_request: Request):
    try:
        # Extract API key from request header (passed by ccproxy)
        api_key = extract_api_key(raw_request)

        # Use Pydantic's optimized JSON validation directly from raw bytes
        body = await raw_request.body()
        request = ClaudeMessagesRequest.model_validate_json(body, strict=False)
        model_id = request.model

        # Check if thinking is enabled
        has_thinking = False
        if request.thinking is not None:
            has_thinking = request.thinking.type == "enabled"
            logger.debug(
                f"🧠 Thinking type check: {request.thinking.type}, enabled: {has_thinking}"
            )
        logger.debug(f"🧠 Final thinking decision: has_thinking={has_thinking}")

        if model_id not in CUSTOM_OPENAI_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_id}'. Check models.yaml for available models.",
            )

        model_config = CUSTOM_OPENAI_MODELS[model_id]
        logger.debug(f"model config: {model_config}")

        # Check if we have an API key (either from header or model config)
        if model_config.get("api_key"):
            logger.debug(f"Using model-specific API key for {model_id}")
        elif not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Either provide it in the Authorization header or configure it in models.yaml."
            )

        # Determine message type based on content
        message_type = determine_message_type(request)

        logger.info(
            f"📊 REQUEST: Type={message_type}, Source-Model={model_id}, Target-Model={model_config.get('model_name')}, Stream={model_config.get('can_stream')}"
        )

        model_format = get_model_format(model_id) or "openai"

        # Check for Claude Code subscription models first (special anthropic format)
        if is_claude_code_model(model_id):
            from .claude_code import ClaudeCodeErrorResponse

            provider_model_name = model_config.get("model_name", model_id)
            logger.info(f"🔗 CLAUDE CODE FORMAT: Model={model_id}, Target={provider_model_name}")

            # Make the request - may return error response or streaming generator
            result = await handle_claude_code_request(
                request,
                model_id,
                model_name=provider_model_name,
            )

            # Check if it's an error response - return with correct HTTP status
            if isinstance(result, ClaudeCodeErrorResponse):
                logger.info(f"🔗 CLAUDE CODE ERROR: status={result.status_code}")
                return result.to_json_response()

            # Success - wrap streaming generator with usage tracking
            tracked_stream = convert_anthropic_streaming_with_usage_tracking(
                result,
                request,
                model_id,
            )

            hooked_generator = hook_streaming_response(
                tracked_stream,
                request,
                model_id,
            )
            return StreamingResponse(
                hooked_generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        if model_format == "anthropic":
            logger.info(f"🔗 ANTHROPIC FORMAT: Type={message_type}, Source-Model={model_id}, Target-Model={model_config.get('model_name')}")
            return await handle_direct_claude_request(
                request, model_id, model_config, raw_request
            )

        if model_format == "gemini":
            provider_model_name = model_config.get("model_name", model_id)
            if is_gemini_model(model_id):
                logger.info(f"🔗 GEMINI FORMAT: Model={model_id}")
                provider_generator = handle_gemini_request(
                    request,
                    model_id,
                    model_name=provider_model_name,
                )
            elif is_antigravity_model(model_id):
                logger.info(f"🔗 ANTIGRAVITY FORMAT: Model={model_id}")
                provider_generator = handle_antigravity_request(
                    request,
                    model_id,
                    model_name=provider_model_name,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="format=gemini requires provider=gemini or provider=antigravity.",
                )

            hooked_generator = hook_streaming_response(
                convert_gemini_streaming_response_to_anthropic(
                    provider_generator,
                    request,
                    model_id,
                ),
                request,
                model_id,
            )
            return StreamingResponse(
                hooked_generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        if model_format != "openai":
            raise HTTPException(
                status_code=400,
                detail=f"Unknown format '{model_format}' for model '{model_id}'.",
            )

        # Convert Anthropic request to OpenAI format (default path)
        openai_request = request.to_openai_request()
        openai_request['store'] = False

        # GROQ DEBUG: Log the OpenAI request for tool call debugging
        if request.tools:
            logger.debug(f"🔧 TOOL_DEBUG: Original request has {len(request.tools)} tools")
            logger.debug(f"🔧 TOOL_DEBUG: Tools: {[tool.name for tool in request.tools]}")

        if openai_request.get('tools'):
            logger.debug(f"🔧 TOOL_DEBUG: OpenAI request has {len(openai_request['tools'])} tools")
            logger.debug(f"🔧 TOOL_DEBUG: OpenAI tools: {openai_request['tools']}")
            logger.debug(f"🔧 TOOL_DEBUG: tool_choice: {openai_request.get('tool_choice')}")

        # Trigger request hooks
        openai_request = hook_manager.trigger_request_hooks(openai_request)

        # Check for Codex/Gemini/Antigravity provider
        provider_generator = None
        if is_codex_model(model_id):
            logger.info(f"🔗 CODEX MODE: Model={model_id}")
            openai_request["model"] = model_config.get("model_name", model_id)
            provider_generator = await handle_codex_request(openai_request, model_id)

        if provider_generator:
            hooked_generator = hook_streaming_response(
                convert_openai_streaming_response_to_anthropic(
                    provider_generator, request, model_id
                ),
                request,
                model_id,
            )
            return StreamingResponse(
                hooked_generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        # Create OpenAI client for the model with the extracted API key
        client = create_openai_client(model_id, api_key)
        openai_request["model"] = model_config.get("model_name")

        # Add extra headers if defined in model config
        openai_request["extra_headers"] = model_config["extra_headers"]
        openai_request["extra_body"] = model_config["extra_body"]
        openai_request["temperature"] = model_config["temperature"]

        # Handle thinking/reasoning based on model capabilities

        # 1. OpenAI native `reasoning_effort`
        reasoning_effort = model_config.get("reasoning_effort")
        if reasoning_effort in ["minimal", "low", "medium", "high"]:
            openai_request["reasoning_effort"] = reasoning_effort

        # 2. Custom `thinkingConfig` in `extra_body` for Gemini-style thinking
        if model_config.get("extra_body") and "thinkingConfig" in model_config["extra_body"]:
            # doc https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerationConfig#ThinkingConfig
            # Start with the base thinking configuration from the model
            thinking_params = model_config["extra_body"]["thinkingConfig"].copy()
            if has_thinking:
                # If thinking is enabled but no budget is specified, default to dynamic.
                if "thinkingBudget" not in thinking_params:
                    thinking_params["thinkingBudget"] = -1
            else:
                # To disable thinking for Gemini, set the budget to 0.
                thinking_params["thinkingBudget"] = 0

            openai_request["extra_body"]["thinkingConfig"] = thinking_params

        # Intelligent tool_choice adjustment for better model consistency
        # Based on test findings from claude_code_interruption_test:
        # - Claude models naturally tend to use tools in interruption/verification scenarios
        # - Other models (DeepSeek, etc.) may not use tools when tool_choice is None or auto
        # - tool_choice=required ensures consistent behavior across all models
        # - Exception: Thinking models don't support tool_choice=required (API limitation)
        # IMPORTANT: Only set tool_choice if we actually have tools
        if (
            openai_request.get("tools")
            and len(openai_request.get("tools", [])) > 0
        ):
            openai_request["tool_choice"] = "auto"
            logger.debug("🔧 TOOL_DEBUG: Set tool_choice to 'auto' because tools are present")
        else:
            # Remove tool_choice if no tools are present (OpenAI API requirement)
            if "tool_choice" in openai_request:
                del openai_request["tool_choice"]
                logger.debug("🔧 TOOL_DEBUG: Removed tool_choice because no tools are present")

        # Only log basic info about the request, not the full details
        logger.debug(
            f"Request for model: {openai_request.get('model')},stream: {openai_request.get('stream', False)},thinking_mode:{openai_request['extra_body'].get('thinking')}"
        )

        # Use OpenAI SDK for streaming
        num_tools = len(request.tools) if request.tools else 0

        log_request_beautifully(
            "POST",
            raw_request.url.path,
            model_id,
            openai_request.get("model"),
            len(openai_request["messages"]),
            num_tools,
            200,  # Assuming success at this point
        )

        # Build complete request with OpenAI SDK type validation
        # Handle max_tokens for custom models vs standard models
        max_tokens = min(model_config.get("max_tokens"), request.max_tokens)
        openai_request["max_tokens"] = max_tokens

        openai_request["stream"] = model_config["can_stream"]

        # Add stream_options to include usage data in streaming responses
        if openai_request["stream"]:
            openai_request["stream_options"] = {"include_usage": True}

        # Sanitize request: remove unsupported fields and fix message ordering
        sanitize_openai_request(openai_request)
        if "messages" in openai_request:
            openai_request["messages"] = sanitize_openai_messages(openai_request["messages"])

        # Handle streaming mode
        # Use OpenAI SDK async streaming
        if openai_request["stream"]:
            logger.debug(f"🔧 TOOL_DEBUG: Starting streaming request with model: {openai_request.get('model')}")

            # Log the actual request being sent to help debug
            logger.debug(f"🔧 TOOL_DEBUG: Full OpenAI request: {openai_request}")

            try:
                response_generator: AsyncStream[
                    ChatCompletionChunk
                ] = await client.chat.completions.create(**openai_request)
                logger.debug("🔧 TOOL_DEBUG: Successfully created streaming response generator")

                # Convert OpenAI chunks to Anthropic SSE, then apply hooks
                hooked_generator = hook_streaming_response(
                    convert_openai_streaming_response_to_anthropic(
                        response_generator, request, model_id
                    ),
                    request,
                    model_id,
                )
                return StreamingResponse(
                    hooked_generator,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except Exception as e:
                logger.error(f"TOOL_DEBUG: Error creating streaming response: {e}")
                logger.error(f"TOOL_DEBUG: Request details: model={openai_request.get('model')}, tools={len(openai_request.get('tools', []))}")
                log_openai_api_error(e, "streaming")
                raise
        else:
            start_time = time.time()
            try:
                logger.debug(f"🔧 TOOL_DEBUG: Making non-streaming request with model: {openai_request.get('model')}")
                openai_response: ChatCompletion = await client.chat.completions.create(
                    **openai_request
                )
                logger.debug("🔧 TOOL_DEBUG: Successfully received non-streaming response")

                # GROQ DEBUG: Log the raw OpenAI response for tool call analysis
                if hasattr(openai_response, 'choices') and openai_response.choices:
                    choice = openai_response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                        logger.debug(f"🔧 TOOL_DEBUG: Non-streaming response has {len(choice.message.tool_calls)} tool calls")
                        for i, tc in enumerate(choice.message.tool_calls):
                            logger.debug(f"🔧 TOOL_DEBUG: Tool call {i}: {tc}")

            except Exception as e:
                error_context = {
                    "model_id": model_id,
                    "request_model": openai_request.get("model"),
                    "api_base": str(getattr(client, "base_url", "unknown")),
                    "error_type": type(e).__name__,
                }
                logger.error(f"TOOL_DEBUG: API call failed with context: {json.dumps(error_context, indent=2)}")
                log_openai_api_error(e, "non-streaming")
                if openai_request.get('tools'):
                    logger.error(f"TOOL_DEBUG: Error occurred with {len(openai_request['tools'])} tools present")
                raise

            logger.debug(
                f"✅ RESPONSE RECEIVED: Model={openai_request.get('model')}, Time={time.time() - start_time:.2f}s"
            )

            # Convert OpenAI response to Anthropic format
            anthropic_response = convert_openai_response_to_anthropic(
                openai_response, request
            )

            # --- HOOK: Trigger response hooks ---
            response_dict = anthropic_response.model_dump(exclude_none=True)
            hooked_response_dict = hook_manager.trigger_response_hooks(response_dict)
            # ------------------------------------

            # Update global usage statistics and log usage information
            update_global_usage_stats(
                anthropic_response.usage, model_id, "Non-streaming"
            )

            return JSONResponse(content=hooked_response_dict)

    except Exception as e:
        error_details = _extract_error_details(e)
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")

        error_message = _format_error_message(e, error_details)
        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message) from e


@app.post("/openai/v1/chat/completions")
async def create_chat_completion(raw_request: Request):
    """
    OpenAI-compatible Chat Completions endpoint.

    Accepts requests in OpenAI format, converts to Anthropic format internally,
    routes through the configured model, and returns the response in OpenAI format.
    """
    try:
        # Extract API key from request header
        api_key = extract_api_key(raw_request)

        # Parse OpenAI request body
        body = await raw_request.body()
        openai_payload = json.loads(body)

        model_id = openai_payload.get("model", "")
        stream = openai_payload.get("stream", False)

        if model_id not in CUSTOM_OPENAI_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_id}'. Check models.yaml for available models.",
            )

        model_config = CUSTOM_OPENAI_MODELS[model_id]

        # Check API key
        if not model_config.get("api_key") and not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Either provide it in the Authorization header or configure it in models.yaml."
            )

        # Convert OpenAI request to Anthropic format
        openai_converter = OpenAIConverter()
        anthropic_request = openai_converter.request_to_anthropic(openai_payload)
        # Override model to use the configured model_id for routing
        anthropic_request.model = model_id

        num_tools = len(anthropic_request.tools) if anthropic_request.tools else 0
        logger.info(
            f"📊 OPENAI->ANTHROPIC: Model={model_id}, Target={model_config.get('model_name')}, Stream={stream}, Tools={num_tools}"
        )

        # Route through the main pipeline - call the internal handler directly
        # We need to reconstruct the request handling logic here
        model_format = get_model_format(model_id) or "openai"

        if model_format == "anthropic":
            # Direct Anthropic format
            response = await handle_direct_claude_request(
                anthropic_request, model_id, model_config, raw_request
            )

            # For streaming, wrap the response to convert back to OpenAI format
            if stream and isinstance(response, StreamingResponse):
                async def convert_streaming_to_openai():
                    streaming_converter = OpenAIToAnthropicStreamingConverter()
                    # Get the original body iterator
                    body_iterator = response.body_iterator
                    async for chunk in streaming_converter.stream_from_anthropic(
                        body_iterator, model=model_config.get("model_name", model_id)
                    ):
                        yield chunk

                return StreamingResponse(
                    convert_streaming_to_openai(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            elif isinstance(response, JSONResponse):
                # Non-streaming: convert response to OpenAI format
                anthropic_response_data = json.loads(response.body.decode())
                from .types import ClaudeMessagesResponse
                anthropic_response = ClaudeMessagesResponse.model_validate(anthropic_response_data)
                openai_response = openai_converter.response_from_anthropic(anthropic_response)
                return JSONResponse(content=openai_response)
            return response

        elif model_format == "gemini":
            # Gemini/Antigravity format
            provider_model_name = model_config.get("model_name", model_id)
            if is_gemini_model(model_id):
                provider_generator = handle_gemini_request(
                    anthropic_request,
                    model_id,
                    model_name=provider_model_name,
                )
            elif is_antigravity_model(model_id):
                provider_generator = handle_antigravity_request(
                    anthropic_request,
                    model_id,
                    model_name=provider_model_name,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="format=gemini requires provider=gemini or provider=antigravity.",
                )

            # Convert Gemini -> Anthropic -> OpenAI streaming
            anthropic_stream = convert_gemini_streaming_response_to_anthropic(
                provider_generator,
                anthropic_request,
                model_id,
            )

            streaming_converter = OpenAIToAnthropicStreamingConverter()
            openai_stream = streaming_converter.stream_from_anthropic(
                anthropic_stream, model=model_config.get("model_name", model_id)
            )

            return StreamingResponse(
                openai_stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )

        else:
            # OpenAI format - pass through with model name substitution
            # This is the simplest case: OpenAI -> OpenAI
            client = create_openai_client(model_id, api_key)
            openai_payload["model"] = model_config.get("model_name", model_id)
            openai_payload["extra_headers"] = model_config.get("extra_headers", {})
            openai_payload["extra_body"] = model_config.get("extra_body", {})

            # Handle max_tokens
            max_tokens = openai_payload.get("max_tokens") or openai_payload.get("max_completion_tokens")
            if max_tokens:
                max_tokens = min(model_config.get("max_tokens", 8192), max_tokens)
                openai_payload["max_tokens"] = max_tokens

            # Sanitize payload: remove fields not supported by OpenAI SDK
            # These fields may be sent by clients (e.g., thinking for Anthropic compatibility)
            sanitize_openai_request(openai_payload)

            # Sanitize messages to fix tool call ordering issues
            if "messages" in openai_payload:
                openai_payload["messages"] = sanitize_openai_messages(openai_payload["messages"])

            if stream:
                openai_payload["stream_options"] = {"include_usage": True}
                response_generator = await client.chat.completions.create(**openai_payload)
                return StreamingResponse(
                    _forward_openai_stream(response_generator),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            else:
                response = await client.chat.completions.create(**openai_payload)
                return JSONResponse(content=response.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        error_details = _extract_error_details(e)
        logger.error(f"Error in /v1/chat/completions: {json.dumps(error_details, indent=2)}")
        error_message = _format_error_message(e, error_details)
        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message) from e


async def _forward_openai_stream(stream):
    """Forward OpenAI streaming response as SSE."""
    async for chunk in stream:
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/gemini/v1beta/models/{model_path:path}:generateContent")
async def gemini_generate_content(model_path: str, raw_request: Request):
    """
    Gemini-compatible GenerateContent endpoint (non-streaming).

    Accepts requests in Gemini format, converts to Anthropic format internally,
    routes through the configured model, and returns the response in Gemini format.
    """
    return await _handle_gemini_request(model_path, raw_request, streaming=False)


@app.post("/gemini/v1beta/models/{model_path:path}:streamGenerateContent")
async def gemini_stream_generate_content(model_path: str, raw_request: Request):
    """
    Gemini-compatible StreamGenerateContent endpoint.

    Accepts requests in Gemini format, converts to Anthropic format internally,
    routes through the configured model, and returns streaming response in Gemini format.
    """
    return await _handle_gemini_request(model_path, raw_request, streaming=True)


async def _handle_gemini_request(model_path: str, raw_request: Request, streaming: bool):
    """Common handler for Gemini generateContent and streamGenerateContent."""
    try:
        # Extract API key from request header
        api_key = extract_api_key(raw_request)

        # Parse Gemini request body
        body = await raw_request.body()
        gemini_payload = json.loads(body)

        # Extract model_id from path (e.g., "gemini-1.5-pro" from "gemini-1.5-pro:generateContent")
        model_id = model_path
        if ":" in model_id:
            model_id = model_id.split(":")[0]

        # Add model to payload if not present
        if "model" not in gemini_payload:
            gemini_payload["model"] = model_id

        if model_id not in CUSTOM_OPENAI_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{model_id}'. Check models.yaml for available models.",
            )

        model_config = CUSTOM_OPENAI_MODELS[model_id]

        # Check API key
        if not model_config.get("api_key") and not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Either provide it in the Authorization header or configure it in models.yaml."
            )

        # Convert Gemini request to Anthropic format
        gemini_converter = GeminiConverter()
        anthropic_request = gemini_converter.request_to_anthropic(gemini_payload)
        # Override model to use the configured model_id for routing
        anthropic_request.model = model_id
        # Set stream flag
        anthropic_request.stream = streaming

        num_tools = len(anthropic_request.tools) if anthropic_request.tools else 0
        logger.info(
            f"📊 GEMINI->ANTHROPIC: Model={model_id}, Target={model_config.get('model_name')}, Stream={streaming}, Tools={num_tools}"
        )

        # Route through the main pipeline
        model_format = get_model_format(model_id) or "openai"

        if model_format == "anthropic":
            # Direct Anthropic format
            response = await handle_direct_claude_request(
                anthropic_request, model_id, model_config, raw_request
            )

            if streaming and isinstance(response, StreamingResponse):
                # Convert streaming response to Gemini format
                streaming_converter = GeminiStreamingConverter()
                body_iterator = response.body_iterator
                gemini_stream = streaming_converter.stream_from_anthropic(
                    body_iterator, model=model_config.get("model_name", model_id)
                )
                return StreamingResponse(
                    gemini_stream,
                    media_type="application/x-ndjson",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            elif isinstance(response, JSONResponse):
                # Convert non-streaming response to Gemini format
                anthropic_response_data = json.loads(response.body.decode())
                from .types import ClaudeMessagesResponse
                anthropic_response = ClaudeMessagesResponse.model_validate(anthropic_response_data)
                gemini_response = gemini_converter.response_from_anthropic(anthropic_response)
                return JSONResponse(content=gemini_response)
            return response

        elif model_format == "gemini":
            # Gemini/Antigravity format - pass through with conversion
            provider_model_name = model_config.get("model_name", model_id)
            if is_gemini_model(model_id):
                provider_generator = handle_gemini_request(
                    anthropic_request,
                    model_id,
                    model_name=provider_model_name,
                )
            elif is_antigravity_model(model_id):
                provider_generator = handle_antigravity_request(
                    anthropic_request,
                    model_id,
                    model_name=provider_model_name,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="format=gemini requires provider=gemini or provider=antigravity.",
                )

            # Convert Gemini -> Anthropic -> Gemini (for consistent format)
            anthropic_stream = convert_gemini_streaming_response_to_anthropic(
                provider_generator,
                anthropic_request,
                model_id,
            )

            streaming_converter = GeminiStreamingConverter()
            gemini_stream = streaming_converter.stream_from_anthropic(
                anthropic_stream, model=model_config.get("model_name", model_id)
            )

            return StreamingResponse(
                gemini_stream,
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # OpenAI format - convert to OpenAI, then back to Gemini
            openai_converter = OpenAIConverter()
            client = create_openai_client(model_id, api_key)
            openai_request = openai_converter.request_from_anthropic(anthropic_request)
            openai_request["model"] = model_config.get("model_name", model_id)
            openai_request["extra_headers"] = model_config.get("extra_headers", {})
            openai_request["extra_body"] = model_config.get("extra_body", {})

            # Handle max_tokens
            max_tokens = model_config.get("max_tokens", 4096)
            openai_request["max_tokens"] = min(max_tokens, anthropic_request.max_tokens)

            if streaming:
                openai_request["stream"] = True
                openai_request["stream_options"] = {"include_usage": True}
                response_generator = await client.chat.completions.create(**openai_request)

                # OpenAI -> Anthropic -> Gemini streaming
                anthropic_stream = convert_openai_streaming_response_to_anthropic(
                    response_generator, anthropic_request, model_id
                )
                streaming_converter = GeminiStreamingConverter()
                gemini_stream = streaming_converter.stream_from_anthropic(
                    anthropic_stream, model=model_config.get("model_name", model_id)
                )

                return StreamingResponse(
                    gemini_stream,
                    media_type="application/x-ndjson",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                openai_request["stream"] = False
                response = await client.chat.completions.create(**openai_request)
                # OpenAI -> Anthropic -> Gemini response
                anthropic_response = openai_converter.response_to_anthropic(response, anthropic_request)
                gemini_response = gemini_converter.response_from_anthropic(anthropic_response)
                return JSONResponse(content=gemini_response)

    except HTTPException:
        raise
    except Exception as e:
        error_details = _extract_error_details(e)
        logger.error(f"Error in Gemini endpoint: {json.dumps(error_details, indent=2)}")
        error_message = _format_error_message(e, error_details)
        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message) from e


@app.post("/anthropic/v1/messages/count_tokens")
async def count_tokens(raw_request: Request):
    fallback_response = ClaudeTokenCountResponse(input_tokens=1000)
    try:
        # Use Pydantic's optimized JSON validation directly from raw bytes
        body = await raw_request.body()
        request = ClaudeTokenCountRequest.model_validate_json(body, strict=False)
    except Exception as e:
        logger.warning(f"Token count request parsing failed; returning fallback: {e}")
        return fallback_response

    display_model = request.model
    if "/" in display_model:
        display_model = display_model.split("/")[-1]

    num_tools = len(request.tools) if request.tools else 0
    log_request_beautifully(
        "POST",
        raw_request.url.path,
        display_model,
        request.model,
        len(request.messages),
        num_tools,
        200,
    )

    try:
        token_count = request.calculate_tokens()
        return ClaudeTokenCountResponse(input_tokens=token_count)
    except Exception as e:
        logger.error(f"Error in local token counting: {e}")
        return fallback_response


@app.get("/anthropic/v1/stats")
async def get_stats():
    """Returns the comprehensive token usage statistics for the current session."""
    return global_usage_stats.get_session_summary()


@app.post("/anthropic/v1/messages/test_conversion")
async def test_message_conversion(raw_request: Request):
    """
    Test endpoint for direct message format conversion.

    This endpoint converts Anthropic format to OpenAI format and sends the request
    directly to the specified model without any server-side model switching.
    Useful for testing specific model integrations and message format conversion.
    """
    try:
        # Extract API key from request header (may be None if model has its own key)
        api_key = extract_api_key(raw_request)

        # Use Pydantic's optimized JSON validation directly from raw bytes
        body = await raw_request.body()
        request = ClaudeMessagesRequest.model_validate_json(body, strict=False)
        original_model = request.model

        # Check if we have an API key (either from header or model config)
        model_config = CUSTOM_OPENAI_MODELS.get(original_model, {})
        if not model_config.get("api_key") and not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Either provide it in the Authorization header or configure it in models.yaml."
            )

        logger.info(f"🧪 TEST CONVERSION: Direct test for model {original_model}")

        # Convert Anthropic request to OpenAI format
        openai_request = request.to_openai_request()
        openai_request['store'] = False

        # Create OpenAI client - uses model-specific key if available, otherwise header key
        client = create_openai_client(original_model, api_key)
        # model_id -> model_name in CUSTOM_OPENAI_MODELS configs
        openai_request["model"] = CUSTOM_OPENAI_MODELS[request.model]["model_name"]

        # Add stream_options to include usage data in streaming responses
        if request.stream:
            openai_request["stream_options"] = {"include_usage": True}

        logger.debug(
            f"🧪 Converted request for {original_model}: {json.dumps({k: v for k, v in openai_request.items() if k != 'messages'}, indent=2)}"
        )

        # Log the request
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            "/v1/messages/test_conversion",
            f"{original_model} (DIRECT TEST)",
            openai_request.get("model"),
            len(openai_request["messages"]),
            num_tools,
            200,
        )

        # Handle streaming mode
        if request.stream:
            logger.info(f"🧪 Starting direct streaming test for {original_model}")
            response_generator = await client.chat.completions.create(**openai_request)
            return StreamingResponse(
                convert_openai_streaming_response_to_anthropic(
                    response_generator, request, original_model
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        else:
            # Regular completion
            logger.info(f"🧪 Starting direct completion test for {original_model}")
            start_time = time.time()
            openai_response = await client.chat.completions.create(**openai_request)
            logger.info(f"🧪 Direct test completed in {time.time() - start_time:.2f}s")

            # Convert OpenAI response to Anthropic format
            anthropic_response = convert_openai_response_to_anthropic(
                openai_response, request
            )
            return anthropic_response

    except Exception as e:
        error_details = _extract_error_details(e)
        logger.error(
            f"🧪 Error in test conversion: {json.dumps(error_details, indent=2)}"
        )

        error_message = _format_error_message(e, error_details)
        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message) from e


@app.get("/test-connection")
async def test_connection():
    """Test API connectivity to configured providers"""
    test_results = {}
    overall_status = "success"

    # Test custom models
    if CUSTOM_OPENAI_MODELS:
        test_results["custom_models"] = {
            "status": "configured",
            "count": len(CUSTOM_OPENAI_MODELS),
            "models": list(CUSTOM_OPENAI_MODELS.keys()),
            "message": f"{len(CUSTOM_OPENAI_MODELS)} custom models configured",
        }
    else:
        test_results["custom_models"] = {
            "status": "not_configured",
            "message": "No custom models configured",
        }

    # Return appropriate status code
    if overall_status == "success":
        return {
            "status": overall_status,
            "message": "API connectivity test completed",
            "timestamp": datetime.now().isoformat(),
            "results": test_results,
        }
    else:
        return JSONResponse(
            status_code=207,  # Multi-status
            content={
                "status": overall_status,
                "message": "Some API tests failed or not configured",
                "timestamp": datetime.now().isoformat(),
                "results": test_results,
            },
        )


# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"


def log_request_beautifully(
    method, path, claude_model, openai_model, num_messages, num_tools, status_code
):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]

    # Format status code
    status_str = (
        f"✓ {status_code} OK"
        if status_code == 200
        else f"✗ {status_code}"
    )

    # Put it all together in a clear, beautiful format
    log_line = f"{method} {endpoint} {status_str}"
    model_line = f"{claude_model} → {openai_display} {num_tools} tools {num_messages} messages"

    # Use logger instead of print to maintain consistency with log levels
    # This will only show up when log level is INFO or lower
    logger.info(log_line)
    logger.info(model_line)


def determine_message_type(request):
    """Determine message type based on request content."""
    message_type = "text"
    if hasattr(request, 'tools') and request.tools:
        message_type = "tool_call"
    elif hasattr(request, 'images') and request.images:
        message_type = "multimodal"
    elif hasattr(request, 'messages'):
        last_message = request.messages[-1].content if request.messages else ""
        if isinstance(last_message, list):
            # Check if any content item is not text
            for item in last_message:
                if hasattr(item, 'type') and item.type != 'text':
                    message_type = "multimodal"
                    break
    return message_type


if __name__ == "__main__":
    # This block is only executed when the script is run directly,
    # not when it's imported by another script.
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run the Claude Code Proxy Server.")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument(
        "--models", type=Path, default=None, help="Path to models.yaml"
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to config.json"
    )
    args = parser.parse_args()

    # Override config from command line
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    # Load custom config file if specified
    if args.config:
        from .config_manager import load_config_file

        config_data = load_config_file(args.config)
        if config_data:
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Load custom models file if specified
    if args.models:
        from .client import load_models_config

        load_models_config(args.models)

    # Re-initialize logging
    setup_logging()

    # Print initial configuration status
    logger.info("Configuration loaded")

    # Run the Server
    uvicorn.run(
        "anthropic_proxy.server:app",
        host=config.host,
        port=config.port,
        log_config=None,
        reload=False,
    )
