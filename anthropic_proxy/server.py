"""
FastAPI server for the anthropic proxy.
This module contains the FastAPI application and API endpoints.
"""

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncStream,
    AuthenticationError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
)

from hook import hook_manager, load_all_plugins

from .client import (
    CUSTOM_OPENAI_MODELS,
    create_openai_client,
    determine_model_by_router,
    initialize_custom_models,
)
from .config import config, setup_env_file_monitoring, setup_logging
from .converter import (
    clean_gemini_schema,
    convert_openai_response_to_anthropic,
)
from .streaming import convert_openai_streaming_response_to_anthropic
from .types import (
    ClaudeMessagesRequest,
    ClaudeTokenCountRequest,
    ClaudeTokenCountResponse,
    global_usage_stats,
)
from .utils import (
    _extract_error_details,
    _format_error_message,
    update_global_usage_stats,
)

logger = logging.getLogger(__name__)

# Global variables for file monitoring
env_file_watcher_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan event handler."""
    # Startup
    # Note: Logging is not configured at this point
    # It will be configured only when running the script directly.
    initialize_custom_models()
    load_all_plugins()
    setup_env_file_monitoring()
    yield
    # Shutdown (if needed)
    global env_file_watcher_task
    if env_file_watcher_task and not env_file_watcher_task.done():
        env_file_watcher_task.cancel()
        try:
            await env_file_watcher_task
        except asyncio.CancelledError:
            pass


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


async def hook_streaming_response(response_generator, request, routed_model):
    """Wraps a streaming response generator to apply response hooks to each event."""
    async for event_str in convert_openai_streaming_response_to_anthropic(
        response_generator, request, routed_model
    ):
        try:
            # The event string is like "event: <type>\ndata: <json>\n\n"
            if "data: " not in event_str:
                yield event_str
                continue

            # Split event string to process data part
            header, _, data_part = event_str.partition("data: ")

            if data_part.strip() == "[DONE]":
                yield event_str
                continue

            if not data_part.strip():
                yield event_str
                continue

            data_dict = json.loads(data_part)
            hooked_data_dict = hook_manager.trigger_response_hooks(data_dict)
            new_data_str = json.dumps(hooked_data_dict)

            yield f"{header}data: {new_data_str}\n\n"

        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Could not parse and hook streaming event. Error: {e}")
            yield event_str


@app.post("/v1/messages")
async def create_message(raw_request: Request):
    try:
        # Use Pydantic's optimized JSON validation directly from raw bytes
        body = await raw_request.body()
        request = ClaudeMessagesRequest.model_validate_json(body, strict=False)
        original_model = request.model

        # Calculate token count for routing decisions
        token_count = request.calculate_tokens()

        # Check if thinking is enabled
        has_thinking = False
        if request.thinking is not None:
            has_thinking = request.thinking.type == "enabled"
            logger.debug(
                f"🧠 Thinking type check: {request.thinking.type}, enabled: {has_thinking}"
            )
        logger.info(f"🧠 Final thinking decision: has_thinking={has_thinking}")

        # Use router to determine the actual model to use
        routed_model = determine_model_by_router(
            original_model, token_count, has_thinking
        )

        # Final routing check: if the token count exceeds the model's max_input_tokens,
        # switch to the long context model as a fallback.
        if routed_model in CUSTOM_OPENAI_MODELS:
            model_config_check = CUSTOM_OPENAI_MODELS[routed_model]
            max_input_for_model = model_config_check.get("max_input_tokens", 0)

            if token_count > max_input_for_model:
                long_context_model = config.router_config["long_context"]
                logger.info(
                    f" ROUTING: Token count {token_count} exceeds max input {max_input_for_model} for '{routed_model}'. "
                    f"Switching to long context model: '{long_context_model}'."
                )
                routed_model = long_context_model
            else:
                logger.info(
                    f" ROUTING: Token count {token_count} is within max input {max_input_for_model} for '{routed_model}'. "
                    f"No change needed."
                )
        else:
            # Fallback for models not in custom config (e.g., standard Anthropic models)
            # using the global threshold.
            if token_count > config.long_context_threshold:
                long_context_model = config.router_config["long_context"]
                logger.info(
                    f" ROUTING: Token count {token_count} exceeds global threshold {config.long_context_threshold}. "
                    f"Switching to long context model: '{long_context_model}'."
                )
                routed_model = long_context_model

        # Most of time it would be default model
        model_config = CUSTOM_OPENAI_MODELS[routed_model]
        logger.debug(f"routed model config: {model_config}")

        if model_config is None:
            raise Exception(f"model {routed_model} not defined")

        # Get the display name for logging, just the model name without provider prefix
        display_model = routed_model

        logger.info(
            f"📊 PROCESSING REQUEST: Original={original_model} → Routed={routed_model}, Tokens={token_count}, Stream={request.stream}"
        )

        # Convert Anthropic request to OpenAI format
        openai_request = request.to_openai_request()

        # Trigger request hooks
        openai_request = hook_manager.trigger_request_hooks(openai_request)

        # Create OpenAI client for the model
        client = create_openai_client(routed_model)
        openai_request["model"] = model_config.get("model_name")

        # Add extra headers if defined in model config
        openai_request["extra_headers"] = model_config["extra_headers"]
        openai_request["extra_body"] = model_config["extra_body"]

        # Handle thinking/reasoning based on model capabilities

        # 1. OpenAI native `reasoning_effort`
        if (
            has_thinking
            and model_config.get("reasoning_effort")
            and model_config["reasoning_effort"] in ["low", "medium", "high"]
        ):
            openai_request["reasoning_effort"] = model_config["reasoning_effort"]

        # 2. Custom `thinking` and `thinkingConfig` in `extra_body`
        if model_config.get("extra_body"):
            # For doubao-style thinking
            # see https://www.volcengine.com/docs/82379/1449737#fa3f44fa
            if model_config["extra_body"].get("thinking") and isinstance(
                model_config["extra_body"].get("thinking"), dict
            ):
                if has_thinking:
                    openai_request["extra_body"]["thinking"] = {"type": "auto"}
                else:
                    # Pass the thinking block but disable it.
                    openai_request["extra_body"]["thinking"] = {"type": "disabled"}

            # For Gemini-style thinking
            if "thinkingConfig" in model_config["extra_body"]:
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
        if (
            not has_thinking
            and openai_request.get("tools")
            and len(openai_request.get("tools", [])) > 0
        ):
            current_tool_choice = openai_request.get("tool_choice")
            if not current_tool_choice:
                openai_request["tool_choice"] = "auto"

        # Only log basic info about the request, not the full details
        logger.debug(
            f"Request for model: {openai_request.get('model')},stream: {openai_request.get('stream', False)},thinking_mode:{openai_request['extra_body'].get('thinking')}"
        )

        # Use OpenAI SDK for streaming
        num_tools = len(request.tools) if request.tools else 0

        log_request_beautifully(
            "POST",
            raw_request.url.path,
            f"{original_model} → {display_model}",
            openai_request.get("model"),
            len(openai_request["messages"]),
            num_tools,
            200,  # Assuming success at this point
        )

        # Build complete request with OpenAI SDK type validation
        # Handle max_tokens for custom models vs standard models
        max_tokens = min(model_config.get("max_tokens"), request.max_tokens)
        openai_request["max_tokens"] = max_tokens

        # Handle streaming mode
        # Use OpenAI SDK async streaming
        if request.stream:
            response_generator: AsyncStream[
                ChatCompletionChunk
            ] = await client.chat.completions.create(**openai_request)
            # Wrap the generator to apply response hooks
            hooked_generator = hook_streaming_response(
                response_generator, request, routed_model
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
        else:
            start_time = time.time()
            openai_response: ChatCompletion = await client.chat.completions.create(
                **openai_request
            )

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
                anthropic_response.usage, routed_model, "Non-streaming"
            )

            return JSONResponse(content=hooked_response_dict)

    except Exception as e:
        error_details = _extract_error_details(e)
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")

        error_message = _format_error_message(e, error_details)
        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message)


@app.post("/v1/messages/count_tokens")
async def count_tokens(raw_request: Request):
    try:
        # Use Pydantic's optimized JSON validation directly from raw bytes
        body = await raw_request.body()
        request = ClaudeTokenCountRequest.model_validate_json(body, strict=False)
        # Log the incoming token count request
        original_model = request.model

        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        # Use the local tiktoken-based function for counting
        try:
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                request.model,
                len(request.messages),
                num_tools,
                200,  # Assuming success at this point
            )

            # Count tokens using the local tiktoken-based function
            token_count = request.calculate_tokens()

            # Return Anthropic-style response
            return ClaudeTokenCountResponse(input_tokens=token_count)

        except Exception as e:
            logger.error(f"Error in local token counting: {e}")
            # Fallback to a simple approximation
            return ClaudeTokenCountResponse(input_tokens=1000)  # Default fallback

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


@app.get("/v1/stats")
async def get_stats():
    """Returns the comprehensive token usage statistics for the current session."""
    return global_usage_stats.get_session_summary()


@app.post("/v1/messages/test_conversion")
async def test_message_conversion(raw_request: Request):
    """
    Test endpoint for direct message format conversion without routing.

    This endpoint converts Anthropic format to OpenAI format and sends the request
    directly to the specified model without going through the intelligent routing system.
    Useful for testing specific model integrations and message format conversion.
    """
    try:
        # Use Pydantic's optimized JSON validation directly from raw bytes
        body = await raw_request.body()
        request = ClaudeMessagesRequest.model_validate_json(body, strict=False)
        original_model = request.model

        logger.info(f"🧪 TEST CONVERSION: Direct test for model {original_model}")

        # Convert Anthropic request to OpenAI format
        openai_request = request.to_openai_request()

        # Create OpenAI client for the model
        client = create_openai_client(original_model)
        # model_id -> model_name in CUSTOM_OPENAI_MODELS configs
        openai_request["model"] = CUSTOM_OPENAI_MODELS[request.model]["model_name"]

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
        raise HTTPException(status_code=status_code, detail=error_message)


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
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"

    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"

    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

    # Format status code
    status_str = (
        f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}"
        if status_code == 200
        else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    )

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"

    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()


if __name__ == "__main__":
    # This block is only executed when the script is run directly,
    # not when it's imported by another script.
    import argparse

    parser = argparse.ArgumentParser(description="Run the Claude Code Proxy Server.")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload on code changes."
    )
    args = parser.parse_args()

    # Re-initialize logging for the main process, especially for reload scenario
    setup_logging()

    # Print initial configuration status
    print(f"✅ Configuration loaded: Providers={config.validate_api_keys()}")
    print(
        f"🔀 Router Config: Default={config.router_config['default']} Background={config.router_config['background']}, Think={config.router_config['think']}, LongContext={config.router_config['long_context']}"
    )

    # Run the Server
    uvicorn.run(
        "anthropic_proxy.server:app",
        host=config.host,
        port=config.port,
        log_config=None,
        reload=args.reload,
    )
