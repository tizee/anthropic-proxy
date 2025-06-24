import json
import logging
import os
import os.path
import sys
import time
from datetime import datetime
from typing import Any

import tiktoken
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    AuthenticationError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
)

from models import (
    ClaudeMessagesRequest,
    ClaudeTokenCountRequest,
    ClaudeTokenCountResponse,
    ModelDefaults,
    convert_openai_response_to_anthropic,
    convert_openai_streaming_response_to_anthropic,
    global_usage_stats,
    update_global_usage_stats,
)

# Load environment variables from .env file
load_dotenv()


class Config:
    """Universal proxy server configuration with intelligent routing"""

    def __init__(self):
        # Router configuration for intelligent model selection
        self.router_config = {
            "background": os.environ.get("ROUTER_BACKGROUND", "deepseek-v3-250324"),
            "think": os.environ.get("ROUTER_THINK", "deepseek-r1-250528"),
            "long_context": os.environ.get("ROUTER_LONG_CONTEXT", "gemini-2.5-pro"),
            "default": os.environ.get("ROUTER_DEFAULT", "deepseek-r1-250324"),
        }

        # Token thresholds
        self.long_context_threshold = int(
            os.environ.get(
                "LONG_CONTEXT_THRESHOLD", str(ModelDefaults.LONG_CONTEXT_THRESHOLD)
            )
        )

        # Server configuration
        self.host = os.environ.get("HOST", ModelDefaults.DEFAULT_HOST)
        self.port = int(os.environ.get("PORT", str(ModelDefaults.DEFAULT_PORT)))
        self.log_level = os.environ.get("LOG_LEVEL", ModelDefaults.DEFAULT_LOG_LEVEL)
        self.log_file_path = os.environ.get("LOG_FILE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log"))

        # Request limits and timeouts
        self.max_tokens_limit = int(
            os.environ.get("MAX_TOKENS_LIMIT", str(ModelDefaults.MAX_TOKENS_LIMIT))
        )
        self.max_retries = int(
            os.environ.get("MAX_RETRIES", str(ModelDefaults.DEFAULT_MAX_RETRIES))
        )

        # Custom models configuration file
        self.custom_models_file = os.environ.get("CUSTOM_MODELS_FILE", "models.yaml")

        # Custom API keys storage
        self.custom_api_keys = {}
        self.model_pricing = {}

    def validate_api_keys(self):
        """Validate that at least one provider API key is configured"""
        providers_configured = []
        if self.custom_api_keys:
            providers_configured.append("custom")

        return providers_configured

    def add_custom_api_key(self, key_name: str, key_value: str):
        """Add a custom API key"""
        self.custom_api_keys[key_name] = key_value

    def get_api_key_for_provider(self, provider: str) -> str | None:
        """Get API key for a specific provider"""
        if provider in self.custom_api_keys:
            return self.custom_api_keys[provider]
        return None


# Initialize configuration
try:
    config = Config()
    if config.log_level.lower() == "debug":
        # DEBUG mode - OpenAI SDK uses standard logging
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.INFO)
    # Ensure log directory exists
    log_dir = os.path.dirname(config.log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    print(f"✅ Configuration loaded: Providers={config.validate_api_keys()}")
    print(
        f"🔀 Router Config: Background={config.router_config['background']}, Think={config.router_config['think']}, LongContext={config.router_config['long_context']}"
    )
except Exception as e:
    print(f"🔴 Configuration Error: {e}")
    sys.exit(1)

# Configure logging
logger = logging.getLogger()
logger.setLevel(getattr(logging, config.log_level.upper()))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# File handler
file_handler = logging.FileHandler(config.log_file_path, mode='w')
file_handler.setFormatter(formatter)

# Stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)

# Initialize tiktoken encoder for token counting
try:
    enc = tiktoken.get_encoding("cl100k_base")
    logger.debug("✅ TikToken encoder initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize TikToken encoder: {e}")
    enc = None

# OpenAI SDK client configurations will be handled per-request

# Configure uvicorn to be quieter
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator",
        ]

        if hasattr(record, "msg") and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True


# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())


# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)


# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(
            ColorizedFormatter("%(asctime)s - %(levelname)s - %(message)s")
        )

app = FastAPI()

# Dictionary to store custom OpenAI-compatible model configurations
CUSTOM_OPENAI_MODELS = {}


# Function to load custom model configurations
def load_custom_models(config_file=None):
    """Load custom OpenAI-compatible model configurations from YAML file."""
    global CUSTOM_OPENAI_MODELS

    if config_file is None:
        config_file = config.custom_models_file

    if not os.path.exists(config_file):
        logger.warning(f"Custom models config file not found: {config_file}")
        return

    try:
        with open(config_file) as file:
            models = yaml.safe_load(file)

        if not models:
            logger.warning(f"No models found in config file: {config_file}")
            return

        # Store model pricing information for cost calculation
        model_pricing = {}

        for model in models:
            if "model_id" not in model or "api_base" not in model:
                logger.warning(
                    f"Invalid model configuration, missing required fields: {model}"
                )
                continue

            model_id = model["model_id"]
            model_name = model.get("model_name", model_id)

            # Set default pricing if not provided
            input_cost = model.get(
                "input_cost_per_token", ModelDefaults.DEFAULT_INPUT_COST_PER_TOKEN
            )
            output_cost = model.get(
                "output_cost_per_token", ModelDefaults.DEFAULT_OUTPUT_COST_PER_TOKEN
            )

            CUSTOM_OPENAI_MODELS[model_id] = {
                "model_id": model_id,
                "model_name": model_name,
                "api_base": model["api_base"],
                "api_key_name": model.get("api_key_name", "OPENAI_API_KEY"),
                "can_stream": model.get("can_stream", True),
                "max_tokens": model.get("max_tokens", ModelDefaults.DEFAULT_MAX_TOKENS),
                "input_cost_per_token": input_cost,
                "output_cost_per_token": output_cost,
                "max_input_tokens": model.get(
                    "max_input_tokens", ModelDefaults.DEFAULT_MAX_INPUT_TOKENS
                ),
                # openai request extra options
                "extra_headers": model.get("extra_headers", None),
                "extra_body": model.get("extra_body", None),
                "reasoning_effort": model.get("reasoning_effort", None),
            }

            # Store pricing info for cost calculation
            model_variations = [
                f"{model_name}",
            ]

            for variation in model_variations:
                model_pricing[variation] = {
                    "input_cost_per_token": input_cost,
                    "output_cost_per_token": output_cost,
                }

            logger.info(
                f"\nLoaded custom OpenAI-compatible model:\n{CUSTOM_OPENAI_MODELS[model_id]}"
            )

        # Store pricing information globally for cost calculation
        if model_pricing:
            # Add to global model pricing dictionary
            if not hasattr(config, "model_pricing"):
                config.model_pricing = {}
            config.model_pricing.update(model_pricing)
            logger.info(f"Loaded pricing for {len(model_pricing)} model variations")

    except Exception as e:
        logger.error(f"Error loading custom models: {str(e)}")


def initialize_custom_models():
    """Initialize custom models and API keys. Called when running as main."""
    load_custom_models()

    # Get custom API keys from environment and store in config
    for model_config in CUSTOM_OPENAI_MODELS.values():
        api_key_name = model_config.get("api_key_name")
        if api_key_name:
            api_key_value = os.environ.get(api_key_name)
            if api_key_value:
                config.add_custom_api_key(api_key_name, api_key_value)
            else:
                logger.warning(f"Missing API key for {api_key_name}")


# Initialize custom models and API keys
initialize_custom_models()


def create_openai_client(model_id: str) -> AsyncOpenAI:
    """Create OpenAI client for the given model and return client and request parameters."""
    api_key = None
    base_url = None
    # Custom OpenAI-compatible models
    if CUSTOM_OPENAI_MODELS[model_id]:
        model_config = CUSTOM_OPENAI_MODELS[model_id]
        api_key_name = model_config.get("api_key_name", "OPENAI_API_KEY")
        api_key = config.custom_api_keys.get(api_key_name)
        base_url = model_config["api_base"]
    else:
        raise ValueError(f"Unknown custom model: {model_id}")

    if not api_key:
        raise ValueError(f"No API key available for model: {model_id}")

    # Create client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = AsyncOpenAI(**client_kwargs)
    logger.debug(f"Create OpenAI Client: model={model_id}, base_url={base_url}")
    return client


def determine_model_by_router(
    original_model: str, token_count: int, has_thinking: bool
) -> str:
    """Determine which model to use based on routing logic"""

    logger.debug(
        f"🔀 Router input: model={original_model}, tokens={token_count}, thinking={has_thinking}"
    )

    # If token count is greater than threshold, use long context model (highest priority)
    if token_count > config.long_context_threshold:
        result = config.router_config["long_context"]
        logger.info(
            f"🔀 Router: Using long context model due to token count: {token_count}, result: {result}"
        )
        return result

    # If has thinking enabled, use think model (second priority)
    if has_thinking:
        result = config.router_config["think"]
        logger.info(
            f"🔀 Router: Using think model for thinking request, result: {result}"
        )
        return result

    # If the model is claude-3-5-haiku, use background model
    if "haiku" in original_model.lower():
        result = config.router_config["background"]
        logger.info(
            f"🔀 Router: Using background model for {original_model}, result: {result}"
        )
        return result

    # If the model is sonnet, use default router
    if "sonnet" in original_model.lower():
        result = config.router_config["default"]
        logger.info(
            f"🔀 Router: Using default model for {original_model}, result: {result}"
        )
        return result

    # Default model
    result = config.router_config["default"]
    logger.debug(f"🔀 Router: Using default model for {original_model}")
    logger.info(f"🔀 Router final result: {result}")

    # should be model id
    return result


# Models for Anthropic API requests
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


def _extract_error_details(e: Exception) -> dict[str, Any]:
    """Extract comprehensive error details from an exception, ensuring all values are JSON serializable."""
    import traceback

    error_details = {
        "error": str(e),
        "type": type(e).__name__,
        "traceback": traceback.format_exc(),
    }

    # Special handling for OpenAI API errors
    if isinstance(
        e,
        (
            APIError,
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            AuthenticationError,
        ),
    ):
        if hasattr(e, "status_code"):
            error_details["status_code"] = e.status_code
        if hasattr(e, "code"):
            error_details["code"] = e.code
        if hasattr(e, "param"):
            error_details["param"] = e.param

    # Combine attributes from the exception's dict and common API error attributes
    attrs_to_check = list(getattr(e, "__dict__", {}).keys())
    attrs_to_check.extend(
        ["message", "status_code", "response", "code", "param", "type"]
    )
    attrs_to_check = sorted(list(set(attrs_to_check)))  # Get unique attributes

    for attr in attrs_to_check:
        if (
            hasattr(e, attr)
            and attr not in error_details
            and attr not in ["args", "__traceback__"]
        ):
            value = getattr(e, attr)

            if attr == "response":
                # The 'response' object from httpx/requests is not JSON serializable.
                # Extract its text content if possible.
                if hasattr(value, "text"):
                    error_details[attr] = value.text
                else:
                    error_details[attr] = str(value)
            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                # This value is already JSON serializable
                error_details[attr] = value
            else:
                # For any other non-serializable type, convert to string as a fallback.
                error_details[attr] = str(value)

    return error_details


def _format_error_message(e: Exception, error_details: dict[str, Any]) -> str:
    """Format error message for response."""
    error_message = f"Error: {str(e)}"
    if "message" in error_details and error_details["message"]:
        error_message += f"\nMessage: {error_details['message']}"
    if "response" in error_details and error_details["response"]:
        error_message += f"\nResponse: {error_details['response']}"
    return error_message


@app.post("/v1/messages")
async def create_message(request: ClaudeMessagesRequest, raw_request: Request):
    try:
        # Parse the raw body as JSON since it's bytes
        body = await raw_request.body()
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", "unknown")

        logger.debug(
            f"RAW REQUEST:\n{json.dumps(body_json, indent=4, ensure_ascii=False, sort_keys=True)}"
        )

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

        # Most of time it would be default model
        model_config = CUSTOM_OPENAI_MODELS[routed_model]
        logger.debug(f"routed model config: {model_config}")

        if model_config is None:
            raise Exception(f"model {routed_model} not defined")

        # Update the request model with routed model

        # Get the display name for logging, just the model name without provider prefix
        display_model = routed_model

        logger.info(
            f"📊 PROCESSING REQUEST: Original={original_model} → Routed={routed_model}, Tokens={token_count}, Stream={request.stream}"
        )

        # Convert Anthropic request to OpenAI format
        openai_request = request.to_openai_request()

        # Create OpenAI client for the model
        client = create_openai_client(routed_model)
        openai_request["model"] = model_config.get("model_name")

        # Add extra headers if defined in model config
        openai_request["extra_headers"] = model_config["extra_headers"] or {}
        openai_request["extra_body"] = model_config["extra_body"] or {}

        # doubao-seed models use "thinking" field as the same as Anthropic's
        #  options:
        #   disabled: not output reasoning content
        #   enabled: output reasoning content
        #   auto: enable thinking automatically
        if has_thinking:
            if model_config["reasoning_effort"] and model_config[
                "reasoning_effort"
            ] in ["low", "medium", "high"]:
                openai_request["reasoning_effort"] = model_config["reasoning_effort"]
            openai_request["extra_body"]["thinking"] = {"type": "enabled"}
        elif model_config["reasoning_effort"] and model_config[
                "reasoning_effort"
            ] in ["low", "medium", "high"]:
            # enable the model to think automatically
            openai_request["extra_body"]["thinking"] = {"type": "auto"}
        else:
            # thinking not supported
            openai_request["extra_body"]["thinking"] = {"type": "disabled"}

        # Intelligent tool_choice adjustment for better model consistency
        # Based on test findings from claude_code_interruption_test:
        # - Claude models naturally tend to use tools in interruption/verification scenarios
        # - Other models (DeepSeek, etc.) may not use tools when tool_choice is None or auto
        # - tool_choice=required ensures consistent behavior across all models
        # - Exception: Thinking models don't support tool_choice=required (API limitation)
        if not has_thinking and openai_request.get("tools") and len(openai_request.get("tools", [])) > 0:
            current_tool_choice = openai_request.get("tool_choice")
            if current_tool_choice is None and openai_request["extra_body"]["thinking"] and openai_request["extra_body"]["thinking"]["type"] == "disabled" :
                logger.debug("🔧 Setting tool_choice to 'auto' for better model consistency (was None)")
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
        max_tokens = max(model_config.get("max_tokens"), request.max_tokens)
        openai_request["max_tokens"] = max_tokens

        # Handle streaming mode
        # Use OpenAI SDK async streaming
        if request.stream:
            response_generator: AsyncStream[
                ChatCompletionChunk
            ] = await client.chat.completions.create(**openai_request)
            return StreamingResponse(
                convert_openai_streaming_response_to_anthropic(
                    response_generator, request, routed_model
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

            # Update global usage statistics and log usage information
            update_global_usage_stats(
                anthropic_response.usage, routed_model, "Non-streaming"
            )

            return anthropic_response

    except Exception as e:
        error_details = _extract_error_details(e)
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")

        error_message = _format_error_message(e, error_details)
        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: ClaudeTokenCountRequest, raw_request: Request):
    try:
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
async def test_message_conversion(request: ClaudeMessagesRequest, raw_request: Request):
    """
    Test endpoint for direct message format conversion without routing.

    This endpoint converts Anthropic format to OpenAI format and sends the request
    directly to the specified model without going through the intelligent routing system.
    Useful for testing specific model integrations and message format conversion.
    """
    try:
        # Parse the raw body to get the original model
        body = await raw_request.body()
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", "unknown")

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
                    response_generator, request
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
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")
