import yaml
from typing import Dict, List, Optional, Any
import os.path

from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel
import os
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI, OpenAI
from openai import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionTool,
    ChatCompletionSystemMessageParam,
)
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
import tiktoken
import hashlib

from models import (
    MessagesRequest,
    MessagesResponse,
    Usage,
    ContentBlockText,
    ContentBlockToolUse,
    ContentBlockThinking,
    TokenCountRequest,
    TokenCountResponse,
)


class SessionStats(BaseModel):
    """Statistics for the current proxy session."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_input_cost: str = "0.0"
    total_output_cost: str = "0.0"


# Global state for session statistics
session_stats = SessionStats()

# Load environment variables from .env file
load_dotenv()


class ModelDefaults:
    """Default values and limits for model configurations"""

    # Token limits
    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_MAX_INPUT_TOKENS = 128000
    MAX_TOKENS_LIMIT = 16384
    LONG_CONTEXT_THRESHOLD = 128000

    # Pricing defaults
    DEFAULT_INPUT_COST_PER_TOKEN = 0.000001
    DEFAULT_OUTPUT_COST_PER_TOKEN = 0.000002

    # Hash signature length
    THINKING_SIGNATURE_LENGTH = 16

    # Default server settings
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = 8082
    DEFAULT_LOG_LEVEL = "WARNING"
    DEFAULT_MAX_RETRIES = 2


class Config:
    """Universal proxy server configuration with intelligent routing"""

    def __init__(self):
        # Router configuration for intelligent model selection
        self.router_config = {
            "background": os.environ.get("ROUTER_BACKGROUND", "deepseek-v3-250324"),
            "think": os.environ.get("ROUTER_THINK", "deepseek-r1-250528"),
            "long_context": os.environ.get("ROUTER_LONG_CONTEXT", "gemini-2.5-pro"),
            "default": os.environ.get("ROUTER_DEFAULT", "deepseek-v3-250324"),
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

        # Request limits and timeouts
        self.max_tokens_limit = int(
            os.environ.get("MAX_TOKENS_LIMIT", str(ModelDefaults.MAX_TOKENS_LIMIT))
        )
        self.max_retries = int(
            os.environ.get("MAX_RETRIES", str(ModelDefaults.DEFAULT_MAX_RETRIES))
        )

        # Custom models configuration file
        self.custom_models_file = os.environ.get(
            "CUSTOM_MODELS_FILE", "custom_models.yaml"
        )

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

    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
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
        logging.getLogger("httpx").setLevel(logging.DEBUG)
    print(f"✅ Configuration loaded: Providers={config.validate_api_keys()}")
    print(
        f"🔀 Router Config: Background={config.router_config['background']}, Think={config.router_config['think']}, LongContext={config.router_config['long_context']}"
    )
except Exception as e:
    print(f"🔴 Configuration Error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
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

# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


# Constants for better maintainability
class Constants:
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    ROLE_TOOL = "tool"

    CONTENT_TEXT = "text"
    CONTENT_IMAGE = "image"
    CONTENT_TOOL_USE = "tool_use"
    CONTENT_TOOL_RESULT = "tool_result"

    TOOL_FUNCTION = "function"

    STOP_END_TURN = "end_turn"
    STOP_MAX_TOKENS = "max_tokens"
    STOP_TOOL_USE = "tool_use"
    STOP_ERROR = "error"

    EVENT_MESSAGE_START = "message_start"
    EVENT_MESSAGE_STOP = "message_stop"
    EVENT_MESSAGE_DELTA = "message_delta"
    EVENT_CONTENT_BLOCK_START = "content_block_start"
    EVENT_CONTENT_BLOCK_STOP = "content_block_stop"
    EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
    EVENT_PING = "ping"

    DELTA_TEXT = "text_delta"
    DELTA_INPUT_JSON = "input_json_delta"


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
        with open(config_file, "r") as file:
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
            }

            # Store pricing info for cost calculation
            model_variations = [
                f"custom/{model_id}",  # With custom/ prefix
                f"openai/{model_id}",  # With openai/ prefix
                f"openai/{model_name}",  # With openai/ prefix
            ]

            for variation in model_variations:
                model_pricing[variation] = {
                    "input_cost_per_token": input_cost,
                    "output_cost_per_token": output_cost,
                }

            logger.info(
                f"Loaded custom OpenAI-compatible model: {model_id} → {model_name}"
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


def generate_thinking_signature(thinking_content: str) -> str:
    """Generate a signature for thinking content using SHA-256 hash."""
    if not thinking_content:
        return ""

    # Create SHA-256 hash of the thinking content
    hash_object = hashlib.sha256(thinking_content.encode("utf-8"))
    signature = hash_object.hexdigest()[: ModelDefaults.THINKING_SIGNATURE_LENGTH]
    return f"thinking_{signature}"


def create_openai_client(model: str, is_async: bool = True) -> tuple:
    """Create OpenAI client for the given model and return client and request parameters."""
    api_key = None
    base_url = None
    # Custom OpenAI-compatible models
    if model in CUSTOM_OPENAI_MODELS:
        model_config = CUSTOM_OPENAI_MODELS[model]
        api_key_name = model_config.get("api_key_name", "OPENAI_API_KEY")
        api_key = config.custom_api_keys.get(api_key_name)
        base_url = model_config["api_base"]
        model_name = model_config.get("model_name", model)
    else:
        raise ValueError(f"Unknown custom model: {model}")

    if not api_key:
        raise ValueError(f"No API key available for model: {model}")

    # Create client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    if is_async:
        client = AsyncOpenAI(**client_kwargs)
    else:
        client = OpenAI(**client_kwargs)

    return client, model_name


# Helper function to clean schema for Gemini
def _remove_gemini_incompatible_uri_format(schema: dict) -> dict:
    """Fix only the specific URI format issue that causes MALFORMED_FUNCTION_CALL in Gemini."""
    # Create a copy to avoid modifying the original
    fixed_schema = {}

    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            # Fix properties recursively
            fixed_properties = {}
            for prop_name, prop_schema in value.items():
                if isinstance(prop_schema, dict):
                    fixed_prop = _remove_gemini_incompatible_uri_format(prop_schema)
                    fixed_properties[prop_name] = fixed_prop
                else:
                    fixed_properties[prop_name] = prop_schema
            fixed_schema[key] = fixed_properties
        elif key == "format" and value == "uri":
            # Skip the format field if it's 'uri' - this is the main issue
            logger.debug("Removing 'uri' format for Gemini compatibility")
            continue
        elif isinstance(value, dict):
            # Recursively fix nested objects
            fixed_schema[key] = _remove_gemini_incompatible_uri_format(value)
        elif isinstance(value, list):
            # Handle arrays
            fixed_list = []
            for item in value:
                if isinstance(item, dict):
                    fixed_list.append(_remove_gemini_incompatible_uri_format(item))
                else:
                    fixed_list.append(item)
            fixed_schema[key] = fixed_list
        else:
            # Keep everything else as-is
            fixed_schema[key] = value

    return fixed_schema


def clean_gemini_schema(schema: Any) -> Any:
    """
    Recursively clean and validate JSON schema for Gemini OpenAI API compatibility.

    Gemini requires a strict subset of OpenAPI schema format:
    - Function names: descriptive, no spaces/special chars (underscores/camelCase OK)
    - Parameters: must be object type with properties structure
    - Required: array of mandatory parameter names
    - Enum: use arrays for fixed value sets
    """
    if isinstance(schema, dict):
        # Create a copy to avoid modifying the original
        cleaned_schema = {}

        for key, value in schema.items():
            # Skip problematic fields that cause MALFORMED_FUNCTION_CALL
            if key in [
                "additionalProperties",  # Gemini doesn't support this
                "default",  # Can cause validation issues
                "examples",  # Not part of core OpenAPI subset
                "title",  # Not needed for function parameters
                "$schema",  # Schema metadata not needed
                "$id",  # Schema metadata not needed
                "definitions",  # Complex references not supported
                "$ref",  # References not supported
                "allOf",
                "anyOf",
                "oneOf",  # Complex schema combinations
                "not",  # Negation schemas not supported
                "patternProperties",  # Advanced property patterns
                "dependencies",  # Complex dependency validation
                "const",  # Use enum instead
                "contains",  # Array contains validation
                "propertyNames",  # Property name validation
                "if",
                "then",
                "else",  # Conditional schemas
                "readOnly",
                "writeOnly",  # OpenAPI 3.0 specific fields
                "deprecated",  # OpenAPI metadata
                "external_docs",  # OpenAPI documentation
                "xml",  # XML-specific metadata
                "example",  # Use examples array instead
            ]:
                logger.debug(
                    f"Removing unsupported field '{key}' for Gemini compatibility"
                )
                continue

            # Handle format field - only allow very basic formats
            if key == "format":
                # Gemini only supports very limited formats
                allowed_formats = {"date-time", "email", "uri"}  # Minimal set
                if value not in allowed_formats:
                    logger.debug(
                        f"Removing unsupported format '{value}' for Gemini compatibility"
                    )
                    continue
                # Even for allowed formats, be conservative and remove format
                # as it can cause validation issues
                logger.debug(
                    f"Removing format '{value}' for maximum Gemini compatibility"
                )
                continue

            # Handle string length constraints - be more lenient
            if key in ["minLength", "maxLength"] and isinstance(value, int):
                # Keep reasonable constraints but avoid extremes
                if key == "minLength" and value < 0:
                    continue
                if key == "maxLength" and value > 1000000:  # 1M chars max
                    cleaned_schema[key] = 1000000
                    continue

            # Handle numeric constraints
            if key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]:
                # Keep basic numeric constraints
                if isinstance(value, (int, float)):
                    cleaned_schema[key] = value
                continue

            # Handle pattern - remove complex regex patterns
            if key == "pattern":
                # Gemini may not support complex regex patterns
                logger.debug(f"Removing pattern '{value}' for Gemini compatibility")
                continue

            # Handle multipleOf - keep simple ones
            if key == "multipleOf" and isinstance(value, (int, float)) and value > 0:
                cleaned_schema[key] = value
                continue

            # Recursively clean nested schemas
            if isinstance(value, dict):
                cleaned_value = clean_gemini_schema(value)
                if cleaned_value:  # Only add if not empty
                    cleaned_schema[key] = cleaned_value
            elif isinstance(value, list):
                cleaned_value = [clean_gemini_schema(item) for item in value]
                # Filter out empty items
                cleaned_value = [
                    item for item in cleaned_value if item is not None and item != {}
                ]
                if cleaned_value:  # Only add if not empty
                    cleaned_schema[key] = cleaned_value
            else:
                # Keep primitive values
                cleaned_schema[key] = value

        return cleaned_schema
    elif isinstance(schema, list):
        # Recursively clean items in a list
        cleaned_list = [clean_gemini_schema(item) for item in schema]
        # Filter out empty items
        return [item for item in cleaned_list if item is not None and item != {}]

    return schema


def validate_gemini_function_schema(tool_def: dict) -> tuple[bool, str]:
    """
    Validate a function definition for Gemini OpenAI API compatibility.

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        # Check basic structure
        if not isinstance(tool_def, dict):
            return False, "Tool definition must be a dictionary"

        if tool_def.get("type") != "function":
            return False, "Tool type must be 'function'"

        function_def = tool_def.get("function")
        if not isinstance(function_def, dict):
            return False, "Tool must have a 'function' object"

        # Validate function name
        name = function_def.get("name")
        if not name or not isinstance(name, str):
            return False, "Function must have a non-empty string name"

        # Check name format - no spaces, only alphanumeric, underscores, hyphens
        import re

        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", name):
            return (
                False,
                f"Function name '{name}' contains invalid characters. Use alphanumeric, underscores, or hyphens only",
            )

        # Validate description
        description = function_def.get("description", "")
        if not isinstance(description, str):
            return False, "Function description must be a string"

        if len(description.strip()) == 0:
            return False, "Function description cannot be empty (required for Gemini)"

        # Validate parameters schema
        parameters = function_def.get("parameters")
        if parameters is None:
            # Empty parameters are allowed
            return True, ""

        if not isinstance(parameters, dict):
            return False, "Parameters must be a dictionary"

        # For Gemini, parameters must be object type
        param_type = parameters.get("type")
        if param_type is None:
            return False, "Parameters must have a 'type' field"

        if param_type != "object":
            return False, f"Parameters type must be 'object', got '{param_type}'"

        # Validate properties if present
        properties = parameters.get("properties")
        if properties is not None:
            if not isinstance(properties, dict):
                return False, "Properties must be a dictionary"

            for prop_name, prop_schema in properties.items():
                if not isinstance(prop_name, str) or not prop_name.strip():
                    return (
                        False,
                        f"Property name must be a non-empty string, got '{prop_name}'",
                    )

                if not isinstance(prop_schema, dict):
                    return False, f"Property '{prop_name}' schema must be a dictionary"

                # Each property must have a type
                prop_type = prop_schema.get("type")
                if prop_type is None:
                    return False, f"Property '{prop_name}' must have a 'type' field"

                # Validate allowed types for Gemini
                allowed_types = {
                    "string",
                    "number",
                    "integer",
                    "boolean",
                    "array",
                    "object",
                }
                if prop_type not in allowed_types:
                    return (
                        False,
                        f"Property '{prop_name}' has unsupported type '{prop_type}'. Allowed: {allowed_types}",
                    )

                # Validate array items
                if prop_type == "array":
                    items = prop_schema.get("items")
                    if items is None:
                        return (
                            False,
                            f"Array property '{prop_name}' must have 'items' definition",
                        )

                    if isinstance(items, dict):
                        items_type = items.get("type")
                        if items_type is None:
                            return (
                                False,
                                f"Array property '{prop_name}' items must have a 'type' field",
                            )

                        if items_type not in allowed_types:
                            return (
                                False,
                                f"Array property '{prop_name}' items has unsupported type '{items_type}'",
                            )

        # Validate required array if present
        required = parameters.get("required")
        if required is not None:
            if not isinstance(required, list):
                return False, "Required must be an array"

            for req_field in required:
                if not isinstance(req_field, str) or not req_field.strip():
                    return (
                        False,
                        f"Required field must be a non-empty string, got '{req_field}'",
                    )

                # Check that required fields exist in properties
                if properties and req_field not in properties:
                    return (
                        False,
                        f"Required field '{req_field}' not found in properties",
                    )

        return True, ""

    except Exception as e:
        return False, f"Validation error: {str(e)}"


# Token counting and routing functions
def count_tokens_in_messages(messages: List[Dict], system=None, tools=None) -> int:
    """Count tokens in messages using tiktoken"""
    if enc is None:
        logger.warning("TikToken encoder not available, using approximate count")
        return 1000  # Fallback estimate

    total_tokens = 0

    try:
        # Count message tokens
        if messages:
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    total_tokens += len(enc.encode(content))
                elif isinstance(content, list):
                    for content_part in content:
                        if isinstance(content_part, dict):
                            if content_part.get("type") == "text":
                                text = content_part.get("text", "")
                                total_tokens += len(enc.encode(text))
                            elif content_part.get("type") == "tool_use":
                                tool_input = content_part.get("input", {})
                                total_tokens += len(enc.encode(json.dumps(tool_input)))
                            elif content_part.get("type") == "tool_result":
                                tool_content = content_part.get("content", "")
                                if isinstance(tool_content, str):
                                    total_tokens += len(enc.encode(tool_content))
                                else:
                                    total_tokens += len(
                                        enc.encode(json.dumps(tool_content))
                                    )

        # Count system message tokens
        if system:
            if isinstance(system, str):
                total_tokens += len(enc.encode(system))
            elif isinstance(system, list):
                for item in system:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_tokens += len(enc.encode(item.get("text", "")))

        # Count tool tokens
        if tools:
            for tool in tools:
                tool_name = tool.get("name", "")
                tool_desc = tool.get("description", "")
                tool_schema = tool.get("input_schema", {})
                total_tokens += len(enc.encode(tool_name + tool_desc))
                total_tokens += len(enc.encode(json.dumps(tool_schema)))

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 1000  # Fallback estimate

    return total_tokens


def count_tokens_in_response(
    response_content: str = "", thinking_content: str = "", tool_calls: list = None
) -> int:
    """Count tokens in response content using tiktoken"""
    if enc is None:
        logger.warning(
            "TikToken encoder not available for response counting, using approximate count"
        )
        return (
            len(response_content.split()) * 1.3
        )  # Rough estimate: 1.3 tokens per word

    total_tokens = 0

    try:
        # Count main response content tokens
        if response_content:
            total_tokens += len(enc.encode(response_content))

        # Count thinking content tokens
        if thinking_content:
            total_tokens += len(enc.encode(thinking_content))

        # Count tool calls tokens
        if tool_calls:
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    # Dict format
                    if "function" in tool_call:
                        func_data = tool_call["function"]
                        name = func_data.get("name", "")
                        arguments = func_data.get("arguments", "")
                        total_tokens += len(enc.encode(name + arguments))
                    elif "name" in tool_call and "input" in tool_call:
                        # Anthropic format
                        name = tool_call.get("name", "")
                        input_data = tool_call.get("input", {})
                        total_tokens += len(enc.encode(name + json.dumps(input_data)))
                elif hasattr(tool_call, "function"):
                    # OpenAI format
                    name = getattr(tool_call.function, "name", "")
                    arguments = getattr(tool_call.function, "arguments", "")
                    total_tokens += len(enc.encode(name + arguments))

    except Exception as e:
        logger.error(f"Error counting response tokens: {e}")
        # Fallback estimation
        return len(response_content.split()) * 1.3 if response_content else 0

    return int(total_tokens)


def calculate_request_tokens(request: MessagesRequest) -> int:
    """Helper function to calculate tokens for a request, handling data conversion."""
    messages_dict = [msg.model_dump() for msg in request.messages]
    system_dict = request.system
    if isinstance(system_dict, list):
        system_dict = [s.model_dump() for s in system_dict]
    tools_dict = (
        [tool.model_dump() for tool in request.tools] if request.tools else None
    )
    return count_tokens_in_messages(messages_dict, system_dict, tools_dict)


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
    return result


# Shared model validation logic
def validate_and_map_model(original_model: str, info=None) -> str:
    """Validate and map model names by adding provider prefixes if missing."""
    new_model = original_model
    logger.debug(f"📋 Auto-prefixing model: Original='{original_model}'")

    # Remove existing provider prefixes for clean matching
    clean_v = original_model
    if clean_v.startswith("custom/"):
        parts = original_model.split("/", 1)
        clean_v = parts[1]

    mapped = False
    # Add provider prefixes if they are missing
    if clean_v in CUSTOM_OPENAI_MODELS and not original_model.startswith("custom/"):
        new_model = f"custom/{clean_v}"
        mapped = True

    if mapped:
        logger.info(f"📌 Auto-prefixing: '{original_model}' ➡️ '{new_model}'")

    # Store the original model in the values dictionary if info is provided
    if info and hasattr(info, "data") and isinstance(info.data, dict):
        info.data["original_model"] = original_model

    return new_model


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


def _extract_system_content(system) -> str:
    """Extract system content from various formats."""
    if isinstance(system, str):
        return system
    elif isinstance(system, list):
        system_content = ""
        for block in system:
            if hasattr(block, "text"):
                system_content += block.text
            elif isinstance(block, dict) and "text" in block:
                system_content += block["text"]
        return system_content
    return ""


def _extract_message_content(content) -> str:
    """Extract message content and convert to text format."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        content_text = ""
        for block in content:
            if hasattr(block, "text"):
                content_text += block.text
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    content_text += block["text"]
                elif block.get("type") == "tool_result":
                    # Convert tool result to text representation
                    if "content" in block:
                        if isinstance(block["content"], str):
                            content_text += f"Tool result: {block['content']}"
                        elif isinstance(block["content"], list):
                            for result_block in block["content"]:
                                if (
                                    isinstance(result_block, dict)
                                    and "text" in result_block
                                ):
                                    content_text += (
                                        f"Tool result: {result_block['text']}"
                                    )
        return content_text or "..."
    return "..."


def convert_anthropic_to_openai_request(
    request: MessagesRequest, model: str
) -> Dict[str, Any]:
    """Convert Anthropic API request to OpenAI API format using OpenAI SDK types for validation."""

    # Build OpenAI messages with type validation
    openai_messages = []

    # Convert system message if present
    if request.system:
        system_content = _extract_system_content(request.system)
        if system_content:
            # Validate using TypedDict type (returns dict)
            system_msg: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": system_content,
            }
            openai_messages.append(system_msg)

    # Convert messages
    for msg in request.messages:
        content_text = _extract_message_content(msg.content)

        if msg.role == "user":
            openai_msg = {
                "role": "user",
                "content": content_text,
            }
        elif msg.role == "assistant":
            openai_msg = {
                "role": "assistant",
                "content": content_text,
            }
        else:
            # Fallback for other roles
            openai_msg = {"role": msg.role, "content": content_text}

        openai_messages.append(openai_msg)

    # Build tools with type validation
    openai_tools = None
    if request.tools:
        openai_tools = []
        for tool in request.tools:
            # Handle both Anthropic tool format and function-based format
            if hasattr(tool, "name"):
                # Anthropic tool format (our models.py Tool class)
                tool_params = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                    },
                }
                if hasattr(tool, "input_schema"):
                    # Apply Gemini schema cleaning if model contains 'gemini'
                    if "gemini" in model.lower():
                        tool_params["function"]["parameters"] = clean_gemini_schema(
                            tool.input_schema
                        )
                    else:
                        tool_params["function"]["parameters"] = tool.input_schema

                # Validate using OpenAI SDK type
                logger.debug(
                    f"original Claude Tool format: {tool}"
                )
                logger.debug(
                    f"before validate tool params: {tool_params}"
                )
                validated_tool = ChatCompletionTool.model_validate(tool_params)
                logger.debug(
                    f"after validate tool params: {validated_tool}"
                )
                openai_tools.append(validated_tool.model_dump(exclude_none=True))
                logger.debug(
                    f"openai_tools append: {validated_tool}"
                )

    # Handle tool_choice with type validation
    tool_choice = None
    if request.tool_choice:
        tool_choice = request.tool_choice.to_openai()
        logger.debug(
            f"openai tool choice param: {tool_choice} <- {request.tool_choice}")

    # Build complete request with OpenAI SDK type validation
    request_params = {
        "model": model,
        "messages": openai_messages,
        "stream": request.stream,
        "max_tokens": min(
            CUSTOM_OPENAI_MODELS[model].get("max_tokens"), request.max_tokens
        ),
        "temperature": request.temperature,
        "top_p": request.top_p,
    }

    if openai_tools:
        request_params["tools"] = openai_tools
    if tool_choice:
        request_params["tool_choice"] = tool_choice

    logger.debug(
        f"original request: {request}"
    )
    logger.debug(
        f"openai request: {request_params}"
    )
    # Return the request with validated components
    # Individual components (messages, tools) are already validated using OpenAI SDK types
    return request_params


def _extract_error_details(e: Exception) -> Dict[str, Any]:
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


def _format_error_message(e: Exception, error_details: Dict[str, Any]) -> str:
    """Format error message for response."""
    error_message = f"Error: {str(e)}"
    if "message" in error_details and error_details["message"]:
        error_message += f"\nMessage: {error_details['message']}"
    if "response" in error_details and error_details["response"]:
        error_message += f"\nResponse: {error_details['response']}"
    return error_message


def update_session_stats(model: str, input_tokens: int, output_tokens: int):
    """Update the global session statistics with new token counts and costs."""
    global session_stats
    input_cost_per_token = 0.0
    output_cost_per_token = 0.0
    # Clean model name to look up in custom models
    clean_model_name = model
    if model.startswith("custom/"):
        clean_model_name = model[7:]
    if clean_model_name in CUSTOM_OPENAI_MODELS:
        model_config = CUSTOM_OPENAI_MODELS[clean_model_name]
        input_cost_per_token = model_config.get("input_cost_per_token", 0.0)
        output_cost_per_token = model_config.get("output_cost_per_token", 0.0)
    else:
        # Use stored pricing from config if available, otherwise use defaults
        if hasattr(config, "model_pricing") and model in config.model_pricing:
            pricing = config.model_pricing[model]
            input_cost_per_token = pricing.get(
                "input_cost_per_token", ModelDefaults.DEFAULT_INPUT_COST_PER_TOKEN
            )
            output_cost_per_token = pricing.get(
                "output_cost_per_token", ModelDefaults.DEFAULT_OUTPUT_COST_PER_TOKEN
            )
        else:
            # Use default pricing for unknown models
            input_cost_per_token = ModelDefaults.DEFAULT_INPUT_COST_PER_TOKEN
            output_cost_per_token = ModelDefaults.DEFAULT_OUTPUT_COST_PER_TOKEN
            logger.warning(f"Using default pricing for model {model}")
    # Calculate costs for this request
    input_cost = input_tokens * input_cost_per_token
    output_cost = output_tokens * output_cost_per_token
    # Update global stats
    session_stats.total_input_tokens += input_tokens
    session_stats.total_output_tokens += output_tokens
    session_stats.total_input_cost = "{:.12f}".format(
        float(session_stats.total_input_cost) + input_cost
    )
    session_stats.total_output_cost = "{:.12f}".format(
        float(session_stats.total_output_cost) + output_cost
    )
    logger.info(
        f"📊 STATS UPDATE: Model={model}, Input={input_tokens}t/${input_cost:.6f}, Output={output_tokens}t/${output_cost:.6f}"
    )
    logger.info(
        f"SESSION TOTALS: Input={session_stats.total_input_tokens}t/${float(session_stats.total_input_cost):.6f}, Output={session_stats.total_output_tokens}t/${float(session_stats.total_output_cost):.6f}"
    )


def _process_image_content_block(block, image_parts: List[Dict]) -> None:
    """Process an image content block by adding it to image_parts."""
    if (
        isinstance(block.source, dict)
        and block.source.get("type") == "base64"
        and "media_type" in block.source
        and "data" in block.source
    ):
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                },
            }
        )


def _finalize_user_message(
    text_parts: List[str],
    image_parts: List[Dict],
    is_gemini_model: bool,
    is_openai_model: bool,
    is_custom_model: bool,
    litellm_messages: List[Dict],
    pending_tool_messages: List[Dict],
) -> None:
    """Finalize user message based on model type."""
    if text_parts or image_parts:
        if is_openai_model or is_custom_model:
            # OpenAI/Custom: Convert to simple string format
            text_content = "".join(text_parts).strip()
            if image_parts:
                text_content += (
                    "\n[Note: Image content present but not displayed in text format]"
                )
            if text_content:
                litellm_messages.append(
                    {"role": Constants.ROLE_USER, "content": text_content}
                )
        else:
            # Gemini: Support structured content
            content_parts = []
            text_content = "".join(text_parts).strip()
            if text_content:
                content_parts.append(
                    {"type": Constants.CONTENT_TEXT, "text": text_content}
                )
            content_parts.extend(image_parts)

            litellm_messages.append(
                {
                    "role": Constants.ROLE_USER,
                    "content": content_parts[0]["text"]
                    if len(content_parts) == 1
                    and content_parts[0]["type"] == Constants.CONTENT_TEXT
                    else content_parts,
                }
            )
    # Add any pending tool messages (only for Gemini)
    if is_gemini_model:
        litellm_messages.extend(pending_tool_messages)


def clean_tool_markers(content: str) -> str:
    """Clean up tool call markers and malformed content (DeepSeek-R1 style)."""
    if not content:
        return content

    # Remove specific tool call markers
    content = re.sub(r"<｜tool▁call▁end｜>", "", content)
    content = re.sub(r"<\|tool_call_end\|>", "", content)

    # Remove DeepSeek-R1 function call blocks completely
    patterns_to_remove = [
        # Pattern 1: function FunctionName\n```json\n{...}\n```
        r"function\s+\w+\s*\n\s*```json\s*\n\s*{[\s\S]*?}\s*\n\s*```",
        # Pattern 2: function FunctionName ```json\n{...}\n```
        r"function\s+\w+\s*```json\s*\n\s*{[\s\S]*?}\s*\n\s*```",
        # Pattern 3: function FunctionName\n```json{...}```
        r"function\s+\w+\s*\n?\s*```json\s*{.*?}\s*```",
        # Pattern 4: More flexible - function name followed by any JSON in code blocks
        r"function\s+\w+[\s\S]*?```(?:json)?\s*{[\s\S]*?}\s*```",
    ]

    for pattern in patterns_to_remove:
        content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)

    # Remove standalone function names that appear by themselves
    content = re.sub(r"^\s*function\s+\w+\s*$", "", content, flags=re.MULTILINE)

    # Clean up excessive whitespace
    content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
    content = re.sub(r"^\s*\n+", "", content)  # Remove leading newlines
    content = content.strip()

    return content


def parse_tool_calls_from_content(content: str) -> list:
    """Extract tool calls from malformed content (DeepSeek-R1 style)."""
    tool_calls = []

    # DeepSeek-R1 specific patterns
    patterns = [
        # Pattern 1: function FunctionName\n```json\n{...}\n```
        r"function\s+(\w+)\s*\n\s*```json\s*\n\s*({[\s\S]*?})\s*\n\s*```",
        # Pattern 2: function FunctionName ```json\n{...}\n```
        r"function\s+(\w+)\s*```json\s*\n\s*({[\s\S]*?})\s*\n\s*```",
        # Pattern 3: function FunctionName\n```json{...}```
        r"function\s+(\w+)\s*\n?\s*```json\s*({.*?})\s*```",
        # Pattern 4: More flexible - function name followed by any JSON in code blocks
        r"function\s+(\w+)[\s\S]*?```(?:json)?\s*({[\s\S]*?})\s*```",
    ]

    logger.debug(f"Attempting to parse tool calls from content: {content[:200]}...")

    for i, pattern in enumerate(patterns):
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        if matches:
            logger.debug(f"Pattern {i + 1} found {len(matches)} matches")
            for match in matches:
                function_name = match[0]
                json_str = match[1].strip()

                # Clean up the JSON string
                json_str = re.sub(r"\n\s*", " ", json_str)  # Remove extra whitespace
                json_str = re.sub(r",\s*}", "}", json_str)  # Remove trailing commas
                json_str = re.sub(
                    r",\s*]", "]", json_str
                )  # Remove trailing commas in arrays

                try:
                    arguments = json.loads(json_str)
                    tool_call_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": json.dumps(arguments)
                                if isinstance(arguments, dict)
                                else json_str,
                            },
                        }
                    )
                    logger.info(
                        f"Successfully extracted tool call: {function_name} with args: {json.dumps(arguments)[:100]}"
                    )
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse JSON for function {function_name}: {json_str[:100]}... Error: {e}"
                    )
                    # Try to create a tool call with raw arguments
                    tool_call_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": json.dumps({"raw_input": json_str}),
                            },
                        }
                    )

            # If we found matches with this pattern, stop trying others
            if tool_calls:
                break

    if not tool_calls:
        logger.debug("No tool calls found using standard patterns")

    return tool_calls


def _parse_litellm_response_content(litellm_response, default_response_id: str) -> dict:
    """Extract response data from LiteLLM response in various formats."""
    response_data = {
        "response_id": default_response_id,
        "raw_content": "",
        "tool_calls": None,
        "thinking_content": None,
        "finish_reason": "stop",
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    # Handle LiteLLM ModelResponse object format
    if hasattr(litellm_response, "choices") and hasattr(litellm_response, "usage"):
        choices = litellm_response.choices
        message = choices[0].message if choices else None
        response_data["raw_content"] = getattr(message, "content", "") or ""
        response_data["tool_calls"] = getattr(message, "tool_calls", None)

        # Extract reasoning_content from deepseek reasoning models
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            response_data["thinking_content"] = message.reasoning_content
            logger.debug(
                f"Extracted reasoning_content: {len(message.reasoning_content)} characters"
            )

        response_data["finish_reason"] = choices[0].finish_reason if choices else "stop"
        response_data["response_id"] = getattr(
            litellm_response, "id", default_response_id
        )

        if hasattr(litellm_response, "usage"):
            usage = litellm_response.usage
            response_data["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
            response_data["completion_tokens"] = getattr(usage, "completion_tokens", 0)

    # Handle dictionary response format
    elif isinstance(litellm_response, dict):
        choices = litellm_response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        response_data["raw_content"] = message.get("content", "") or ""
        response_data["tool_calls"] = message.get("tool_calls")

        # Extract reasoning_content from deepseek reasoning models (dict format)
        if message.get("reasoning_content"):
            response_data["thinking_content"] = message.get("reasoning_content")
            logger.debug(
                f"Extracted reasoning_content (dict): {len(message.get('reasoning_content'))} characters"
            )

        response_data["finish_reason"] = (
            choices[0].get("finish_reason", "stop") if choices else "stop"
        )
        usage = litellm_response.get("usage", {})
        response_data["prompt_tokens"] = usage.get("prompt_tokens", 0)
        response_data["completion_tokens"] = usage.get("completion_tokens", 0)
        response_data["response_id"] = litellm_response.get("id", default_response_id)

    return response_data


def _extract_tool_call_data(tool_call) -> tuple:
    """Extract tool call data from different formats."""
    if isinstance(tool_call, dict):
        tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
        function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
        name = function_data.get("name", "")
        arguments_str = function_data.get("arguments", "{}")
    elif hasattr(tool_call, "id") and hasattr(tool_call, Constants.TOOL_FUNCTION):
        tool_id = tool_call.id
        name = tool_call.function.name
        arguments_str = tool_call.function.arguments
    else:
        return None, None, None

    return tool_id, name, arguments_str


def _parse_tool_arguments(arguments_str: str) -> dict:
    """Parse tool arguments safely."""
    try:
        arguments_dict = json.loads(arguments_str)
        if not isinstance(arguments_dict, dict):
            arguments_dict = {"input": arguments_dict}
        return arguments_dict
    except json.JSONDecodeError:
        return {"raw_arguments": arguments_str}


def convert_openai_to_anthropic(
    openai_response, original_request: MessagesRequest
) -> MessagesResponse:
    """Convert OpenAI response back to Anthropic API format using OpenAI SDK type validation."""
    try:
        # Validate and extract response data using OpenAI SDK types
        response_id = f"msg_{uuid.uuid4()}"
        content_text = ""
        tool_calls = None
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        thinking_content = ""

        logger.debug(f"Converting OpenAI response: {type(openai_response)}")

        # Validate response using OpenAI SDK types
        validated_response = None
        try:
            if hasattr(openai_response, "model_dump"):
                # Already a Pydantic model
                validated_response = openai_response
            else:
                # Validate raw response
                validated_response = ChatCompletion.model_validate(openai_response)

            # Extract data from validated response using model_dump
            response_dict = validated_response.model_dump()

            # Extract response data safely from validated structure
            if response_dict.get("choices"):
                choice = response_dict["choices"][0]

                # Extract message content
                message = choice.get("message", {})
                content_text = message.get("content", "")

                # Extract reasoning_content for thinking models (OpenAI format)
                if message.get("reasoning_content"):
                    thinking_content = message.get("reasoning_content")
                    logger.debug(
                        f"Extracted reasoning_content: {len(thinking_content)} characters"
                    )

                # Extract tool calls if present
                if message.get("tool_calls"):
                    tool_calls = message["tool_calls"]

                # Extract finish reason
                finish_reason = choice.get("finish_reason", "stop")

            # Extract usage information
            usage = response_dict.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

        except Exception as validation_error:
            logger.warning(
                f"OpenAI response validation failed: {validation_error}, using fallback extraction"
            )
            # Fallback to original extraction method
            if hasattr(openai_response, "choices") and openai_response.choices:
                choice = openai_response.choices[0]

                # Extract message content
                if hasattr(choice, "message") and choice.message:
                    if hasattr(choice.message, "content") and choice.message.content:
                        content_text = choice.message.content

                    # Extract reasoning_content for thinking models (fallback)
                    if (
                        hasattr(choice.message, "reasoning_content")
                        and choice.message.reasoning_content
                    ):
                        thinking_content = choice.message.reasoning_content
                        logger.debug(
                            f"Extracted reasoning_content (fallback): {len(thinking_content)} characters"
                        )

                    # Extract tool calls if present
                    if (
                        hasattr(choice.message, "tool_calls")
                        and choice.message.tool_calls
                    ):
                        tool_calls = choice.message.tool_calls

                # Extract finish reason
                if hasattr(choice, "finish_reason"):
                    finish_reason = choice.finish_reason

            # Extract usage information
            if hasattr(openai_response, "usage") and openai_response.usage:
                if hasattr(openai_response.usage, "prompt_tokens"):
                    prompt_tokens = openai_response.usage.prompt_tokens
                if hasattr(openai_response.usage, "completion_tokens"):
                    completion_tokens = openai_response.usage.completion_tokens

        logger.debug(f"Raw content extracted: {len(content_text)} characters")
        logger.debug(f"Tool calls from response: {tool_calls}")

        # Enhanced debugging for Claude Code tool testing
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug("=== ENHANCED DEBUG INFO ===")
            if thinking_content:
                logger.debug(f"Thinking content preview: {thinking_content[:500]}...")
            if content_text:
                logger.debug(f"Raw content text: {repr(content_text)}")
            logger.debug("=== END DEBUG INFO ===")

        # Clean up content text - remove tool call markers if present
        content_text = clean_tool_markers(content_text) if content_text else ""

        # Try to extract tool calls from malformed content if no proper tool_calls found
        if not tool_calls and content_text:
            # Check for DeepSeek-R1 style function calls first
            if "function" in content_text and (
                "```json" in content_text or "{" in content_text
            ):
                extracted_tool_calls = parse_tool_calls_from_content(content_text)
                if extracted_tool_calls:
                    tool_calls = extracted_tool_calls
                    logger.debug(
                        f"Extracted {len(tool_calls)} tool calls from DeepSeek-R1 style content"
                    )
                    # Clean the content to remove the tool call markup
                    content_text = clean_tool_markers(content_text)

        # Build content blocks
        content_blocks = []

        # Add thinking content first if present (for Claude Code display)
        if thinking_content:
            thinking_signature = generate_thinking_signature(thinking_content)
            content_blocks.append(
                ContentBlockThinking(
                    type="thinking",
                    thinking=thinking_content,
                    signature=thinking_signature,
                )
            )
            logger.debug(
                f"Added thinking content block: {len(thinking_content)} characters, signature: {thinking_signature}"
            )

        # Add text content if present
        if content_text:
            content_blocks.append(
                ContentBlockText(type=Constants.CONTENT_TEXT, text=content_text)
            )

        # Process tool calls
        if tool_calls:
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for tool_call in tool_calls:
                try:
                    # Extract tool call data from OpenAI format
                    if hasattr(tool_call, "id"):
                        tool_id = tool_call.id
                    else:
                        tool_id = f"call_{uuid.uuid4().hex[:8]}"

                    if hasattr(tool_call, "function"):
                        name = tool_call.function.name
                        arguments_str = tool_call.function.arguments
                    else:
                        # Fallback for other formats
                        tool_id, name, arguments_str = _extract_tool_call_data(
                            tool_call
                        )

                    if not tool_id or not name:
                        continue

                    arguments_dict = _parse_tool_arguments(arguments_str)
                    content_blocks.append(
                        ContentBlockToolUse(
                            type=Constants.CONTENT_TOOL_USE,
                            id=tool_id,
                            name=name,
                            input=arguments_dict,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error processing tool call: {e}")
                    continue

        # Ensure at least one content block
        if not content_blocks:
            content_blocks.append(
                ContentBlockText(type=Constants.CONTENT_TEXT, text="")
            )

        # Map finish reason to Anthropic format
        if finish_reason == "length":
            stop_reason = Constants.STOP_MAX_TOKENS
        elif finish_reason == "tool_calls":
            stop_reason = Constants.STOP_TOOL_USE
        elif finish_reason is None and tool_calls:
            stop_reason = Constants.STOP_TOOL_USE
        else:
            stop_reason = Constants.STOP_END_TURN

        # Final debug info for Claude message creation
        if logger.isEnabledFor(10):  # DEBUG level
            logger.debug(f"Creating Claude response with {len(content_blocks)} content blocks:")
            for i, block in enumerate(content_blocks):
                block_type = getattr(block, 'type', 'unknown')
                logger.debug(f"  Block {i}: {block_type}")
                if block_type == 'thinking':
                    thinking_text = getattr(block, 'thinking', '')
                    logger.debug(f"    Thinking length: {len(thinking_text)}")
                elif block_type == 'text':
                    text = getattr(block, 'text', '')
                    logger.debug(f"    Text: {repr(text[:200])}...")
                elif block_type == 'tool_use':
                    name = getattr(block, 'name', 'unknown')
                    logger.debug(f"    Tool: {name}")

        return MessagesResponse(
            id=response_id,
            model=original_request.original_model or original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=content_blocks,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=prompt_tokens, output_tokens=completion_tokens),
        )

    except Exception as e:
        logger.error(f"Error converting response: {e}")
        return MessagesResponse(
            id=f"msg_error_{uuid.uuid4()}",
            model=original_request.original_model or original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=[
                ContentBlockText(
                    type=Constants.CONTENT_TEXT, text="Response conversion error"
                )
            ],
            stop_reason=Constants.STOP_ERROR,
            usage=Usage(input_tokens=0, output_tokens=0),
        )


def _send_message_start_event(message_id: str, model: str):
    """Send message_start event."""
    message_data = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        },
    }
    return f"event: message_start\ndata: {json.dumps(message_data)}\n\n"


def _send_content_block_start_event(index: int, block_type: str, **kwargs):
    """Send content_block_start event."""
    content_block = {"type": block_type, **kwargs}
    if block_type == "text":
        content_block["text"] = ""
    elif block_type == "tool_use":
        # Ensure tool_use blocks have required fields
        if "id" not in content_block:
            content_block["id"] = f"toolu_{uuid.uuid4().hex[:24]}"
        if "name" not in content_block:
            content_block["name"] = ""
        if "input" not in content_block:
            content_block["input"] = {}
    return f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': content_block})}\n\n"


# see https://docs.anthropic.com/en/docs/build-with-claude/streaming#event-types
def _send_content_block_delta_event(index: int, delta_type: str, content: str):
    """Send content_block_delta event."""
    delta = {"type": delta_type}
    if delta_type == "text_delta":
        delta["text"] = content
    elif delta_type == "input_json_delta":
        delta["partial_json"] = content
    elif delta_type == "thinking_delta":
        delta["thinking"] = content
    elif delta_type == "signature_delta":
        delta["signature"] = content
    return f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': index, 'delta': delta})}\n\n"


def _send_content_block_stop_event(index: int):
    """Send content_block_stop event."""
    return f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"


def _normalize_block(block):
    if isinstance(block, BaseModel):
        return block.model_dump(exclude_none=True)  # pydantic ≥v2
    return block


def _send_message_delta_event(
    stop_reason: str, output_tokens: int, content_blocks=None
):
    """Send message_delta event."""
    usage = {"output_tokens": output_tokens}
    delta = {"stop_reason": stop_reason, "stop_sequence": None}

    # Include content blocks in the delta if provided (required for proper Anthropic format)
    if content_blocks is not None:
        # Convert dict content blocks to proper Pydantic models
        converted_blocks = [_normalize_block(b) for b in content_blocks]
        delta["content"] = converted_blocks

    return f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': delta, 'usage': usage})}\n\n"


def _send_message_stop_event():
    """Send message_stop event."""
    return f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


def _send_ping_event():
    """Send ping event."""
    return f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"


def _map_finish_reason_to_stop_reason(finish_reason: str) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    if finish_reason == "length":
        return "max_tokens"
    elif finish_reason == "tool_calls":
        return "tool_use"
    else:
        return "end_turn"


async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from OpenAI SDK and convert to Anthropic format."""
    has_sent_stop_reason = False
    is_tool_use = False
    input_tokens = 0
    output_tokens = 0
    current_content_blocks = []
    accumulated_text = ""
    accumulated_thinking = ""
    try:
        # Send initial events
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield _send_message_start_event(message_id, original_request.model)
        yield _send_ping_event()

        # State tracking variables
        content_block_index = 0
        text_block_started = False
        text_block_closed = False

        # Tool call state
        tool_json = ""
        current_tool_id = None
        current_tool_name = None

        # Thinking/reasoning state
        thinking_block_started = False
        thinking_block_closed = False

        logger.debug(f"Starting streaming for model: {original_request.model}")

        # Process each chunk
        async for chunk in response_generator:
            try:
                # Check if this is the end of the response with usage data
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    if hasattr(chunk.usage, "prompt_tokens"):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, "completion_tokens"):
                        output_tokens = chunk.usage.completion_tokens
                    # Calculate cost based on token counts
                    model_to_use = original_request.model
                    if model_to_use.startswith("custom/"):
                        model_name = model_to_use[7:]  # Remove 'custom/' prefix
                        if model_name in CUSTOM_OPENAI_MODELS:
                            config = CUSTOM_OPENAI_MODELS[model_name]
                            input_cost = config.get(
                                "input_cost_per_token",
                                ModelDefaults.DEFAULT_INPUT_COST_PER_TOKEN,
                            )
                            output_cost = config.get(
                                "output_cost_per_token",
                                ModelDefaults.DEFAULT_OUTPUT_COST_PER_TOKEN,
                            )
                            total_cost = (input_tokens * input_cost) + (
                                output_tokens * output_cost
                            )
                            logger.info(
                                f"Stream complete - Model: {model_to_use}, Input tokens: {input_tokens}, Output tokens: {output_tokens}, Cost: ${total_cost:.8f}"
                            )
                        else:
                            logger.warning(
                                f"Could not find model configuration for {model_to_use}"
                            )

                    # Update session statistics for streaming
                    update_session_stats(
                        model=model_to_use,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                # Handle content from choices
                if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                    choice = chunk.choices[0]

                    # Get the delta from the choice
                    if hasattr(choice, "delta"):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, "message", {})

                    # Check for finish_reason to know when we're done
                    if hasattr(choice, "finish_reason"):
                        finish_reason = choice.finish_reason
                    elif isinstance(choice, dict):
                        finish_reason = choice.finish_reason
                    else:
                        finish_reason = None

                    # Handle tool calls first (they take priority)
                    delta_tool_calls = None
                    if hasattr(delta, "tool_calls"):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and "tool_calls" in delta:
                        delta_tool_calls = delta["tool_calls"]

                    if delta_tool_calls:
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]

                        for tool_call in delta_tool_calls:
                            # If we haven't started a tool yet, we need to handle any accumulated text first
                            if not is_tool_use:
                                # If we have accumulated text, send it first
                                if accumulated_text and not text_block_started:
                                    text_block = {"type": "text", "text": ""}
                                    current_content_blocks.append(text_block)
                                    yield _send_content_block_start_event(
                                        content_block_index, "text"
                                    )
                                    text_block_started = True

                                    # Send the accumulated text
                                    yield _send_content_block_delta_event(
                                        content_block_index,
                                        "text_delta",
                                        accumulated_text,
                                    )

                                    # Update the content block
                                    current_content_blocks[content_block_index][
                                        "text"
                                    ] = accumulated_text

                                    # Close the text block
                                    yield _send_content_block_stop_event(
                                        content_block_index
                                    )
                                    text_block_closed = True
                                    content_block_index += 1
                                elif text_block_started and not text_block_closed:
                                    # Close any open text block
                                    yield _send_content_block_stop_event(
                                        content_block_index
                                    )
                                    text_block_closed = True
                                    content_block_index += 1
                                elif not text_block_started and not text_block_closed:
                                    # Even if no text, we might need to close the implicit text block
                                    text_block = {"type": "text", "text": ""}
                                    current_content_blocks.append(text_block)
                                    yield _send_content_block_start_event(
                                        content_block_index, "text"
                                    )
                                    yield _send_content_block_stop_event(
                                        content_block_index
                                    )
                                    text_block_closed = True
                                    content_block_index += 1

                                # Now start the tool use block
                                is_tool_use = True

                                # Extract tool info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get("function", {})
                                    current_tool_name = (
                                        function.get("name", "")
                                        if isinstance(function, dict)
                                        else ""
                                    )
                                    current_tool_id = tool_call.get(
                                        "id", f"toolu_{uuid.uuid4().hex[:24]}"
                                    )
                                else:
                                    function = getattr(tool_call, "function", None)
                                    current_tool_name = (
                                        getattr(function, "name", "")
                                        if function
                                        else ""
                                    )
                                    current_tool_id = getattr(
                                        tool_call,
                                        "id",
                                        f"toolu_{uuid.uuid4().hex[:24]}",
                                    )

                                # Create tool use block
                                tool_block = {
                                    "type": "tool_use",
                                    "id": current_tool_id,
                                    "name": current_tool_name,
                                    "input": {},
                                }
                                current_content_blocks.append(tool_block)

                                yield _send_content_block_start_event(
                                    content_block_index,
                                    "tool_use",
                                    id=current_tool_id,
                                    name=current_tool_name,
                                )
                                tool_json = ""

                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and "function" in tool_call:
                                function = tool_call.get("function", {})
                                arguments = (
                                    function.get("arguments", "")
                                    if isinstance(function, dict)
                                    else ""
                                )
                            elif hasattr(tool_call, "function"):
                                function = getattr(tool_call, "function", None)
                                arguments = (
                                    getattr(function, "arguments", "")
                                    if function
                                    else ""
                                )

                            # If we have arguments, send them as a delta
                            if arguments:
                                tool_json += arguments

                                # Try to parse JSON to update the content block
                                try:
                                    parsed_json = json.loads(tool_json)
                                    current_content_blocks[content_block_index][
                                        "input"
                                    ] = parsed_json
                                except json.JSONDecodeError:
                                    # JSON not yet complete, continue accumulating
                                    pass

                                # Send the delta
                                yield _send_content_block_delta_event(
                                    content_block_index, "input_json_delta", arguments
                                )

                    # Handle reasoning/thinking content first (from deepseek reasoning models)
                    delta_reasoning = None
                    if hasattr(delta, "reasoning_content"):
                        delta_reasoning = delta.reasoning_content
                    elif isinstance(delta, dict) and "reasoning_content" in delta:
                        delta_reasoning = delta["reasoning_content"]

                    if delta_reasoning is not None and delta_reasoning != "":
                        accumulated_thinking += delta_reasoning

                        # Start thinking block if not started
                        if not thinking_block_started:
                            thinking_block = {"type": "thinking", "thinking": ""}
                            current_content_blocks.append(thinking_block)
                            yield _send_content_block_start_event(
                                content_block_index, "thinking"
                            )
                            thinking_block_started = True

                        # Send thinking delta
                        yield _send_content_block_delta_event(
                            content_block_index, "thinking_delta", delta_reasoning
                        )

                        # Update content block
                        if content_block_index < len(current_content_blocks):
                            current_content_blocks[content_block_index]["thinking"] = (
                                accumulated_thinking
                            )

                    # Check if reasoning is complete (no more reasoning deltas and we have normal content)
                    elif thinking_block_started and not thinking_block_closed:
                        # If we have normal content coming and thinking was active, close thinking block
                        delta_content = None
                        if hasattr(delta, "content"):
                            delta_content = delta.content
                        elif isinstance(delta, dict) and "content" in delta:
                            delta_content = delta["content"]

                        if delta_content is not None and delta_content != "":
                            # Close thinking block and prepare for text block
                            # Generate signature for completed thinking content
                            if content_block_index < len(current_content_blocks):
                                thinking_signature = generate_thinking_signature(
                                    accumulated_thinking
                                )
                                current_content_blocks[content_block_index][
                                    "signature"
                                ] = thinking_signature
                                # Send signature delta before closing thinking block
                                yield _send_content_block_delta_event(
                                    content_block_index,
                                    "signature_delta",
                                    thinking_signature,
                                )
                            yield _send_content_block_stop_event(content_block_index)
                            thinking_block_closed = True
                            content_block_index += 1

                    # Handle text content - check for text content first
                    delta_content = None
                    if hasattr(delta, "content"):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and "content" in delta:
                        delta_content = delta["content"]

                    # If we have text content and we're currently in tool use mode, end the tool use first
                    if delta_content is not None and delta_content != "" and is_tool_use:
                        # End current tool call block
                        yield _send_content_block_stop_event(content_block_index)
                        content_block_index += 1
                        is_tool_use = False
                        # Reset tool state
                        tool_json = ""
                        current_tool_id = None
                        current_tool_name = None

                    # Handle text content
                    if delta_content is not None and delta_content != "" and not is_tool_use:
                        accumulated_text += delta_content

                        # Start text block if not started
                        if not text_block_started:
                            text_block = {"type": "text", "text": ""}
                            current_content_blocks.append(text_block)
                            yield _send_content_block_start_event(
                                content_block_index, "text"
                            )
                            text_block_started = True

                        # Send text delta
                        yield _send_content_block_delta_event(
                            content_block_index, "text_delta", delta_content
                        )

                        # Update content block
                        if content_block_index < len(current_content_blocks):
                            current_content_blocks[content_block_index]["text"] = (
                                accumulated_text
                            )

                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        logger.debug(f"Processing finish_reason: {finish_reason}")
                        has_sent_stop_reason = True

                        # Close thinking block if it was started
                        if thinking_block_started and not thinking_block_closed:
                            # Generate signature for completed thinking content
                            if content_block_index < len(current_content_blocks):
                                thinking_signature = generate_thinking_signature(
                                    accumulated_thinking
                                )
                                current_content_blocks[content_block_index][
                                    "signature"
                                ] = thinking_signature
                                # Send signature delta before closing thinking block
                                yield _send_content_block_delta_event(
                                    content_block_index,
                                    "signature_delta",
                                    thinking_signature,
                                )
                            yield _send_content_block_stop_event(content_block_index)
                            thinking_block_closed = True
                            content_block_index += 1

                        # If we haven't started any blocks yet, start and immediately close a text block
                        if (
                            not text_block_started
                            and not is_tool_use
                            and not thinking_block_started
                        ):
                            text_block = {"type": "text", "text": ""}
                            current_content_blocks.append(text_block)
                            yield _send_content_block_start_event(
                                content_block_index, "text"
                            )
                            yield _send_content_block_stop_event(content_block_index)
                        elif text_block_started or is_tool_use:
                            # Close the current content block
                            yield _send_content_block_stop_event(content_block_index)

                        # Determine stop reason
                        stop_reason = _map_finish_reason_to_stop_reason(finish_reason)
                        logger.debug(f"Mapped stop_reason: {stop_reason}")

                        # Send message delta with final content and stop reason
                        yield _send_message_delta_event(
                            stop_reason, output_tokens, current_content_blocks
                        )

                        # Send message stop
                        yield _send_message_stop_event()
                        logger.debug("Streaming completed successfully")
                        has_sent_stop_reason = True
                        return

            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue

        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            logger.debug("No finish_reason received, closing stream manually")
            # If we haven't started any blocks yet, start and immediately close a text block
            if not text_block_started and not is_tool_use:
                text_block = {"type": "text", "text": ""}
                current_content_blocks.append(text_block)
                yield _send_content_block_start_event(content_block_index, "text")
                yield _send_content_block_stop_event(content_block_index)
            elif text_block_started or is_tool_use:
                # Close the current content block if there is one
                yield _send_content_block_stop_event(content_block_index)

            # Send final events
            stop_reason = "tool_use" if is_tool_use else "end_turn"
            yield _send_message_delta_event(
                stop_reason, output_tokens, current_content_blocks
            )
            yield _send_message_stop_event()
            logger.debug("Streaming completed without finish_reason")

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = (
            f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        )
        logger.error(error_message)

        # Send error events
        logger.error(f"Streaming error: {error_message}")
        yield _send_message_delta_event("error", 0, [])
        yield _send_message_stop_event()

    finally:
        if not has_sent_stop_reason:
            stop_reason = "tool_use" if is_tool_use else "end_turn"
            yield _send_message_delta_event(
                stop_reason, output_tokens, current_content_blocks
            )
            yield _send_message_stop_event()
            logger.debug(
                "Streaming finally: emitted missing message_delta + message_stop"
            )

        # Fallback: Update session stats if not already done (when input_tokens is 0)
        if input_tokens == 0:
            logger.debug("No usage info from streaming, calculating tokens manually")
            try:
                # Calculate input tokens from original request
                calculated_input_tokens = calculate_request_tokens(original_request)

                # Calculate output tokens from accumulated content
                calculated_output_tokens = count_tokens_in_response(
                    response_content=accumulated_text,
                    thinking_content=accumulated_thinking,
                    tool_calls=None,  # Tool calls are handled separately in streaming
                )

                # Update session stats with calculated values
                update_session_stats(
                    model=original_request.model,
                    input_tokens=calculated_input_tokens,
                    output_tokens=calculated_output_tokens,
                )
                logger.debug(
                    f"Updated session stats from manual streaming calculation: input={calculated_input_tokens}, output={calculated_output_tokens}"
                )

            except Exception as e:
                logger.error(f"Error in manual streaming token calculation: {e}")
                # Use conservative estimates as last resort
                update_session_stats(
                    model=original_request.model,
                    input_tokens=0,  # Stream fallback: input already counted elsewhere
                    output_tokens=len(accumulated_text.split())
                    if accumulated_text
                    else 50,  # Word-based estimate
                )
                logger.debug("Updated session stats with fallback streaming estimates")


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    try:
        # Parse the raw body as JSON since it's bytes
        body = await raw_request.body()
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", "unknown")

        # Calculate token count for routing decisions
        token_count = calculate_request_tokens(request)

        # Check if thinking is enabled
        has_thinking = False
        if request.thinking is not None:
            has_thinking = request.thinking.model_dump().get("type") == "enabled"
            logger.debug(
                f"🧠 Thinking type check: {request.thinking.type}, enabled: {has_thinking}"
            )
        logger.info(f"🧠 Final thinking decision: has_thinking={has_thinking}")

        # Use router to determine the actual model to use
        routed_model = determine_model_by_router(
            original_model, token_count, has_thinking
        )

        # Update the request model with routed model
        request.model = routed_model

        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        # Get the routed display name
        routed_display_model = routed_model
        if "/" in routed_display_model:
            routed_display_model = routed_display_model.split("/")[-1]

        logger.info(
            f"📊 PROCESSING REQUEST: Original={original_model} → Routed={routed_model}, Tokens={token_count}, Stream={request.stream}"
        )

        # Convert Anthropic request to OpenAI format
        openai_request = convert_anthropic_to_openai_request(request, request.model)

        # Create OpenAI client for the model
        client, model_name = create_openai_client(request.model, is_async=True)
        openai_request["model"] = model_name

        # Only log basic info about the request, not the full details
        logger.debug(
            f"Request for model: {openai_request.get('model')}, stream: {openai_request.get('stream', False)}"
        )

        # Handle streaming mode
        if request.stream:
            # Use OpenAI SDK for streaming
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                f"{display_model} → {routed_display_model}",
                openai_request.get("model"),
                len(openai_request["messages"]),
                num_tools,
                200,  # Assuming success at this point
            )
            # Use OpenAI SDK async streaming
            response_generator = await client.chat.completions.create(**openai_request)

            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream",
            )
        else:
            # Use OpenAI SDK for regular completion
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                f"{display_model} → {routed_display_model}",
                openai_request.get("model"),
                len(openai_request["messages"]),
                num_tools,
                200,  # Assuming success at this point
            )
            start_time = time.time()
            openai_response = await client.chat.completions.create(**openai_request)

            logger.debug(
                f"✅ RESPONSE RECEIVED: Model={openai_request.get('model')}, Time={time.time() - start_time:.2f}s"
            )

            # Convert OpenAI response to Anthropic format
            anthropic_response = convert_openai_to_anthropic(openai_response, request)

            # Extract usage to update session stats
            if hasattr(openai_response, "usage") and openai_response.usage:
                prompt_tokens = openai_response.usage.prompt_tokens
                completion_tokens = openai_response.usage.completion_tokens
                update_session_stats(
                    model=openai_request.get("model"),
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                )
                logger.debug(
                    f"Updated session stats from OpenAI usage: input={prompt_tokens}, output={completion_tokens}"
                )
            else:
                # Fallback: Calculate tokens manually using tiktoken
                logger.debug(
                    "No usage info from OpenAI response, calculating tokens manually"
                )
                try:
                    # Calculate input tokens from original request
                    input_tokens = calculate_request_tokens(request)

                    # Calculate output tokens from response
                    response_content = ""
                    thinking_content = ""
                    tool_calls = None

                    # Extract content from anthropic_response
                    if (
                        hasattr(anthropic_response, "content")
                        and anthropic_response.content
                    ):
                        for content_block in anthropic_response.content:
                            if hasattr(content_block, "text"):
                                response_content += content_block.text
                            elif (
                                hasattr(content_block, "type")
                                and content_block.type == "tool_use"
                            ):
                                if tool_calls is None:
                                    tool_calls = []
                                tool_calls.append(
                                    {
                                        "name": getattr(content_block, "name", ""),
                                        "input": getattr(content_block, "input", {}),
                                    }
                                )

                    # Extract thinking content if present
                    if (
                        hasattr(anthropic_response, "thinking")
                        and anthropic_response.thinking
                    ):
                        thinking_content = anthropic_response.thinking

                    output_tokens = count_tokens_in_response(
                        response_content, thinking_content, tool_calls
                    )

                    # Update session stats with calculated values
                    update_session_stats(
                        model=openai_request.get("model"),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                    logger.debug(
                        f"Updated session stats from manual calculation: input={input_tokens}, output={output_tokens}"
                    )

                except Exception as e:
                    logger.error(f"Error in manual token calculation: {e}")
                    # Use conservative estimates as last resort
                    update_session_stats(
                        model=openai_request.get("model"),
                        input_tokens=token_count,  # We already calculated this earlier
                        output_tokens=100,  # Conservative estimate
                    )
                    logger.debug(
                        f"Updated session stats with fallback estimates: input={token_count}, output=100"
                    )

            return anthropic_response

    except Exception as e:
        error_details = _extract_error_details(e)
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")

        error_message = _format_error_message(e, error_details)
        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message)


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model

        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/") :]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/") :]

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
            token_count = calculate_request_tokens(request)

            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)

        except Exception as e:
            logger.error(f"Error in local token counting: {e}")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


@app.get("/v1/stats", response_model=SessionStats)
async def get_stats():
    """Returns the token usage statistics for the current session."""
    return session_stats


@app.post("/v1/messages/test_conversion")
async def test_message_conversion(request: MessagesRequest, raw_request: Request):
    """
    Test endpoint for direct message format conversion without routing.

    This endpoint converts Anthropic format to LiteLLM format and sends the request
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
        openai_request = convert_anthropic_to_openai_request(request, original_model)

        # Create OpenAI client for the model
        client, model_name = create_openai_client(original_model, is_async=True)
        openai_request["model"] = model_name

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
                handle_streaming(response_generator, request),
                media_type="text/event-stream",
            )
        else:
            # Regular completion
            logger.info(f"🧪 Starting direct completion test for {original_model}")
            start_time = time.time()
            openai_response = await client.chat.completions.create(**openai_request)
            logger.info(f"🧪 Direct test completed in {time.time() - start_time:.2f}s")

            # Convert OpenAI response to Anthropic format
            anthropic_response = convert_openai_to_anthropic(openai_response, request)
            return anthropic_response

    except Exception as e:
        error_details = _extract_error_details(e)
        logger.error(
            f"🧪 Error in test conversion: {json.dumps(error_details, indent=2)}"
        )

        error_message = _format_error_message(e, error_details)
        status_code = error_details.get("status_code", 500)
        raise HTTPException(status_code=status_code, detail=error_message)


@app.get("/health")
async def health_check():
    """Enhanced health check with detailed proxy server information"""
    try:
        # Gather API key status for different providers
        api_status = {
            "anthropic": bool(config.anthropic_api_key),
            "openai": bool(config.openai_api_key),
            "gemini": bool(config.gemini_api_key),
            "custom_models": len(CUSTOM_OPENAI_MODELS),
        }

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "proxy_type": "Universal Anthropic API Proxy",
            "supported_providers": {
                "custom": {
                    "configured": api_status["custom_models"] > 0,
                    "count": api_status["custom_models"],
                    "models": list(CUSTOM_OPENAI_MODELS.keys())[:3]
                    if api_status["custom_models"] > 0
                    else [],
                },
            },
            "router_config": config.router_config,
            "features": {
                "streaming": True,
                "tool_use": True,
                "image_support": True,
                "cost_tracking": True,
                "model_mapping": True,
            },
        }

        return health_status

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": "Health check failed",
                "message": str(e),
            },
        )


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


@app.get("/")
async def root():
    """Enhanced root endpoint with comprehensive proxy information"""
    return {
        "message": "Universal Anthropic API Proxy for LiteLLM",
        "description": "Supports Anthropic Claude, OpenAI, Gemini, and custom OpenAI-compatible models",
        "version": "2.0.0",
        "status": "running",
        "capabilities": {
            "providers": ["custom"],
            "features": [
                "streaming",
                "tool_use",
                "image_support",
                "cost_tracking",
                "model_mapping",
            ],
        },
        "configuration": {
            "router_config": config.router_config,
            "custom_models_count": len(CUSTOM_OPENAI_MODELS),
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
        },
        "documentation": {
            "anthropic_format": "Uses Anthropic Claude API format",
            "intelligent_routing": "Automatically routes models based on token count, 'thinking' flag, and model name.",
            "streaming": "Supports streaming responses with proper SSE format",
            "tools": "Full tool use support with proper formatting",
        },
    }


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
    uvicorn.run(
        app, host=config.host, port=config.port, log_level=config.log_level.lower()
    )
