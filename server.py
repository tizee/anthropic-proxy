import yaml
from typing import Dict, List, Optional, Union, Literal, Any
import os.path

from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, field_validator
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys

# Load environment variables from .env file
load_dotenv()

class Config:
    """Universal proxy server configuration"""
    def __init__(self):
        # API Keys
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")

        # Provider preference and model mapping
        self.preferred_provider = os.environ.get("PREFERRED_PROVIDER", "google").lower()

        # Set default models based on preferred provider
        if self.preferred_provider == "google":
            default_big_model = "gemini-2.5-pro-preview-03-25"
            default_small_model = "gemini-2.0-flash"
        elif self.preferred_provider == "openai":
            default_big_model = "gpt-4o"
            default_small_model = "gpt-4o-mini"
        else:  # custom
            default_big_model = "custom-large-model"
            default_small_model = "custom-small-model"

        self.big_model = os.environ.get("BIG_MODEL", default_big_model)
        self.small_model = os.environ.get("SMALL_MODEL", default_small_model)

        # Server configuration
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8082"))
        self.log_level = os.environ.get("LOG_LEVEL", "WARNING")

        # Request limits and timeouts
        self.max_tokens_limit = int(os.environ.get("MAX_TOKENS_LIMIT", "16384"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "2"))

        # Custom models configuration file
        self.custom_models_file = os.environ.get("CUSTOM_MODELS_FILE", "custom_models.yaml")

        # Custom API keys storage
        self.custom_api_keys = {}

    def validate_api_keys(self):
        """Validate that at least one provider API key is configured"""
        providers_configured = []

        if self.anthropic_api_key:
            providers_configured.append("anthropic")

        if self.openai_api_key:
            providers_configured.append("openai")

        if self.gemini_api_key:
            providers_configured.append("gemini")

        return providers_configured

    def add_custom_api_key(self, key_name: str, key_value: str):
        """Add a custom API key"""
        self.custom_api_keys[key_name] = key_value

    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        if provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "openai":
            return self.openai_api_key
        elif provider == "gemini":
            return self.gemini_api_key
        elif provider in self.custom_api_keys:
            return self.custom_api_keys[provider]
        return None

# Initialize configuration
try:
    config = Config()
    if config.log_level.lower() == "debug":
        # DEBUG mode
        litellm._turn_on_debug()
    print(f"✅ Configuration loaded: Providers={config.validate_api_keys()}, Preferred='{config.preferred_provider}', BIG='{config.big_model}', SMALL='{config.small_model}'")
except Exception as e:
    print(f"🔴 Configuration Error: {e}")
    sys.exit(1)

# LiteLLM settings
# drop parameters when changing models
litellm.drop_params = True
litellm.num_retries = config.max_retries

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level.upper()),
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn
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
            "cost_calculator"
        ]

        if hasattr(record, 'msg') and isinstance(record.msg, str):
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
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

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
        with open(config_file, 'r') as file:
            models = yaml.safe_load(file)

        if not models:
            logger.warning(f"No models found in config file: {config_file}")
            return

        # Dictionary to register models with LiteLLM pricing system
        models_to_register = {}

        for model in models:
            if "model_id" not in model or "api_base" not in model:
                logger.warning(f"Invalid model configuration, missing required fields: {model}")
                continue

            model_id = model["model_id"]
            model_name = model.get("model_name", model_id)

            # Set default pricing if not provided
            input_cost = model.get("input_cost_per_token", 0.000001)  # Default if not specified
            output_cost = model.get("output_cost_per_token", 0.000002)  # Default if not specified

            CUSTOM_OPENAI_MODELS[model_id] = {
                "model_name": model_name,
                "api_base": model["api_base"],
                "api_key_name": model.get("api_key_name", "OPENAI_API_KEY"),
                "can_stream": model.get("can_stream", True),
                "max_tokens": model.get("max_tokens", 8192),
                "input_cost_per_token": input_cost,
                "output_cost_per_token": output_cost,
                "max_input_tokens": model.get("max_input_tokens", 128000),
                "supports_response_schema": True,
                "supports_tool_choice": True,
                "enable_thinking": model.get("enable_thinking", False),
                "reasoning_effort": model.get("reasoning_effort", None)
            }

            # Register model pricing with LiteLLM - ensure all possible model names are registered
            # Format: Register each possible way the model might be referenced
            model_variations = [
                f"openai/{model_id}",         # With openai/ prefix
                f"openai/{model_name}",       # With openai/ prefix
            ]

            for variation in model_variations:
                models_to_register[variation] = {
                    "max_tokens": CUSTOM_OPENAI_MODELS[model_id]["max_tokens"],
                    "input_cost_per_token": input_cost,
                    "output_cost_per_token": output_cost,
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "max_input_tokens": model.get("max_input_tokens", 128000),
                    "supports_response_schema": True,
                    "supports_tool_choice":True
                }

            logger.info(f"Loaded custom OpenAI-compatible model: {model_id} → {model_name}")

        # Register all model pricing with LiteLLM
        if models_to_register:
            try:
                from litellm import register_model
                # Register models with LiteLLM pricing system
                register_model(models_to_register)
                logger.info(f"Registered {len(models_to_register)} model variations with LiteLLM pricing system")

                # Verify registration (add debug output to confirm)
                from litellm import model_cost
                for model_key in models_to_register:
                    if model_key in model_cost:
                        logger.info(f"✅ Successfully registered model pricing for: {model_key} = {model_cost[model_key]}")
                    else:
                        logger.warning(f"⚠️ Failed to verify pricing registration for: {model_key}")

            except Exception as e:
                logger.error(f"Failed to register custom models with LiteLLM pricing system: {str(e)}")

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

# List of OpenAI models
OPENAI_MODELS = [
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    "gpt-4.5-preview",
    "gpt-4o",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "gpt-4o-mini-audio-preview"
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro-preview-0514",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-preview-0514",
    "gemini-pro",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash-exp",
    "gemini-exp-1206"
]

# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = True

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model_field(cls, v, info): # Renamed to avoid conflict
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{config.preferred_provider}', BIG='{config.big_model}', SMALL='{config.small_model}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('custom/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False

        # Check for custom models first
        if clean_v in CUSTOM_OPENAI_MODELS:
            new_model = f"custom/{clean_v}"
            mapped = True

        # Map Haiku to SMALL_MODEL based on provider preference
        if 'haiku' in clean_v.lower():
            if config.preferred_provider == "google" and config.small_model in GEMINI_MODELS:
                new_model = f"gemini/{config.small_model}"
                mapped = True
            elif config.preferred_provider == "openai" and config.small_model in OPENAI_MODELS:
                new_model = f"openai/{config.small_model}"
                mapped = True
            elif config.preferred_provider == "custom" and config.small_model in CUSTOM_OPENAI_MODELS:
                new_model = f"custom/{config.small_model}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if config.preferred_provider == "google" and config.big_model in GEMINI_MODELS:
                new_model = f"gemini/{config.big_model}"
                mapped = True
            elif config.preferred_provider == "openai" and config.big_model in OPENAI_MODELS:
                new_model = f"openai/{config.big_model}"
                mapped = True
            elif config.preferred_provider == "custom" and config.big_model in CUSTOM_OPENAI_MODELS:
                new_model = f"custom/{config.big_model}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in CUSTOM_OPENAI_MODELS and not v.startswith('custom/'):
                new_model = f"custom/{clean_v}"
                mapped = True  # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.info(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             # If no mapping occurred and no prefix exists, log warning or decide default
             if not v.startswith(('openai/', 'gemini/', 'anthropic/', 'custom/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator('model')
    def validate_model_token_count(cls, v, info): # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        # NOTE: Pydantic validators might not share state easily if not class methods
        # Re-implementing the logic here for clarity, could be refactored
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{config.preferred_provider}', BIG='{config.big_model}', SMALL='{config.small_model}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('custom/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False

        # Check for custom models first
        if clean_v in CUSTOM_OPENAI_MODELS:
            new_model = f"custom/{clean_v}"
            mapped = True

        # Map Haiku to SMALL_MODEL based on provider preference
        if 'haiku' in clean_v.lower():
            if config.preferred_provider == "google" and config.small_model in GEMINI_MODELS:
                new_model = f"gemini/{config.small_model}"
                mapped = True
            elif config.preferred_provider == "openai" and config.small_model in OPENAI_MODELS:
                new_model = f"openai/{config.small_model}"
                mapped = True
            elif config.preferred_provider == "custom" and config.small_model in CUSTOM_OPENAI_MODELS:
                new_model = f"custom/{config.small_model}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if config.preferred_provider == "google" and config.big_model in GEMINI_MODELS:
                new_model = f"gemini/{config.big_model}"
                mapped = True
            elif config.preferred_provider == "openai" and config.big_model in OPENAI_MODELS:
                new_model = f"openai/{config.big_model}"
                mapped = True
            elif config.preferred_provider == "custom" and config.big_model in CUSTOM_OPENAI_MODELS:
                new_model = f"custom/{config.big_model}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in CUSTOM_OPENAI_MODELS and not v.startswith('custom/'):
                new_model = f"custom/{clean_v}"
                mapped = True  # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             if not v.startswith(('openai/', 'gemini/', 'anthropic/', 'custom/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence",
                                  "tool_use", "error"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

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

# Not using validation function as we're using the environment API key

def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"

def _extract_system_text(system_content) -> str:
    """Extract system text from various system content formats."""
    if isinstance(system_content, str):
        return system_content
    elif isinstance(system_content, list):
        text_parts = []
        for block in system_content:
            if hasattr(block, 'type') and block.type == Constants.CONTENT_TEXT:
                text_parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT:
                text_parts.append(block.get("text", ""))
        return "\n\n".join(text_parts)
    return ""

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (supports Gemini, OpenAI, and custom models)."""

    # Determine the target model type
    is_gemini_model = anthropic_request.model.startswith("gemini/")
    is_openai_model = anthropic_request.model.startswith("openai/")
    is_custom_model = anthropic_request.model.startswith("custom/")

    logger.debug(f"Converting messages for model type: gemini={is_gemini_model}, openai={is_openai_model}, custom={is_custom_model}")

    litellm_messages = []

    # System message handling
    if anthropic_request.system:
        system_text = _extract_system_text(anthropic_request.system)
        if system_text.strip():
            litellm_messages.append({"role": Constants.ROLE_SYSTEM, "content": system_text.strip()})
            logger.debug(f"Added system message: {len(system_text)} characters")

    # Process messages
    for i, msg in enumerate(anthropic_request.messages):
        logger.debug(f"Processing message {i+1}/{len(anthropic_request.messages)}: role={msg.role}, type={type(msg.content)}")
        if isinstance(msg.content, str):
            litellm_messages.append({"role": msg.role, "content": msg.content})
            continue

        # Process content blocks - accumulate different types
        text_parts = []
        image_parts = []
        tool_calls = []
        pending_tool_messages = []

        logger.debug(f"Processing {len(msg.content)} content blocks")
        for j, block in enumerate(msg.content):
            logger.debug(f"Block {j+1}: type={block.type}")
            if block.type == Constants.CONTENT_TEXT:
                text_parts.append(block.text)
            elif block.type == Constants.CONTENT_IMAGE:
                if (isinstance(block.source, dict) and
                    block.source.get("type") == "base64" and
                    "media_type" in block.source and "data" in block.source):
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.source['media_type']};base64,{block.source['data']}"
                        }
                    })
            elif block.type == Constants.CONTENT_TOOL_USE and msg.role == Constants.ROLE_ASSISTANT:
                tool_calls.append({
                    "id": block.id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
            elif block.type == Constants.CONTENT_TOOL_RESULT and msg.role == Constants.ROLE_USER:
                if is_gemini_model:
                    # Gemini: Split user message when tool_result is encountered
                    if text_parts or image_parts:
                        content_parts = []
                        text_content = "".join(text_parts).strip()
                        if text_content:
                            content_parts.append({"type": Constants.CONTENT_TEXT, "text": text_content})
                        content_parts.extend(image_parts)

                        litellm_messages.append({
                            "role": Constants.ROLE_USER,
                            "content": content_parts[0]["text"] if len(content_parts) == 1 and content_parts[0]["type"] == Constants.CONTENT_TEXT else content_parts
                        })
                        text_parts.clear()
                        image_parts.clear()

                    # Add tool result as separate "tool" role message
                    parsed_content = parse_tool_result_content(block.content)
                    pending_tool_messages.append({
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": block.tool_use_id,
                        "content": parsed_content
                    })
                else:
                    # OpenAI/Custom: Convert tool result to inline text
                    tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                    parsed_content = parse_tool_result_content(block.content)
                    text_parts.append(f"Tool result for {tool_id}:\n{parsed_content}\n")

        # Finalize message based on role
        if msg.role == Constants.ROLE_USER:
            # Add any remaining text/image content
            if text_parts or image_parts:
                if is_openai_model or is_custom_model:
                    # OpenAI/Custom: Convert to simple string format
                    text_content = "".join(text_parts).strip()
                    if image_parts:
                        text_content += "\n[Note: Image content present but not displayed in text format]"
                    if text_content:
                        litellm_messages.append({"role": Constants.ROLE_USER, "content": text_content})
                else:
                    # Gemini: Support structured content
                    content_parts = []
                    text_content = "".join(text_parts).strip()
                    if text_content:
                        content_parts.append({"type": Constants.CONTENT_TEXT, "text": text_content})
                    content_parts.extend(image_parts)

                    litellm_messages.append({
                        "role": Constants.ROLE_USER,
                        "content": content_parts[0]["text"] if len(content_parts) == 1 and content_parts[0]["type"] == Constants.CONTENT_TEXT else content_parts
                    })
            # Add any pending tool messages (only for Gemini)
            if is_gemini_model:
                litellm_messages.extend(pending_tool_messages)

        elif msg.role == Constants.ROLE_ASSISTANT:
            assistant_msg = {"role": Constants.ROLE_ASSISTANT}

            # Handle content for assistant messages
            if is_openai_model or is_custom_model:
                # OpenAI/Custom: Convert to simple string format
                text_content = "".join(text_parts).strip()
                if text_content:
                    assistant_msg["content"] = text_content
                elif not tool_calls:
                    assistant_msg["content"] = ""  # Empty content for OpenAI

                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
            else:
                # Gemini: Support structured content
                content_parts = []
                text_content = "".join(text_parts).strip()
                if text_content:
                    content_parts.append({"type": Constants.CONTENT_TEXT, "text": text_content})
                content_parts.extend(image_parts)

                if content_parts:
                    assistant_msg["content"] = content_parts[0]["text"] if len(content_parts) == 1 and content_parts[0]["type"] == Constants.CONTENT_TEXT else content_parts
                else:
                    assistant_msg["content"] = None

                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

            # Only add message if it has actual content or tool calls
            if assistant_msg.get("content") or assistant_msg.get("tool_calls"):
                litellm_messages.append(assistant_msg)

    # Build final LiteLLM request
    litellm_request = {
        "model": anthropic_request.model,
        "messages": litellm_messages,
        "max_tokens": min(anthropic_request.max_tokens, config.max_tokens_limit),
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Add optional parameters
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    if anthropic_request.top_p is not None:
        litellm_request["top_p"] = anthropic_request.top_p
    if anthropic_request.top_k is not None:
        litellm_request["topK"] = anthropic_request.top_k

    # Add tools with model-specific formatting
    if anthropic_request.tools:
        valid_tools = []
        for tool in anthropic_request.tools:
            if tool.name and tool.name.strip():
                # Clean schema for Gemini models only
                if is_gemini_model:
                    schema = clean_gemini_schema(tool.input_schema)
                else:
                    # Keep original schema for OpenAI/Custom models
                    schema = tool.input_schema

                valid_tools.append({
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema
                    }
                })
        if valid_tools:
            litellm_request["tools"] = valid_tools
            logger.debug(f"Added {len(valid_tools)} tools to request")

    # Add tool choice configuration
    if anthropic_request.tool_choice:
        choice_type = anthropic_request.tool_choice.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in anthropic_request.tool_choice:
            litellm_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": anthropic_request.tool_choice["name"]}
            }
        else:
            litellm_request["tool_choice"] = "auto"

    # Add thinking configuration based on model capabilities (following gemini-server.py pattern)
    if anthropic_request.thinking is not None:
        # Handle different thinking/reasoning systems based on model type
        if anthropic_request.model.startswith("gemini/"):
            # Gemini models use thinkingConfig (always set for Gemini)
            if anthropic_request.thinking.enabled:
                litellm_request["thinkingConfig"] = {"thinkingBudget": 24576}
            else:
                litellm_request["thinkingConfig"] = {"thinkingBudget": 0}
        elif anthropic_request.model.startswith("openai/") and anthropic_request.thinking.enabled:
            # OpenAI models use reasoning_effort (only when enabled)
            litellm_request["reasoning_effort"] = "medium"  # Default to medium
        elif anthropic_request.model.startswith("custom/"):
            # Check custom model configuration
            custom_model_name = anthropic_request.model[7:]  # Remove 'custom/' prefix
            if custom_model_name in CUSTOM_OPENAI_MODELS:
                model_config = CUSTOM_OPENAI_MODELS[custom_model_name]

                # Check if it supports Gemini-style thinking
                if model_config.get("enable_thinking", False):
                    if anthropic_request.thinking.enabled:
                        litellm_request["thinkingConfig"] = {"thinkingBudget": 24576}
                    else:
                        litellm_request["thinkingConfig"] = {"thinkingBudget": 0}
                # Check if it supports OpenAI-style reasoning_effort
                elif model_config.get("reasoning_effort") is not None and anthropic_request.thinking.enabled:
                    litellm_request["reasoning_effort"] = model_config.get("reasoning_effort")
        # Don't add any thinking parameters for models that don't support them
        # This prevents 422 errors from APIs that don't recognize the parameters

    # Add user metadata if provided
    if (anthropic_request.metadata and
        "user_id" in anthropic_request.metadata and
        isinstance(anthropic_request.metadata["user_id"], str)):
        litellm_request["user"] = anthropic_request.metadata["user_id"]

    return litellm_request

def clean_tool_markers(content: str) -> str:
    """Clean up tool call markers and malformed content."""
    if not content:
        return content

    # Remove specific tool call markers from your example
    content = re.sub(r'<｜tool▁call▁end｜>', '', content)
    content = re.sub(r'<\|tool_call_end\|>', '', content)

    # Remove bash/function blocks
    content = re.sub(r'```bash\s*function\s+\w+.*?```', '', content, flags=re.DOTALL)

    # Remove standalone function + json blocks but preserve other content
    content = re.sub(r'function\s+\w+\s*```json\s*{.*?}\s*```(?:<｜tool▁call▁end｜>)?', '', content, flags=re.DOTALL)

    # Remove standalone function names that appear by themselves
    content = re.sub(r'^\s*function\s+\w+\s*$', '', content, flags=re.MULTILINE)

    # Clean up excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    content = re.sub(r'^\s*\n+', '', content)  # Remove leading newlines
    content = content.strip()

    return content

def parse_tool_calls_from_content(content: str) -> list:
    """Extract tool calls from malformed content."""
    tool_calls = []

    # Look for multiple patterns of tool calls
    patterns = [
        # Pattern: function FunctionName \n ```json\n{...}\n```
        r'function\s+(\w+)\s*\n?\s*```json\s*\n?\s*({.*?})\s*\n?\s*```',
        # Pattern: function FunctionName ```json{...}```
        r'function\s+(\w+)\s*```json\s*({.*?})\s*```',
        # Pattern: function FunctionName followed by JSON on next lines
        r'function\s+(\w+)\s*\n\s*```json\s*\n({[\s\S]*?})\s*\n```'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            function_name = match[0]
            json_str = match[1].strip()
            try:
                arguments = json.loads(json_str)
                tool_calls.append({
                    "id": f"tool_{uuid.uuid4()}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                })
                logger.debug(f"Extracted tool call: {function_name}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON for function {function_name}: {json_str[:100]}... Error: {e}")

    # If no matches found, try a more flexible approach
    if not tool_calls:
        # Look for just function names followed by any JSON-like structure
        loose_pattern = r'function\s+(\w+).*?({[^}]*})'
        matches = re.findall(loose_pattern, content, re.DOTALL)
        for match in matches:
            function_name = match[0]
            json_str = match[1]
            try:
                # Try to fix common JSON issues
                json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                arguments = json.loads(json_str)
                tool_calls.append({
                    "id": f"tool_{uuid.uuid4()}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments)
                    }
                })
                logger.debug(f"Extracted tool call (loose): {function_name}")
            except json.JSONDecodeError:
                # Last resort - create tool call with raw string
                tool_calls.append({
                    "id": f"tool_{uuid.uuid4()}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps({"raw_input": json_str})
                    }
                })
                logger.warning(f"Created raw tool call for {function_name}: {json_str[:50]}...")

    return tool_calls

def _extract_response_data(litellm_response, default_response_id: str) -> dict:
    """Extract response data from LiteLLM response in various formats."""
    response_data = {
        "response_id": default_response_id,
        "raw_content": "",
        "tool_calls": None,
        "finish_reason": "stop",
        "prompt_tokens": 0,
        "completion_tokens": 0
    }

    # Handle LiteLLM ModelResponse object format
    if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
        choices = litellm_response.choices
        message = choices[0].message if choices else None
        response_data["raw_content"] = getattr(message, 'content', "") or ""
        response_data["tool_calls"] = getattr(message, 'tool_calls', None)
        response_data["finish_reason"] = choices[0].finish_reason if choices else "stop"
        response_data["response_id"] = getattr(litellm_response, 'id', default_response_id)

        if hasattr(litellm_response, 'usage'):
            usage = litellm_response.usage
            response_data["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
            response_data["completion_tokens"] = getattr(usage, "completion_tokens", 0)

    # Handle dictionary response format
    elif isinstance(litellm_response, dict):
        choices = litellm_response.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        response_data["raw_content"] = message.get("content", "") or ""
        response_data["tool_calls"] = message.get("tool_calls")
        response_data["finish_reason"] = choices[0].get("finish_reason", "stop") if choices else "stop"
        usage = litellm_response.get("usage", {})
        response_data["prompt_tokens"] = usage.get("prompt_tokens", 0)
        response_data["completion_tokens"] = usage.get("completion_tokens", 0)
        response_data["response_id"] = litellm_response.get("id", default_response_id)

    return response_data

def convert_litellm_to_anthropic(litellm_response, original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM response back to Anthropic API format."""
    try:
        # Extract response data safely
        response_id = f"msg_{uuid.uuid4()}"
        content_text = ""
        tool_calls = None
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0

        logger.debug(f"Converting LiteLLM response: {type(litellm_response)}")

        # Extract response data based on format
        response_data = _extract_response_data(litellm_response, response_id)
        response_id = response_data["response_id"]
        raw_content = response_data["raw_content"]
        tool_calls = response_data["tool_calls"]
        finish_reason = response_data["finish_reason"]
        prompt_tokens = response_data["prompt_tokens"]
        completion_tokens = response_data["completion_tokens"]

        logger.debug(f"Raw content extracted: {len(raw_content)} characters")

        # Clean up content text - remove tool call markers if present
        content_text = clean_tool_markers(raw_content) if raw_content else ""

        # Try to extract tool calls from malformed content if no proper tool_calls found
        if not tool_calls and raw_content and 'function' in raw_content:
            extracted_tool_calls = parse_tool_calls_from_content(raw_content)
            if extracted_tool_calls:
                tool_calls = extracted_tool_calls
                logger.debug(f"Extracted {len(tool_calls)} tool calls from malformed content")

        # Build content blocks
        content_blocks = []

        # Add text content if present
        if content_text:
            content_blocks.append(ContentBlockText(type=Constants.CONTENT_TEXT, text=content_text))

        # Process tool calls
        if tool_calls:
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for tool_call in tool_calls:
                try:
                    # Extract tool call data from different formats
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
                        continue

                    if not name:
                        continue

                    # Parse tool arguments safely
                    try:
                        arguments_dict = json.loads(arguments_str)
                        # Ensure arguments_dict is always a dictionary
                        if not isinstance(arguments_dict, dict):
                            # If it's a list or other type, wrap it in a dictionary
                            arguments_dict = {"input": arguments_dict}
                    except json.JSONDecodeError:
                        arguments_dict = {"raw_arguments": arguments_str}

                    content_blocks.append(ContentBlockToolUse(
                        type=Constants.CONTENT_TOOL_USE,
                        id=tool_id,
                        name=name,
                        input=arguments_dict
                    ))
                except Exception as e:
                    logger.warning(f"Error processing tool call: {e}")
                    continue

        # Ensure at least one content block
        if not content_blocks:
            content_blocks.append(ContentBlockText(type=Constants.CONTENT_TEXT, text=""))

        # Map finish reason to Anthropic format
        if finish_reason == "length":
            stop_reason = Constants.STOP_MAX_TOKENS
        elif finish_reason == "tool_calls":
            stop_reason = Constants.STOP_TOOL_USE
        elif finish_reason is None and tool_calls:
            stop_reason = Constants.STOP_TOOL_USE
        else:
            stop_reason = Constants.STOP_END_TURN

        return MessagesResponse(
            id=response_id,
            model=original_request.original_model or original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=content_blocks,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )

    except Exception as e:
        logger.error(f"Error converting response: {e}")
        return MessagesResponse(
            id=f"msg_error_{uuid.uuid4()}",
            model=original_request.original_model or original_request.model,
            role=Constants.ROLE_ASSISTANT,
            content=[ContentBlockText(type=Constants.CONTENT_TEXT, text="Response conversion error")],
            stop_reason=Constants.STOP_ERROR,
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs

        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"

        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0

        # Process each chunk
        async for chunk in response_generator:
            try:


                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                    # Calculate cost based on token counts
                    model_to_use = original_request.model
                    if model_to_use.startswith("custom/"):
                        model_name = model_to_use[7:]  # Remove 'custom/' prefix
                        if model_name in CUSTOM_OPENAI_MODELS:
                            config = CUSTOM_OPENAI_MODELS[model_name]
                            input_cost = config.get("input_cost_per_token", 0.000001)
                            output_cost = config.get("output_cost_per_token", 0.000002)
                            total_cost = (input_tokens * input_cost) + (output_tokens * output_cost)
                            logger.info(f"Stream complete - Model: {model_to_use}, Input tokens: {input_tokens}, Output tokens: {output_tokens}, Cost: ${total_cost:.8f}")
                        else:
                            logger.warning(f"Could not find model configuration for {model_to_use}")

                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]

                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})

                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)

                    # Process text content
                    delta_content = None

                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']

                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content

                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"

                    # Process tool calls
                    delta_tool_calls = None

                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']

                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]

                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0

                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index

                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")

                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""

                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''

                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, use it
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        # Otherwise, try to parse it
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment, treat it as a string
                                    args_json = arguments

                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""

                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"

                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True

                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"

                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}

                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"

                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue

        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}

            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"

            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)

        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"

        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    try:
        # print the body here
        body = await raw_request.body()

        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")

        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]

        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")

        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)

        # Determine which API key to use based on the model
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = config.openai_api_key
            logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            litellm_request["api_key"] = config.gemini_api_key
            logger.debug(f"Using Gemini API key for model: {request.model}")
        elif request.model.startswith("custom/"):
            # Extract the model name after the 'custom/' prefix
            custom_model_name = request.model[7:]
            if custom_model_name in CUSTOM_OPENAI_MODELS:
                model_config = CUSTOM_OPENAI_MODELS[custom_model_name]
                api_key_name = model_config.get("api_key_name", "OPENAI_API_KEY")

                # Set the API key
                custom_api_key = config.custom_api_keys.get(api_key_name)
                if custom_api_key:
                    litellm_request["api_key"] = custom_api_key
                else:
                    litellm_request["api_key"] = config.openai_api_key  # Fallback to default

                # Set the base URL
                litellm_request["api_base"] = model_config.get("api_base")

                # Set the actual model name if different from ID
                if model_config.get("model_name") and model_config.get("model_name") != custom_model_name:
                    # Save the original model for response
                    original_model = litellm_request["model"]
                    # Use the provider-specific model name for the actual API call
                    litellm_request["model"] = f"openai/{model_config['model_name']}"
                    logger.debug(f"Using custom OpenAI-compatible model: {original_model} → {litellm_request['model']}")
                else:
                    # If no specific model name, use OpenAI prefix with the model ID
                    litellm_request["model"] = f"openai/{custom_model_name}"

                logger.debug(f"Using custom API key ({api_key_name}) for model: {request.model}")
                logger.debug(f"Using custom API base: {litellm_request['api_base']}")
        else:
            litellm_request["api_key"] = config.anthropic_api_key
            logger.debug(f"Using Anthropic API key for model: {request.model}")

        # For OpenAI models - modify request format to work with limitations
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")

            # For OpenAI models, we need to convert content blocks to simple strings
            # and handle other requirements
            for i, msg in enumerate(litellm_request["messages"]):
                # Special case - handle message content directly when it's a list of tool_result
                # This is a specific case we're seeing in the error
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break

                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"Found message with only tool_result content - special handling required")
                        # Extract the content from all tool_result blocks
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])

                            # Handle different formats of content
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # Fall back to string representation of any dict
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"

                        # Replace the list with extracted text
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                        continue  # Skip normal processing for this message

                # 1. Handle content field - normal case
                if "content" in msg:
                    # Check if content is a list (content blocks)
                    if isinstance(msg["content"], list):
                        # Convert complex content blocks to simple string
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # Handle different content block types
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"

                                # Handle tool_result content blocks - extract nested text
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"

                                    # Extract text from the tool_result content
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                # Handle any dict by trying to extract text or convert to JSON
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item) + "\n"
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # Handle dictionary content
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"

                                # Handle tool_use content blocks
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"

                                # Handle image content blocks
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"

                        # Make sure content is never empty for OpenAI models
                        if not text_content.strip():
                            text_content = "..."

                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # Also check for None or empty string content
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..." # Empty content not allowed

                # 2. Remove any fields OpenAI doesn't support in messages
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]

            # 3. Final validation - check for any remaining invalid values and dump full message details
            for i, msg in enumerate(litellm_request["messages"]):
                # Log the message format for debugging
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")

                # If content is still a list or None, replace with placeholder
                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                    # Last resort - stringify the entire content as JSON
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    litellm_request["messages"][i]["content"] = "..." # Fallback placeholder

        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")

        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            # Ensure we use the async version for streaming
            response_generator = await litellm.acompletion(**litellm_request)

            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")

            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)

            return anthropic_response

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()

        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }

        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)

        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)

        # Log all error details
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")

        # Format error for response
        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"

        # Return detailed error
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
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
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]

        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )

        # Use LiteLLM's token_counter function
        try:
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )

            # Count tokens
            token_count = litellm.token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )

            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)

        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/health")
async def health_check():
    """Enhanced health check with detailed proxy server information"""
    try:
        # Gather API key status for different providers
        api_status = {
            "anthropic": bool(config.anthropic_api_key),
            "openai": bool(config.openai_api_key),
            "gemini": bool(config.gemini_api_key),
            "custom_models": len(CUSTOM_OPENAI_MODELS)
        }

        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "proxy_type": "Universal Anthropic API Proxy",
            "supported_providers": {
                "anthropic": {
                    "configured": api_status["anthropic"],
                    "models": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"] if api_status["anthropic"] else []
                },
                "openai": {
                    "configured": api_status["openai"],
                    "models": OPENAI_MODELS[:3] if api_status["openai"] else []
                },
                "gemini": {
                    "configured": api_status["gemini"],
                    "models": GEMINI_MODELS if api_status["gemini"] else []
                },
                "custom": {
                    "configured": api_status["custom_models"] > 0,
                    "count": api_status["custom_models"],
                    "models": list(CUSTOM_OPENAI_MODELS.keys())[:3] if api_status["custom_models"] > 0 else []
                }
            },
            "model_mapping": {
                "big_model": config.big_model,
                "small_model": config.small_model,
                "preferred_provider": config.preferred_provider
            },
            "features": {
                "streaming": True,
                "tool_use": True,
                "image_support": True,
                "cost_tracking": True,
                "model_mapping": True
            }
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
                "message": str(e)
            }
        )

@app.get("/test-connection")
async def test_connection():
    """Test API connectivity to configured providers"""
    test_results = {}
    overall_status = "success"

    # Test Anthropic API if configured
    if config.anthropic_api_key:
        try:
            # Use a simple test - we'll just check if we can make a request structure
            test_results["anthropic"] = {
                "status": "configured",
                "message": "API key configured",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            test_results["anthropic"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            overall_status = "partial"
    else:
        test_results["anthropic"] = {
            "status": "not_configured",
            "message": "ANTHROPIC_API_KEY not set"
        }

    # Test Gemini API if configured
    if config.gemini_api_key:
        try:
            # Simple test request to verify API connectivity
            test_response = await litellm.acompletion(
                model="gemini/gemini-2.0-flash",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                api_key=config.gemini_api_key
            )

            test_results["gemini"] = {
                "status": "success",
                "message": "Successfully connected to Gemini API",
                "model_used": "gemini-2.0-flash",
                "timestamp": datetime.now().isoformat(),
                "response_id": getattr(test_response, 'id', 'unknown')
            }
        except Exception as e:
            test_results["gemini"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check your GEMINI_API_KEY is valid",
                    "Verify API key permissions",
                    "Check rate limits"
                ]
            }
            overall_status = "partial"
    else:
        test_results["gemini"] = {
            "status": "not_configured",
            "message": "GEMINI_API_KEY not set"
        }

    # Test OpenAI API if configured
    if config.openai_api_key:
        try:
            test_results["openai"] = {
                "status": "configured",
                "message": "API key configured",
                "available_models": OPENAI_MODELS[:3],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            test_results["openai"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            overall_status = "partial"
    else:
        test_results["openai"] = {
            "status": "not_configured",
            "message": "OPENAI_API_KEY not set"
        }

    # Test custom models
    if CUSTOM_OPENAI_MODELS:
        test_results["custom_models"] = {
            "status": "configured",
            "count": len(CUSTOM_OPENAI_MODELS),
            "models": list(CUSTOM_OPENAI_MODELS.keys()),
            "message": f"{len(CUSTOM_OPENAI_MODELS)} custom models configured"
        }
    else:
        test_results["custom_models"] = {
            "status": "not_configured",
            "message": "No custom models configured"
        }

    # Return appropriate status code
    if overall_status == "success":
        return {
            "status": overall_status,
            "message": "API connectivity test completed",
            "timestamp": datetime.now().isoformat(),
            "results": test_results
        }
    else:
        return JSONResponse(
            status_code=207,  # Multi-status
            content={
                "status": overall_status,
                "message": "Some API tests failed or not configured",
                "timestamp": datetime.now().isoformat(),
                "results": test_results
            }
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
            "providers": ["anthropic", "openai", "gemini", "custom"],
            "features": ["streaming", "tool_use", "image_support", "cost_tracking", "model_mapping"]
        },
        "configuration": {
            "preferred_provider": config.preferred_provider,
            "big_model": config.big_model,
            "small_model": config.small_model,
            "custom_models_count": len(CUSTOM_OPENAI_MODELS)
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection"
        },
        "documentation": {
            "anthropic_format": "Uses Anthropic Claude API format",
            "model_mapping": "Automatically maps claude-3-haiku → small_model, claude-3-sonnet/opus → big_model",
            "streaming": "Supports streaming responses with proper SSE format",
            "tools": "Full tool use support with proper formatting"
        }
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
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
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
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"


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
    uvicorn.run(app, host=config.host, port=config.port,
                log_level=config.log_level.lower())
