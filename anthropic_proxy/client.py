"""
OpenAI client management and custom model handling.
This module manages OpenAI client creation and custom model configurations.
"""

import logging
from pathlib import Path

import httpx
import yaml
from openai import AsyncOpenAI

from .antigravity import (
    ANTIGRAVITY_ENDPOINT,
    DEFAULT_ANTIGRAVITY_MODELS,
    antigravity_auth,
)
from .codex import CODEX_API_URL, DEFAULT_CODEX_MODELS, codex_auth
from .config import config, parse_token_value
from .gemini import DEFAULT_GEMINI_MODELS, GEMINI_CODE_ASSIST_ENDPOINT, gemini_auth
from .types import ModelDefaults

logger = logging.getLogger(__name__)

# Dictionary to store custom OpenAI-compatible model configurations
CUSTOM_OPENAI_MODELS = {}


def load_models_config(config_file=None):
    """Load custom OpenAI-compatible model configurations from YAML file.

    API keys can be configured per-model in this file, or passed via request
    headers from ccproxy. Model-specific keys take precedence over header keys.
    Pricing should be tracked via provider billing dashboards.
    """
    global CUSTOM_OPENAI_MODELS

    if config_file is None:
        config_file = config.custom_models_file

    if not Path(config_file).exists():
        logger.warning(f"Custom models config file not found: {config_file}")
        return

    try:
        with Path(config_file).open() as file:
            models = yaml.safe_load(file)

        if not models:
            logger.warning(f"No models found in config file: {config_file}")
            return

        for model in models:
            model_id = model.get("model_id")

            # Apply defaults for Codex provider models
            if model.get("provider") == "codex":
                if "api_base" not in model:
                    model["api_base"] = CODEX_API_URL
                if "api_key" not in model:
                    model["api_key"] = "codex-auth" # Placeholder to pass validation

                # Apply default reasoning effort if known model and not specified
                if "reasoning_effort" not in model and model_id in DEFAULT_CODEX_MODELS:
                    model["reasoning_effort"] = DEFAULT_CODEX_MODELS[model_id].get("reasoning_effort")

            # Apply defaults for Gemini provider models
            if model.get("provider") == "gemini":
                if "api_base" not in model:
                    model["api_base"] = GEMINI_CODE_ASSIST_ENDPOINT
                if "api_key" not in model:
                    model["api_key"] = "gemini-auth" # Placeholder

            # Apply defaults for Antigravity provider models
            if model.get("provider") == "antigravity":
                if "api_base" not in model:
                    model["api_base"] = ANTIGRAVITY_ENDPOINT
                if "api_key" not in model:
                    model["api_key"] = "antigravity-auth" # Placeholder

            if "model_id" not in model or "api_base" not in model:
                logger.warning(
                    f"Invalid model configuration, missing required fields: {model}"
                )
                continue

            model_id = model["model_id"]
            model_name = model.get("model_name", model_id)

            # Determine if this model should use direct Claude API mode
            is_direct_mode = model.get("direct", False) or "anthropic.com" in model["api_base"].lower()

            # Determine provider (default to openai if direct=false, anthropic if direct=true)
            provider = model.get("provider")
            if not provider:
                provider = "anthropic" if is_direct_mode else "openai"

            CUSTOM_OPENAI_MODELS[model_id] = {
                "model_id": model_id,
                "model_name": model_name,
                "api_base": model["api_base"],
                "api_key": model.get("api_key"),  # Optional model-specific API key
                "can_stream": model.get("can_stream", True),
                "max_tokens": parse_token_value(
                    model.get("max_tokens"), ModelDefaults.DEFAULT_MAX_TOKENS
                ),
                "context": parse_token_value(
                    model.get("context"), ModelDefaults.LONG_CONTEXT_THRESHOLD
                ),
                "max_input_tokens": parse_token_value(
                    model.get(
                        "max_input_tokens", ModelDefaults.DEFAULT_MAX_INPUT_TOKENS
                    ),
                    ModelDefaults.DEFAULT_MAX_INPUT_TOKENS,
                ),
                # openai request extra options
                "extra_headers": model.get("extra_headers", {}),
                "extra_body": model.get("extra_body", {}),
                "temperature": model.get("temperature", 1.0),
                "reasoning_effort": model.get("reasoning_effort", None),
                # Direct mode configuration
                "direct": is_direct_mode,
                "provider": provider,
            }

            logger.info(
                f"Loaded model: {model_id} -> {model_name} ({model['api_base']})"
            )

    except Exception as e:
        logger.error(f"Error loading custom models: {str(e)}")


def load_codex_models():
    """Auto-register default Codex models if authentication is available."""
    # Check if we have auth data (access or refresh token)
    # We don't need to validate it here, just check existence to avoid cluttering
    # the model list for users who don't use Codex.
    has_auth = codex_auth.has_auth()

    if not has_auth:
        return

    logger.info("Codex authentication detected. Registering default Codex models...")

    for model_id, details in DEFAULT_CODEX_MODELS.items():
        # User defined config in models.yaml takes precedence
        if model_id in CUSTOM_OPENAI_MODELS:
            logger.debug(f"Skipping default Codex model {model_id} (overridden by user config)")
            continue

        CUSTOM_OPENAI_MODELS[model_id] = {
            "model_id": model_id,
            "model_name": details["model_name"],
            "api_base": CODEX_API_URL,
            "api_key": "dummy",
            "can_stream": True,
            "max_tokens": ModelDefaults.DEFAULT_MAX_TOKENS,
            "context": ModelDefaults.LONG_CONTEXT_THRESHOLD,
            "max_input_tokens": ModelDefaults.DEFAULT_MAX_INPUT_TOKENS,
            "extra_headers": {},
            "extra_body": {},
            "temperature": 1.0,
            "reasoning_effort": details.get("reasoning_effort"),
            "direct": False,
            "provider": "codex",
        }
        logger.debug(f"Registered default Codex model: {model_id}")

def load_gemini_models():
    """Auto-register default Gemini models if authentication is available."""
    has_auth = gemini_auth.has_auth()

    if not has_auth:
        return

    logger.info("Gemini authentication detected. Registering default Gemini models...")

    for model_id, details in DEFAULT_GEMINI_MODELS.items():
        if model_id in CUSTOM_OPENAI_MODELS:
            logger.debug(f"Skipping default Gemini model {model_id} (overridden by user config)")
            continue

        CUSTOM_OPENAI_MODELS[model_id] = {
            "model_id": model_id,
            "model_name": details["model_name"],
            "api_base": GEMINI_CODE_ASSIST_ENDPOINT,
            "api_key": "dummy",
            "can_stream": True,
            "max_tokens": ModelDefaults.DEFAULT_MAX_TOKENS,
            "context": ModelDefaults.LONG_CONTEXT_THRESHOLD,
            "max_input_tokens": ModelDefaults.DEFAULT_MAX_INPUT_TOKENS,
            "extra_headers": {},
            "extra_body": {},
            "temperature": 1.0,
            "direct": False,
            "provider": "gemini",
        }
        logger.debug(f"Registered default Gemini model: {model_id}")


def load_antigravity_models():
    """Auto-register default Antigravity models if authentication is available."""
    has_auth = antigravity_auth.has_auth()

    if not has_auth:
        return

    logger.info("Antigravity authentication detected. Registering default Antigravity models...")

    for model_id, details in DEFAULT_ANTIGRAVITY_MODELS.items():
        if model_id in CUSTOM_OPENAI_MODELS:
            logger.debug(f"Skipping default Antigravity model {model_id} (overridden by user config)")
            continue

        CUSTOM_OPENAI_MODELS[model_id] = {
            "model_id": model_id,
            "model_name": details["model_name"],
            "api_base": ANTIGRAVITY_ENDPOINT,
            "api_key": "dummy",
            "can_stream": True,
            "max_tokens": ModelDefaults.DEFAULT_MAX_TOKENS,
            "context": ModelDefaults.LONG_CONTEXT_THRESHOLD,
            "max_input_tokens": ModelDefaults.DEFAULT_MAX_INPUT_TOKENS,
            "extra_headers": {},
            "extra_body": {},
            "temperature": 1.0,
            "direct": False,
            "provider": "antigravity",
        }
        logger.debug(f"Registered default Antigravity model: {model_id}")


def initialize_custom_models():
    """Initialize custom models. Called when running as main.

    API keys can be configured per-model in models.yaml, or passed via
    request headers from ccproxy. Model-specific keys take precedence.
    """
    load_models_config()
    load_codex_models()
    load_gemini_models()
    load_antigravity_models()


def create_openai_client(model_id: str, api_key: str | None) -> AsyncOpenAI:
    """Create OpenAI client for the given model.

    Args:
        model_id: The model identifier from models.yaml
        api_key: The API key passed from request headers (via ccproxy), or None

    Returns:
        AsyncOpenAI client configured for the model's API base URL

    Note:
        Model-specific API keys (from models.yaml) take precedence over header keys.
    """
    if model_id not in CUSTOM_OPENAI_MODELS:
        raise ValueError(f"Unknown custom model: {model_id}")

    model_config = CUSTOM_OPENAI_MODELS[model_id]
    base_url = model_config["api_base"]

    # Use model-specific API key if available, otherwise use the provided one
    client_api_key = model_config.get("api_key") or api_key

    if not client_api_key:
        raise ValueError(f"No API key provided for model: {model_id}")

    # Create client without retry transport (it causes 404 errors with some APIs)
    client_kwargs = {"api_key": client_api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = AsyncOpenAI(**client_kwargs, timeout=httpx.Timeout(60.0))
    logger.debug(f"Create OpenAI Client: model={model_id}, base_url={base_url}, retries={ModelDefaults.DEFAULT_MAX_RETRIES}")
    return client


def create_claude_client(model_id: str, api_key: str | None) -> httpx.AsyncClient:
    """Create direct Claude API client for the given model.

    Args:
        model_id: The model identifier from models.yaml
        api_key: The API key passed from request headers (via ccproxy), or None

    Returns:
        httpx.AsyncClient configured for direct Claude API calls

    Note:
        Model-specific API keys (from models.yaml) take precedence over header keys.
    """
    if model_id not in CUSTOM_OPENAI_MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    model_config = CUSTOM_OPENAI_MODELS[model_id]
    base_url = model_config["api_base"]

    # Use model-specific API key if available, otherwise use the provided one
    client_api_key = model_config.get("api_key") or api_key

    if not client_api_key:
        raise ValueError(f"No API key provided for model: {model_id}")

    # Ensure base_url ends with /v1 for Claude API
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"

    headers = {
        "x-api-key": client_api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    # Add any extra headers from model config
    if model_config.get("extra_headers"):
        headers.update(model_config["extra_headers"])

    # Configure retry transport for better reliability
    transport = httpx.AsyncHTTPTransport(retries=ModelDefaults.DEFAULT_MAX_RETRIES)

    client = httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=httpx.Timeout(60.0),
        transport=transport
    )

    logger.debug(f"Create Claude Client: model={model_id}, base_url={base_url}, retries={ModelDefaults.DEFAULT_MAX_RETRIES}")
    return client


def get_model_config(model_id: str) -> dict:
    """Get model configuration for a given model ID."""
    if model_id not in CUSTOM_OPENAI_MODELS:
        raise ValueError(f"Model {model_id} not found in custom models")
    return CUSTOM_OPENAI_MODELS[model_id]


def list_available_models() -> list:
    """List all available custom models."""
    return list(CUSTOM_OPENAI_MODELS.keys())


def validate_model_exists(model_id: str) -> bool:
    """Check if a model exists in the custom models configuration."""
    return model_id in CUSTOM_OPENAI_MODELS


def is_direct_mode_model(model_id: str) -> bool:
    """Check if a model should use direct Claude API mode."""
    if model_id not in CUSTOM_OPENAI_MODELS:
        return False
    return CUSTOM_OPENAI_MODELS[model_id].get("direct", False)


def is_codex_model(model_id: str) -> bool:
    """Check if a model is a Codex model."""
    if model_id not in CUSTOM_OPENAI_MODELS:
        return False
    return CUSTOM_OPENAI_MODELS[model_id].get("provider") == "codex"


def is_gemini_model(model_id: str) -> bool:
    """Check if a model is a Gemini model."""
    if model_id not in CUSTOM_OPENAI_MODELS:
        return False
    return CUSTOM_OPENAI_MODELS[model_id].get("provider") == "gemini"


def is_antigravity_model(model_id: str) -> bool:
    """Check if a model is an Antigravity model."""
    if model_id not in CUSTOM_OPENAI_MODELS:
        return False
    return CUSTOM_OPENAI_MODELS[model_id].get("provider") == "antigravity"
