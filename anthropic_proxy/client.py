"""
OpenAI client management and custom model handling.
This module manages OpenAI client creation and custom model configurations.
"""

import logging
from pathlib import Path

import httpx
import yaml
from openai import AsyncOpenAI

from .config import config, parse_token_value
from .types import ModelDefaults

logger = logging.getLogger(__name__)

# Dictionary to store custom OpenAI-compatible model configurations
CUSTOM_OPENAI_MODELS = {}


def load_custom_models(config_file=None):
    """Load custom OpenAI-compatible model configurations from YAML file.

    Note: API keys and pricing are no longer handled here.
    API keys come from request headers (via ccproxy).
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
            if "model_id" not in model or "api_base" not in model:
                logger.warning(
                    f"Invalid model configuration, missing required fields: {model}"
                )
                continue

            model_id = model["model_id"]
            model_name = model.get("model_name", model_id)

            # Determine if this model should use direct Claude API mode
            is_direct_mode = model.get("direct", False) or "anthropic.com" in model["api_base"].lower()

            CUSTOM_OPENAI_MODELS[model_id] = {
                "model_id": model_id,
                "model_name": model_name,
                "api_base": model["api_base"],
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
            }

            logger.info(
                f"Loaded model: {model_id} -> {model_name} ({model['api_base']})"
            )

    except Exception as e:
        logger.error(f"Error loading custom models: {str(e)}")


def initialize_custom_models():
    """Initialize custom models. Called when running as main.

    Note: API keys are now passed via request headers from ccproxy,
    so we no longer load them from environment variables.
    """
    load_custom_models()


def create_openai_client(model_id: str, api_key: str) -> AsyncOpenAI:
    """Create OpenAI client for the given model.

    Args:
        model_id: The model identifier from models.yaml
        api_key: The API key passed from request headers (via ccproxy)

    Returns:
        AsyncOpenAI client configured for the model's API base URL
    """
    if model_id not in CUSTOM_OPENAI_MODELS:
        raise ValueError(f"Unknown custom model: {model_id}")

    model_config = CUSTOM_OPENAI_MODELS[model_id]
    base_url = model_config["api_base"]

    if not api_key:
        raise ValueError(f"No API key provided for model: {model_id}")

    # Create client without retry transport (it causes 404 errors with some APIs)
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = AsyncOpenAI(**client_kwargs, timeout=httpx.Timeout(60.0))
    logger.debug(f"Create OpenAI Client: model={model_id}, base_url={base_url}, retries={ModelDefaults.DEFAULT_MAX_RETRIES}")
    return client


def create_claude_client(model_id: str, api_key: str) -> httpx.AsyncClient:
    """Create direct Claude API client for the given model.

    Args:
        model_id: The model identifier from models.yaml
        api_key: The API key passed from request headers (via ccproxy)

    Returns:
        httpx.AsyncClient configured for direct Claude API calls
    """
    if model_id not in CUSTOM_OPENAI_MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    model_config = CUSTOM_OPENAI_MODELS[model_id]
    base_url = model_config["api_base"]

    if not api_key:
        raise ValueError(f"No API key provided for model: {model_id}")

    # Ensure base_url ends with /v1 for Claude API
    if not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
        base_url = base_url + "v1" if base_url.endswith("/") else base_url + "/v1"

    headers = {
        "x-api-key": api_key,
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
    if model_id in CUSTOM_OPENAI_MODELS:
        return CUSTOM_OPENAI_MODELS[model_id]
    else:
        raise ValueError(f"Model {model_id} not found in custom models")


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
