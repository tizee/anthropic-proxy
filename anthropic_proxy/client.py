"""
OpenAI client management and custom model handling.
This module manages OpenAI client creation and custom model configurations.
"""

import logging
import os
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
    """Load custom OpenAI-compatible model configurations from YAML file."""
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

            # Set default pricing if not provided - support both old and new formats
            # Priority: new format > old format > defaults
            input_cost_per_million = model.get("input_cost_per_million_tokens")
            output_cost_per_million = model.get("output_cost_per_million_tokens")

            # Backward compatibility: convert old per-token pricing to per-million
            if input_cost_per_million is None:
                input_cost_per_token = model.get("input_cost_per_token")
                if input_cost_per_token is not None:
                    input_cost_per_million = input_cost_per_token * 1_000_000
                    logger.warning(f"Model {model_id}: Using deprecated 'input_cost_per_token' field. Please use 'input_cost_per_million_tokens' instead.")
                else:
                    input_cost_per_million = ModelDefaults.DEFAULT_INPUT_COST_PER_MILLION_TOKENS

            if output_cost_per_million is None:
                output_cost_per_token = model.get("output_cost_per_token")
                if output_cost_per_token is not None:
                    output_cost_per_million = output_cost_per_token * 1_000_000
                    logger.warning(f"Model {model_id}: Using deprecated 'output_cost_per_token' field. Please use 'output_cost_per_million_tokens' instead.")
                else:
                    output_cost_per_million = ModelDefaults.DEFAULT_OUTPUT_COST_PER_MILLION_TOKENS

            # Determine if this model should use direct Claude API mode
            is_direct_mode = model.get("direct", False) or "anthropic.com" in model["api_base"].lower()

            CUSTOM_OPENAI_MODELS[model_id] = {
                "model_id": model_id,
                "model_name": model_name,
                "api_base": model["api_base"],
                "api_key_name": model.get("api_key_name", "OPENAI_API_KEY"),
                "can_stream": model.get("can_stream", True),
                "max_tokens": parse_token_value(
                    model.get("max_tokens"), ModelDefaults.DEFAULT_MAX_TOKENS
                ),
                "context": parse_token_value(
                    model.get("context"), ModelDefaults.LONG_CONTEXT_THRESHOLD
                ),
                "input_cost_per_million_tokens": input_cost_per_million,
                "output_cost_per_million_tokens": output_cost_per_million,
                # Backward compatibility fields
                "input_cost_per_token": input_cost_per_million / 1_000_000,  # Deprecated
                "output_cost_per_token": output_cost_per_million / 1_000_000,  # Deprecated
                "max_input_tokens": parse_token_value(
                    model.get(
                        "max_input_tokens", ModelDefaults.DEFAULT_MAX_INPUT_TOKENS
                    ),
                    ModelDefaults.DEFAULT_MAX_INPUT_TOKENS,
                ),
                # openai request extra options
                "extra_headers": model.get("extra_headers", {}),
                "extra_body": model.get("extra_body", {}),
                "reasoning_effort": model.get("reasoning_effort", None),
                # Direct mode configuration
                "direct": is_direct_mode,
            }

            # Store pricing info for cost calculation
            model_variations = [
                f"{model_name}",
            ]

            for variation in model_variations:
                model_pricing[variation] = {
                    "input_cost_per_million_tokens": input_cost_per_million,
                    "output_cost_per_million_tokens": output_cost_per_million,
                    # Backward compatibility
                    "input_cost_per_token": input_cost_per_million / 1_000_000,  # Deprecated
                    "output_cost_per_token": output_cost_per_million / 1_000_000,  # Deprecated
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


def create_openai_client(model_id: str) -> AsyncOpenAI:
    """Create OpenAI client for the given model and return client and request parameters."""
    api_key = None
    base_url = None
    # Custom OpenAI-compatible models
    if model_id in CUSTOM_OPENAI_MODELS:
        model_config = CUSTOM_OPENAI_MODELS[model_id]
        api_key_name = model_config.get("api_key_name", "OPENAI_API_KEY")
        api_key = config.custom_api_keys.get(api_key_name)
        base_url = model_config["api_base"]
    else:
        raise ValueError(f"Unknown custom model: {model_id}")

    if not api_key:
        raise ValueError(f"No API key available for model: {model_id}")

    # Create client with retry-enabled HTTP client
    # Configure retry transport for better reliability
    transport = httpx.AsyncHTTPTransport(retries=ModelDefaults.DEFAULT_MAX_RETRIES)
    http_client = httpx.AsyncClient(transport=transport)
    
    client_kwargs = {"api_key": api_key, "http_client": http_client}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = AsyncOpenAI(**client_kwargs)
    logger.debug(f"Create OpenAI Client: model={model_id}, base_url={base_url}, retries={ModelDefaults.DEFAULT_MAX_RETRIES}")
    return client


def create_claude_client(model_id: str) -> httpx.AsyncClient:
    """Create direct Claude API client for the given model."""
    if model_id not in CUSTOM_OPENAI_MODELS:
        raise ValueError(f"Unknown model: {model_id}")

    model_config = CUSTOM_OPENAI_MODELS[model_id]
    api_key_name = model_config.get("api_key_name", "ANTHROPIC_API_KEY")
    api_key = config.custom_api_keys.get(api_key_name)
    base_url = model_config["api_base"]

    if not api_key:
        raise ValueError(f"No API key available for model: {model_id}")

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


def determine_model_by_router(
    original_model: str, token_count: int, has_thinking: bool
) -> str:
    """Determine which model to use based on routing logic"""

    logger.debug(
        f"🔀 Router input: model={original_model}, tokens={token_count}, thinking={has_thinking}"
    )

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

    # Check if token count exceeds long context threshold
    if token_count > ModelDefaults.LONG_CONTEXT_THRESHOLD:
        if "long_context" in config.router_config:
            result = config.router_config["long_context"]
            logger.info(f"🔀 Router: Using long context model for large input ({token_count} tokens), result: {result}")
            return result
        logger.warning(f"🔀 Router: Long context needed ({token_count} tokens) but no long_context router configured")

    # Default model
    result = config.router_config["default"]
    logger.debug(f"🔀 Router: Using default model for {original_model}")
    logger.info(f"🔀 Router final result: {result}")

    # should be model id
    return result


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
