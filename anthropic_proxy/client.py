"""
OpenAI client management and custom model handling.
This module manages OpenAI client creation and custom model configurations.
"""

import logging
import os
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
                "max_tokens": parse_token_value(
                    model.get("max_tokens"), ModelDefaults.DEFAULT_MAX_TOKENS
                ),
                "context": parse_token_value(
                    model.get("context"), ModelDefaults.LONG_CONTEXT_THRESHOLD
                ),
                "input_cost_per_token": input_cost,
                "output_cost_per_token": output_cost,
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
