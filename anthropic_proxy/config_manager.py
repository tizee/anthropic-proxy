"""
Configuration directory and file management.
Handles initialization, creation, and management of config files.
"""

import json
import logging
from pathlib import Path

from .types import ModelDefaults

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "anthropic-proxy"
DEFAULT_LOG_DIR = Path.home() / ".anthropic-proxy"
DEFAULT_MODELS_FILE = DEFAULT_CONFIG_DIR / "models.yaml"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_AUTH_FILE = DEFAULT_CONFIG_DIR / "auth.json"

# Default models.yaml template
DEFAULT_MODELS_TEMPLATE = """# Anthropic Proxy Model Configuration
# Required fields: model_id, api_base
# Optional fields: model_name, api_key, can_stream, max_tokens, max_input_tokens,
#                  context, extra_headers, extra_body, format, direct, reasoning_effort, temperature
#
# API Key Configuration:
# - api_key: Optional per-model API key. If set, it takes precedence over request headers.
# - If api_key is not set, the key from the Authorization header (via ccproxy) is used.
# - API keys stored here are in plain text - use caution in shared environments.
#
# Token notation:
# - 1K = 1000 tokens (SI standard)
# - Use uppercase K for consistency
#
# reasoning_effort supports: minimal, low, medium, high (minimal = no thinking)
# format: openai | anthropic | gemini (routing format; defaults to openai)
# direct: legacy alias (direct: true -> format=anthropic). If format is set, it wins.
# thinking.type can be "enabled" or "disabled"
#
# Auth-provider default model IDs are prefixed to avoid collisions:
# - codex/<model_id>, gemini/<model_id>, antigravity/<model_id>
#
# Example configurations below - replace with your actual models

# Example: OpenAI-compatible model
- model_id: example-model
  model_name: gpt-4o-mini
  api_base: https://api.openai.com/v1
  format: openai
  can_stream: true
  max_tokens: 16K
  context: 128K
  reasoning_effort: minimal

# Example: Anthropic-compatible (direct) model
- model_id: example-claude-model
  model_name: claude-3-5-sonnet-20241022
  api_base: https://api.anthropic.com
  format: anthropic
  can_stream: true
  max_tokens: 8K
  max_input_tokens: 200K

# Note on Codex Models:
# If you have logged in via `anthropic-proxy login`, Codex subscription models
# (e.g., codex/gpt-5.2-codex) are automatically available.
# You can override their settings (e.g., reasoning_effort) here by specifying
# provider: codex (no need for api_base/api_key):
#
# - model_id: codex/gpt-5.2-codex
#   provider: codex
#   reasoning_effort: high
#   max_tokens: 32K
#
# To configure a Codex model using a standard API Key (non-subscription),
# set api_base/api_key and keep provider unset (you can still use a prefixed ID):
#
# - model_id: codex/gpt-5.2-codex
#   model_name: gpt-5.2-codex
#   api_base: https://api.openai.com/v1
#   api_key: sk-...
"""

# Default config.json template
DEFAULT_CONFIG_TEMPLATE = {
    "log_level": "WARNING",
    "log_file_path": str(DEFAULT_LOG_DIR / "server.log"),
    "host": ModelDefaults.DEFAULT_HOST,
    "port": ModelDefaults.DEFAULT_PORT,
    "cleanup_logs_on_start": True,
}


def ensure_config_dir() -> Path:
    """Ensure config directory exists, create if missing.

    Returns:
        Path to config directory
    """
    config_dir = DEFAULT_CONFIG_DIR
    if not config_dir.exists():
        logger.info(f"Creating config directory: {config_dir}")
        config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def ensure_log_dir() -> Path:
    """Ensure log directory exists, create if missing.

    Returns:
        Path to log directory
    """
    log_dir = DEFAULT_LOG_DIR
    if not log_dir.exists():
        logger.info(f"Creating log directory: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def create_default_models_file(force: bool = False) -> Path:
    """Create default models.yaml file if it doesn't exist.

    Args:
        force: If True, overwrite existing file

    Returns:
        Path to models.yaml file
    """
    ensure_config_dir()
    models_file = DEFAULT_MODELS_FILE

    if models_file.exists() and not force:
        logger.debug(f"Models file already exists: {models_file}")
        return models_file

    logger.info(f"Creating default models file: {models_file}")
    models_file.write_text(DEFAULT_MODELS_TEMPLATE, encoding="utf-8")
    return models_file


def create_default_config_file(force: bool = False) -> Path:
    """Create default config.json file if it doesn't exist.

    Args:
        force: If True, overwrite existing file

    Returns:
        Path to config.json file
    """
    ensure_config_dir()
    config_file = DEFAULT_CONFIG_FILE

    if config_file.exists() and not force:
        logger.debug(f"Config file already exists: {config_file}")
        return config_file

    logger.info(f"Creating default config file: {config_file}")
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG_TEMPLATE, f, indent=2)
    return config_file


def initialize_config(force: bool = False) -> tuple[Path, Path]:
    """Initialize all config files and directories.

    Args:
        force: If True, overwrite existing files

    Returns:
        Tuple of (models_file_path, config_file_path)
    """
    ensure_log_dir()
    ensure_config_dir()
    models_file = create_default_models_file(force=force)
    config_file = create_default_config_file(force=force)
    return models_file, config_file


def load_config_file(config_path: Path | None = None) -> dict:
    """Load config.json file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Config dictionary. Empty dict if file doesn't exist.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_FILE

    if not config_path.exists():
        logger.debug(f"Config file not found: {config_path}")
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        logger.debug(f"Loaded config from: {config_path}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return {}


def load_auth_file(auth_path: Path | None = None) -> dict:
    """Load auth.json file.

    Args:
        auth_path: Path to auth file. If None, uses default location.

    Returns:
        Auth dictionary. Empty dict if file doesn't exist.
    """
    if auth_path is None:
        auth_path = DEFAULT_AUTH_FILE

    if not auth_path.exists():
        logger.debug(f"Auth file not found: {auth_path}")
        return {}

    try:
        with auth_path.open("r", encoding="utf-8") as f:
            auth_data = json.load(f)
        logger.debug(f"Loaded auth data from: {auth_path}")
        return auth_data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in auth file {auth_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading auth file {auth_path}: {e}")
        return {}


def save_auth_file(auth_data: dict, auth_path: Path | None = None) -> bool:
    """Save auth data to auth.json file.

    Args:
        auth_data: Dictionary containing auth data
        auth_path: Path to auth file. If None, uses default location.

    Returns:
        True if successful, False otherwise
    """
    if auth_path is None:
        auth_path = DEFAULT_AUTH_FILE

    try:
        ensure_config_dir()

        # Write to temporary file first
        temp_path = auth_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(auth_data, f, indent=2)

        # Atomic rename
        temp_path.replace(auth_path)

        # Set permissions to 0600 (read/write only for owner)
        auth_path.chmod(0o600)

        logger.debug(f"Saved auth data to: {auth_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving auth file {auth_path}: {e}")
        return False


def get_models_file_path() -> Path:
    """Get the path to the models.yaml file.

    Returns:
        Path to models.yaml file
    """
    return DEFAULT_MODELS_FILE


def get_config_file_path() -> Path:
    """Get the path to the config.json file.

    Returns:
        Path to config.json file
    """
    return DEFAULT_CONFIG_FILE


def get_log_dir_path() -> Path:
    """Get the path to the log directory.

    Returns:
        Path to log directory
    """
    return DEFAULT_LOG_DIR


def get_default_log_file_path() -> Path:
    """Get the default log file path.

    Returns:
        Path to default log file
    """
    return DEFAULT_LOG_DIR / "server.log"
