"""
Configuration management for the anthropic_proxy package.
This module handles all configuration loading, validation, and management.
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .types import ModelDefaults

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)





def parse_token_value(value, default_value=None):
    """Parse token value that can be in 'k' format (16k, 66k) or specific number.

    Args:
        value: The value to parse (can be string like "16k", "66k" or integer)
        default_value: Default value to return if parsing fails

    Returns:
        Integer token count

    Examples:
        parse_token_value("16k") -> 16384
        parse_token_value("66k") -> 67584
        parse_token_value("8k") -> 8192
        parse_token_value(8192) -> 8192
        parse_token_value("256k") -> 262144
    """
    if value is None:
        return default_value

    # If it's already an integer, return it
    if isinstance(value, int):
        return value

    # If it's a string, try to parse it
    if isinstance(value, str):
        value = value.strip().lower()

        # Handle 'k' suffix format
        if value.endswith("k"):
            try:
                num_str = value[:-1]  # Remove 'k'
                num = float(num_str)
                return int(num * 1024)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not parse token value '{value}', using default {default_value}"
                )
                return default_value

        # Handle plain number string
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(
                f"Could not parse token value '{value}', using default {default_value}"
            )
            return default_value

    logger.warning(
        f"Unexpected token value type '{type(value)}' for value '{value}', using default {default_value}"
    )

    return default_value


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
        self.long_context_threshold = parse_token_value(
            os.environ.get(
                "LONG_CONTEXT_THRESHOLD", str(ModelDefaults.LONG_CONTEXT_THRESHOLD)
            ),
            ModelDefaults.LONG_CONTEXT_THRESHOLD,
        )

        # Server configuration
        self.host = os.environ.get("HOST", ModelDefaults.DEFAULT_HOST)
        self.port = int(os.environ.get("PORT", str(ModelDefaults.DEFAULT_PORT)))
        self.log_level = os.environ.get("LOG_LEVEL", ModelDefaults.DEFAULT_LOG_LEVEL)
        self.log_file_path = os.environ.get(
            "LOG_FILE_PATH",
            Path(__file__).resolve().parent / "server.log",
        )

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

        # Set the project root path for .env file checking
        self.project_root = self._get_project_root()

    def _get_project_root(self) -> str:
        """Get the project root directory (parent of the package directory)"""
        package_dir = Path(__file__).resolve().parent
        return str(Path(package_dir).parent)

    def check_env_file_exists(self) -> bool:
        """Check if .env file exists in the project root"""
        env_file_path = Path(self.project_root) / ".env"
        return Path(env_file_path).exists()

    def get_env_file_path(self) -> str:
        """Get the full path to the .env file"""
        return str(Path(self.project_root) / ".env")

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


# Global configuration instance
config = Config()


# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
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
        if record.levelno == logging.DEBUG and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)






def setup_logging():
    """Setup logging configuration to be idempotent."""
    # This function is designed to be safe to call multiple times.
    # It ensures that logging handlers are only added once to the root logger,
    # preventing duplicate log entries in multi-process environments (like uvicorn workers).
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        return

    try:
        # Ensure log directory exists
        log_dir = Path(config.log_file_path).parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        # Configure the root logger
        root_logger.setLevel(getattr(logging, config.log_level.upper()))
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Add file handler
        file_handler = logging.FileHandler(config.log_file_path, mode="a")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Add stream handler (for console output)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            ColorizedFormatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(stream_handler)

        # Add custom message filter
        root_logger.addFilter(MessageFilter())

        # Configure uvicorn log levels. Handlers are inherited from the root logger.
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        uvicorn_access_logger.setLevel(logging.INFO)
        uvicorn_access_logger.propagate = True  # Ensure access logs reach root handlers
        if config.log_level.lower() == "debug":
            logging.getLogger("openai").setLevel(logging.INFO)
            logging.getLogger("httpx").setLevel(logging.INFO)

        logger.info("✅ Logging configured for server.")

    except Exception as e:
        print(f"🔴 Error setting up logging: {e}")
        sys.exit(1)
