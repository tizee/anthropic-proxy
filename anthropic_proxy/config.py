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
        value: The value to parse (can be string like "16K", "66k" or integer)
        default_value: Default value to return if parsing fails

    Returns:
        Integer token count

    Examples:
        parse_token_value("16K") -> 16000
        parse_token_value("66k") -> 66000
        parse_token_value("8K") -> 8000
        parse_token_value(8000) -> 8000
        parse_token_value("256K") -> 256000
    """
    if value is None:
        return default_value

    # If it's already an integer, return it
    if isinstance(value, int):
        return value

    # If it's a string, try to parse it
    if isinstance(value, str):
        value = value.strip().lower()

        # Handle 'k' suffix format (both lowercase and uppercase K are supported)
        if value.endswith("k"):
            try:
                num_str = value[:-1]  # Remove 'k'
                num = float(num_str)
                return int(num * 1000)  # Use SI standard: 1K = 1000
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
    """Universal proxy server configuration"""

    def __init__(self):
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

        # Check raw message
        if hasattr(record, "msg") and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False

        # Also check the formatted message (for messages with placeholders)
        try:
            formatted_msg = record.getMessage()
            for phrase in blocked_phrases:
                if phrase in formatted_msg:
                    return False
        except:
            pass

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
    # Configure root logger and uvicorn logs
    root_logger = logging.getLogger()
    log_level = getattr(logging, config.log_level.upper())

    # Set root log level
    root_logger.setLevel(log_level)

    # Configure uvicorn log levels
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.error").setLevel(log_level)

    # For access logs, only enable them at DEBUG or INFO levels
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    if log_level <= logging.INFO:
        uvicorn_access_logger.setLevel(logging.INFO)
        uvicorn_access_logger.propagate = True  # Ensure access logs reach root handlers
    else:
        uvicorn_access_logger.setLevel(logging.WARNING)
        uvicorn_access_logger.propagate = False  # Disable access logs for higher log levels

    # Configure openai and httpx log levels to reduce noise
    if config.log_level.lower() == "debug":
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)
    else:
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    # This function is designed to be safe to call multiple times.
    # Only add handlers if they don't exist yet
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

        # Add rotating file handler with 2MB max size and 1 backup log
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            config.log_file_path,
            mode="a",
            maxBytes=2 * 1024 * 1024,  # 2MB
            backupCount=1,
            encoding="utf-8"
        )
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

        # Additional third-party library log configuration
        if config.log_level.lower() != "debug":
            # Disable verbose debug logging from libraries when not in debug mode
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)

        logger.info("✅ Logging configured for server.")

    except Exception as e:
        logger.critical(f"Error setting up logging: {e}")
        sys.exit(1)
