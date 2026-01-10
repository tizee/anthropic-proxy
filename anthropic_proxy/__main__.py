#!/usr/bin/env python3
"""
Main entry point for the anthropic_proxy package.
This allows the package to be run as: uv run anthropic-proxy
"""

import argparse
import sys
import logging

import uvicorn

from .config import config, setup_logging

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(description="Run the Claude Code Proxy Server.")
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload on code changes."
    )
    parser.add_argument("--host", default=config.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.port, help="Port to bind to")
    args = parser.parse_args()

    # Check for .env file existence
    if not config.check_env_file_exists():
        logger.error("No .env file found in the project root directory!")
        logger.error(f"Expected location: {config.get_env_file_path()}")
        logger.error("Please create a .env file with your API keys and configuration.")
        logger.error("You can reference .env.example for the required format.")
        sys.exit(1)

    # Setup logging for the main process
    setup_logging()

    # Print initial configuration status
    logger.info("✅ Configuration loaded")

    # Run the Server
    uvicorn.run(
        "anthropic_proxy.server:app",
        host=args.host,
        port=args.port,
        log_config=None,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
