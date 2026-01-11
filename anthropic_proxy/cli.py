"""
Command-line interface for anthropic-proxy.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from .config_manager import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_FILE,
    DEFAULT_LOG_DIR,
    DEFAULT_MODELS_FILE,
    initialize_config,
    load_config_file,
)
from .config import config
from . import daemon

logger = logging.getLogger(__name__)


def redact_api_keys(models_data: list, show_keys: bool = False) -> list:
    """Redact API keys from models data.

    Args:
        models_data: List of model configurations
        show_keys: If True, don't redact API keys

    Returns:
        Models data with API keys redacted (unless show_keys=True)
    """
    if show_keys:
        return models_data

    redacted = []
    for model in models_data:
        model_copy = dict(model)
        if "api_key" in model_copy:
            model_copy["api_key"] = "<REDACTED>"
        redacted.append(model_copy)
    return redacted


def print_config(models_path: Path, config_path: Path, show_api_keys: bool = False):
    """Print current configuration.

    Args:
        models_path: Path to models.yaml
        config_path: Path to config.json
        show_api_keys: If True, show actual API key values
    """
    print("\n" + "=" * 60)
    print("Anthropic Proxy Configuration")
    print("=" * 60)

    # Print config file locations
    print(f"\nConfig Directory: {DEFAULT_CONFIG_DIR}")
    print(f"Log Directory: {DEFAULT_LOG_DIR}")
    print(f"Models File: {models_path}")
    print(f"Config File: {config_path}")

    # Load and print config.json
    print("\n" + "-" * 60)
    print("config.json:")
    print("-" * 60)
    config_data = load_config_file(config_path)
    if config_data:
        print(json.dumps(config_data, indent=2))
    else:
        print("(not found or empty)")

    # Load and print models.yaml
    print("\n" + "-" * 60)
    print(f"models.yaml ({'with' if show_api_keys else 'without'} API keys):")
    print("-" * 60)
    if models_path.exists():
        try:
            with models_path.open("r", encoding="utf-8") as f:
                models_data = yaml.safe_load(f)

            if models_data:
                redacted_data = redact_api_keys(models_data, show_api_keys)
                print(yaml.dump(redacted_data, default_flow_style=False, sort_keys=False))
                model_count = len(models_data)
                print(f"\nTotal models: {model_count}")
            else:
                print("(no models configured)")
        except yaml.YAMLError as e:
            print(f"(error loading YAML: {e})")
    else:
        print("(file not found)")

    print("=" * 60 + "\n")


def init_config(force: bool = False):
    """Initialize config directory with default files.

    Args:
        force: If True, overwrite existing files
    """
    print("\n" + "=" * 60)
    print("Initializing Anthropic Proxy Configuration")
    print("=" * 60)

    # Check if files already exist
    models_exists = DEFAULT_MODELS_FILE.exists()
    config_exists = DEFAULT_CONFIG_FILE.exists()

    if models_exists and config_exists and not force:
        print(f"\nConfig files already exist:")
        print(f"  - {DEFAULT_MODELS_FILE}")
        print(f"  - {DEFAULT_CONFIG_FILE}")
        print("\nUse --init-force to overwrite existing files.")
        print("=" * 60 + "\n")
        return

    models_file, config_file = initialize_config(force=force)

    print(f"\nConfig directory: {DEFAULT_CONFIG_DIR}")
    print(f"Log directory: {DEFAULT_LOG_DIR}")
    print(f"\n{'Created' if force or not models_exists else 'Verified'}:")
    print(f"  - {models_file}")
    print(f"  - {config_file}")

    print("\nNext steps:")
    print("  1. Edit models.yaml to add your model configurations")
    print("  2. Edit config.json to customize server settings")
    print("  3. Run: anthropic-proxy start")

    print("=" * 60 + "\n")


def _get_host_port(args: argparse.Namespace) -> tuple[str, int]:
    """Get host and port from args, config, or defaults.

    Args:
        args: Parsed command-line arguments

    Returns:
        Tuple of (host, port)
    """
    config_path = args.config or DEFAULT_CONFIG_FILE
    config_data = load_config_file(config_path)

    host = args.host
    port = args.port

    if config_data:
        if host is None and "host" in config_data:
            host = config_data["host"]
        if port is None and "port" in config_data:
            port = int(config_data["port"])

    return host or config.host, port or config.port


def cmd_start(args: argparse.Namespace) -> None:
    """Start the server as a background daemon.

    Args:
        args: Parsed command-line arguments
    """
    # Check if already running
    status = daemon.get_daemon_status()
    if status is not None:
        print(f"Server is already running (PID: {status.pid})")
        if status.port:
            print(f"  Port: {status.port}")
        if status.uptime:
            print(f"  Uptime: {status.uptime}")
        print("\nUse 'anthropic-proxy stop' to stop the server first.")
        sys.exit(1)

    # Get host and port
    models_path = args.models or DEFAULT_MODELS_FILE
    config_path = args.config or DEFAULT_CONFIG_FILE
    host, port = _get_host_port(args)

    # Start daemon
    try:
        pid = daemon.start_daemon(
            host=host,
            port=port,
            models_path=models_path,
            config_path=config_path,
        )
        print(f"Server started successfully (PID: {pid})")
        print(f"  Host: {host}")
        print(f"  Port: {port}")
        print(f"  Log file: {daemon.DAEMON_LOG_FILE}")
        print("\nUse 'anthropic-proxy status' to check status.")
    except RuntimeError as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop the running daemon.

    Args:
        args: Parsed command-line arguments
    """
    status = daemon.get_daemon_status()
    if status is None:
        print("Server is not running.")
        sys.exit(0)

    print(f"Stopping server (PID: {status.pid})...")
    if daemon.stop_daemon():
        print("Server stopped successfully.")
    else:
        print("Error stopping server.", file=sys.stderr)
        sys.exit(1)


def cmd_restart(args: argparse.Namespace) -> None:
    """Restart the daemon.

    Args:
        args: Parsed command-line arguments
    """
    status = daemon.get_daemon_status()

    if status is not None:
        print(f"Stopping server (PID: {status.pid})...")
        if not daemon.stop_daemon():
            print("Error stopping server.", file=sys.stderr)
            sys.exit(1)
        print("Server stopped.")
    else:
        print("Server was not running.")

    # Start again
    cmd_start(args)


def cmd_status(args: argparse.Namespace) -> None:
    """Show daemon status.

    Args:
        args: Parsed command-line arguments
    """
    status = daemon.get_daemon_status()

    if status is None:
        print("Server Status: STOPPED")
        sys.exit(0)

    print("Server Status: RUNNING")
    print(f"  PID: {status.pid}")
    if status.port:
        print(f"  Port: {status.port}")
    if status.uptime:
        print(f"  Uptime: {status.uptime}")
    print(f"  Log file: {daemon.DAEMON_LOG_FILE}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="anthropic-proxy",
        description="Anthropic API Proxy - Converts Anthropic API requests to OpenAI-compatible APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  anthropic-proxy start              Start server in background
  anthropic-proxy start --port 8080  Start on custom port
  anthropic-proxy status             Show server status
  anthropic-proxy stop               Stop server

For configuration:
  anthropic-proxy --init             Initialize config files
  anthropic-proxy --print-config     View current configuration

For more information, see: https://github.com/tizee/anthropic-proxy
        """
    )

    # Utility options (global)
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to models.yaml configuration file",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to config.json configuration file",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize config directory with default files (skips if files exist)",
    )
    parser.add_argument(
        "--init-force",
        action="store_true",
        help="Force reinitialize config directory, overwriting existing files",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print current configuration and exit",
    )
    parser.add_argument(
        "--show-api-keys",
        action="store_true",
        help="Show actual API key values (default: redacted)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Server control commands",
    )

    # start subcommand
    start_parser = subparsers.add_parser(
        "start",
        help="Start server in background",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  anthropic-proxy start              Start with default config
  anthropic-proxy start --port 8080  Start on custom port
        """,
    )
    start_parser.add_argument(
        "--host",
        default=None,
        help=f"Host to bind to (default: from config or {config.host})",
    )
    start_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port to bind to (default: from config or {config.port})",
    )
    start_parser.set_defaults(func=cmd_start)

    # stop subcommand
    stop_parser = subparsers.add_parser(
        "stop",
        help="Stop running server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  anthropic-proxy stop               Stop the running server
        """,
    )
    stop_parser.set_defaults(func=cmd_stop)

    # restart subcommand
    restart_parser = subparsers.add_parser(
        "restart",
        help="Restart server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  anthropic-proxy restart            Restart the server
  anthropic-proxy restart --port 8080  Restart on custom port
        """,
    )
    restart_parser.add_argument(
        "--host",
        default=None,
        help=f"Host to bind to (default: from config or {config.host})",
    )
    restart_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port to bind to (default: from config or {config.port})",
    )
    restart_parser.set_defaults(func=cmd_restart)

    # status subcommand
    status_parser = subparsers.add_parser(
        "status",
        help="Show server status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  anthropic-proxy status             Show server status
        """,
    )
    status_parser.set_defaults(func=cmd_status)

    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = build_parser()
    return parser.parse_args()


def main():
    """Main CLI entry point."""
    args = parse_args()

    # Handle --init and --init-force
    if args.init_force:
        init_config(force=True)
        return
    elif args.init:
        init_config(force=False)
        return

    # Handle --print-config
    if args.print_config:
        # Ensure config directory exists for printing
        initialize_config(force=False)
        models_path = args.models or DEFAULT_MODELS_FILE
        config_path = args.config or DEFAULT_CONFIG_FILE
        print_config(models_path, config_path, show_api_keys=args.show_api_keys)
        return

    # If a subcommand was specified, execute it
    if hasattr(args, "func"):
        # Ensure config directory exists before running command
        initialize_config(force=False)
        args.func(args)
        return

    # No subcommand and no utility flags - show help
    build_parser().print_help()
    sys.exit(0)
