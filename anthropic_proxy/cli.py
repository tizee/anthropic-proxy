"""
Command-line interface for anthropic-proxy.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml

from . import daemon
from .config import config
from .config_manager import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_FILE,
    DEFAULT_LOG_DIR,
    DEFAULT_MODELS_FILE,
    get_default_log_file_path,
    initialize_config,
    load_config_file,
)

logger = logging.getLogger(__name__)


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


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


def _provider_status_lines() -> list[str]:
    from .antigravity import antigravity_auth
    from .codex import codex_auth
    from .gemini import gemini_auth

    providers = [
        ("Codex", codex_auth.has_auth()),
        ("Gemini", gemini_auth.has_auth()),
        ("Antigravity", antigravity_auth.has_auth()),
    ]

    lines = []
    for name, authed in providers:
        status = f"{Colors.GREEN}AUTHED{Colors.RESET}" if authed else f"{Colors.RED}NOT AUTHED{Colors.RESET}"
        lines.append(f"{name} {status}")
    return lines


def _default_provider_model_ids() -> list[str]:
    from .antigravity import DEFAULT_ANTIGRAVITY_MODELS
    from .codex import DEFAULT_CODEX_MODELS
    from .gemini import DEFAULT_GEMINI_MODELS

    model_ids = []
    model_ids.extend([f"codex/{model_id}" for model_id in DEFAULT_CODEX_MODELS.keys()])
    model_ids.extend([f"gemini/{model_id}" for model_id in DEFAULT_GEMINI_MODELS.keys()])
    model_ids.extend([f"antigravity/{model_id}" for model_id in DEFAULT_ANTIGRAVITY_MODELS.keys()])
    return model_ids


def _load_model_ids(models_path: Path) -> list[str]:
    if not models_path.exists():
        return []

    try:
        with models_path.open("r", encoding="utf-8") as f:
            models_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"{Colors.RED}Error loading models.yaml:{Colors.RESET} {e}", file=sys.stderr)
        return []

    if not models_data:
        return []

    if not isinstance(models_data, list):
        return []

    model_ids = []
    for model in models_data:
        if isinstance(model, dict) and "model_id" in model:
            model_ids.append(str(model["model_id"]))
    return model_ids


def _format_quota_model_lines(models: dict) -> list[str]:
    lines: list[str] = []
    model_ids = sorted(models.keys())
    for model_id in model_ids:
        info = models.get(model_id) or {}
        if not isinstance(info, dict):
            info = {}
        quota = info.get("quotaInfo") or {}
        if not isinstance(quota, dict):
            quota = {}
        remaining = quota.get("remainingFraction")
        remaining_text = "unknown"
        if isinstance(remaining, (int, float)):
            remaining_text = f"{remaining * 100:.0f}%"
        reset_time = quota.get("resetTime") or "unknown"
        recommended = " (recommended)" if info.get("recommended") else ""
        lines.append(
            f"  - {model_id}: remaining {remaining_text}, reset {reset_time}{recommended}"
        )
    return lines


def _run_antigravity_test() -> tuple[str, bool, str, dict]:
    from .antigravity import antigravity_auth, fetch_antigravity_quota_models

    async def _fetch():
        access_token = await antigravity_auth.get_access_token()
        project_id = antigravity_auth.get_project_id()
        await antigravity_auth.ensure_project_context(force=True)
        project_id = antigravity_auth.get_project_id()
        if not project_id:
            raise RuntimeError(
                "Antigravity Project ID could not be resolved. Re-login required."
            )
        data, base_url = await fetch_antigravity_quota_models(access_token, project_id)
        return project_id, False, base_url, data

    return asyncio.run(_fetch())


def cmd_provider(args: argparse.Namespace) -> None:
    """List auth providers or available model IDs."""
    did_output = False
    provider = getattr(args, "provider", None)
    run_test = bool(getattr(args, "test", False))

    if run_test:
        if provider != "antigravity":
            print(f"{Colors.RED}Provider test requires 'antigravity'.{Colors.RESET}", file=sys.stderr)
            sys.exit(1)
        try:
            project_id, _, base_url, data = _run_antigravity_test()
        except Exception as exc:
            print(f"{Colors.RED}Antigravity test failed:{Colors.RESET} {exc}", file=sys.stderr)
            sys.exit(1)

        print("Antigravity quota test")
        print(f"Project ID: {project_id}")
        print(f"Base URL: {base_url}")
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, dict) or not models:
            print("Models: 0")
            print("No models returned.")
        else:
            recommended_count = sum(1 for info in models.values() if isinstance(info, dict) and info.get("recommended"))
            print(f"Models: {len(models)} (recommended: {recommended_count})")
            for line in _format_quota_model_lines(models):
                print(line)
        did_output = True

    if args.list:
        for line in _provider_status_lines():
            print(line)
        did_output = True

    if args.list_models:
        models_path = args.models or DEFAULT_MODELS_FILE
        model_ids = _load_model_ids(models_path)
        seen = set(model_ids)
        for model_id in _default_provider_model_ids():
            if model_id not in seen:
                model_ids.append(model_id)
                seen.add(model_id)
        for model_id in model_ids:
            print(model_id)
        did_output = True

    if not did_output:
        print(f"{Colors.YELLOW}Please specify --list and/or --models.{Colors.RESET}")
        print("Examples:")
        print("  anthropic-proxy provider --list")
        print("  anthropic-proxy provider --models")
        sys.exit(1)


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize config directory with default files.

    Args:
        args: Parsed command-line arguments (with force flag)
    """
    c = Colors
    force_config = getattr(args, "force", False)

    print(f"\n{c.BOLD}Initializing Anthropic Proxy Configuration{c.RESET}")
    print("=" * 50)

    # Check if files already exist
    models_exists = DEFAULT_MODELS_FILE.exists()
    config_exists = DEFAULT_CONFIG_FILE.exists()

    # Models file: never overwrite (user's custom configurations)
    if not models_exists:
        from .config_manager import create_default_models_file
        create_default_models_file(force=False)
        print(f"  {c.GREEN}Created:{c.RESET}  {DEFAULT_MODELS_FILE}")
    else:
        print(f"  {c.CYAN}Exists:{c.RESET}   {DEFAULT_MODELS_FILE}")

    # Config file: overwrite only with --force
    if not config_exists or force_config:
        from .config_manager import create_default_config_file
        create_default_config_file(force=force_config)
        action = "Reset" if force_config and config_exists else "Created"
        print(f"  {c.GREEN}{action}:{c.RESET}  {DEFAULT_CONFIG_FILE}")
    else:
        print(f"  {c.CYAN}Exists:{c.RESET}   {DEFAULT_CONFIG_FILE}")

    # Ensure log directory exists
    from .config_manager import ensure_log_dir
    ensure_log_dir()

    print(f"\n{c.DIM}Directories:{c.RESET}")
    print(f"  {c.CYAN}Config:{c.RESET} {DEFAULT_CONFIG_DIR}")
    print(f"  {c.CYAN}Logs:{c.RESET}   {DEFAULT_LOG_DIR}")

    print(f"\n{c.DIM}Next steps:{c.RESET}")
    print(f"  1. Edit {c.BOLD}models.yaml{c.RESET} to add your model configurations")
    print(f"  2. Edit {c.BOLD}config.json{c.RESET} to customize server settings")
    print(f"  3. Run: {c.BOLD}anthropic-proxy start{c.RESET}")
    print()


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
    c = Colors

    # Check if already running
    status = daemon.get_daemon_status()
    if status is not None:
        print(f"{c.YELLOW}Server is already running{c.RESET}")
        print(f"  {c.CYAN}PID:{c.RESET}    {c.BOLD}{status.pid}{c.RESET}")
        if status.port:
            print(f"  {c.CYAN}Port:{c.RESET}   {c.BOLD}{status.port}{c.RESET}")
        if status.uptime:
            print(f"  {c.CYAN}Uptime:{c.RESET} {status.uptime}")
        print(f"\n{c.DIM}Use 'anthropic-proxy stop' to stop the server first.{c.RESET}")
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
        print(f"{c.GREEN}{c.BOLD}Server started successfully{c.RESET}")
        print(f"  {c.CYAN}PID:{c.RESET}  {c.BOLD}{pid}{c.RESET}")
        print(f"  {c.CYAN}Host:{c.RESET} {host}")
        print(f"  {c.CYAN}Port:{c.RESET} {c.BOLD}{port}{c.RESET}")
        print(f"\n{c.DIM}Logs:{c.RESET}")
        print(f"  {c.CYAN}Server:{c.RESET} {get_default_log_file_path()}")
        print(f"  {c.CYAN}Daemon:{c.RESET} {daemon.DAEMON_LOG_FILE}")
    except RuntimeError as e:
        print(f"{c.RED}Error starting server:{c.RESET} {e}", file=sys.stderr)
        sys.exit(1)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop the running daemon.

    Args:
        args: Parsed command-line arguments
    """
    c = Colors
    status = daemon.get_daemon_status()

    if status is None:
        print(f"{c.DIM}Server is not running.{c.RESET}")
        sys.exit(0)

    print(f"Stopping server {c.DIM}(PID: {status.pid}){c.RESET}...")
    if daemon.stop_daemon():
        print(f"{c.GREEN}{c.BOLD}Server stopped successfully.{c.RESET}")
    else:
        print(f"{c.RED}Error stopping server.{c.RESET}", file=sys.stderr)
        sys.exit(1)


def cmd_restart(args: argparse.Namespace) -> None:
    """Restart the daemon.

    Args:
        args: Parsed command-line arguments
    """
    c = Colors
    status = daemon.get_daemon_status()

    if status is not None:
        print(f"Stopping server {c.DIM}(PID: {status.pid}){c.RESET}...")
        if not daemon.stop_daemon():
            print(f"{c.RED}Error stopping server.{c.RESET}", file=sys.stderr)
            sys.exit(1)
        print(f"{c.GREEN}Server stopped.{c.RESET}")
    else:
        print(f"{c.DIM}Server was not running.{c.RESET}")

    # Start again
    cmd_start(args)


def cmd_status(args: argparse.Namespace) -> None:
    """Show daemon status.

    Args:
        args: Parsed command-line arguments
    """
    status = daemon.get_daemon_status()
    c = Colors

    if status is None:
        print(f"Server Status: {c.RED}{c.BOLD}STOPPED{c.RESET}")
        sys.exit(0)

    print(f"Server Status: {c.GREEN}{c.BOLD}RUNNING{c.RESET}")
    print(f"  {c.CYAN}PID:{c.RESET}    {c.BOLD}{status.pid}{c.RESET}")
    if status.port:
        print(f"  {c.CYAN}Port:{c.RESET}   {c.BOLD}{status.port}{c.RESET}")
    if status.uptime:
        print(f"  {c.CYAN}Uptime:{c.RESET} {status.uptime}")
    print(f"\n{c.DIM}Logs:{c.RESET}")
    print(f"  {c.CYAN}Server:{c.RESET} {get_default_log_file_path()}")
    print(f"  {c.CYAN}Daemon:{c.RESET} {daemon.DAEMON_LOG_FILE}")


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
  anthropic-proxy init               Initialize config files
  anthropic-proxy start              Start server in background
  anthropic-proxy start --port 8080  Start on custom port
  anthropic-proxy status             Show server status
  anthropic-proxy stop               Stop server
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

    # init subcommand
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  anthropic-proxy init               Initialize config files (skip if exist)
  anthropic-proxy init --force       Reset config.json to defaults (keeps models.yaml)
        """,
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Reset config.json to defaults (models.yaml is never overwritten)",
    )
    init_parser.set_defaults(func=cmd_init)

    # login subcommand
    login_parser = subparsers.add_parser(
        "login",
        help="Login to provider subscription (e.g. Codex)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This command opens a browser to authenticate with supported providers.

Examples:
  anthropic-proxy login --codex      Login to OpenAI Codex subscription
  anthropic-proxy login --gemini     Login to Google Gemini subscription
        """,
    )
    login_parser.add_argument(
        "--codex",
        action="store_true",
        help="Login to OpenAI Codex subscription",
    )
    login_parser.add_argument(
        "--gemini",
        action="store_true",
        help="Login to Google Gemini subscription",
    )
    login_parser.add_argument(
        "--antigravity",
        action="store_true",
        help="Login to Google Antigravity (Internal) subscription",
    )
    login_parser.set_defaults(func=cmd_login)

    # provider subcommand
    provider_parser = subparsers.add_parser(
        "provider",
        help="Show auth providers and available models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  anthropic-proxy provider --list
  anthropic-proxy provider --models
  anthropic-proxy provider antigravity --test
        """,
    )
    provider_parser.add_argument(
        "provider",
        nargs="?",
        help="Provider name (e.g. antigravity) for provider-specific actions",
    )
    provider_parser.add_argument(
        "--list",
        action="store_true",
        help="List auth providers and their OAuth status",
    )
    provider_parser.add_argument(
        "--models",
        dest="list_models",
        action="store_true",
        help="List available model IDs (custom + provider defaults)",
    )
    provider_parser.add_argument(
        "--test",
        action="store_true",
        help="Test provider account/quota (e.g. antigravity)",
    )
    provider_parser.set_defaults(func=cmd_provider)

    return parser


def cmd_login(args: argparse.Namespace) -> None:
    """Login to provider subscription.

    Args:
        args: Parsed command-line arguments
    """
    if args.codex:
        from .codex import codex_auth
        codex_auth.login()
        return

    if args.gemini:
        from .gemini import gemini_auth
        gemini_auth.login()
        return

    if args.antigravity:
        from .antigravity import antigravity_auth
        antigravity_auth.login()
        return

    # Default behavior if no flags (or show help)
    # For now, if no flags are provided, we can default to codex OR show help.
    # Given the request is to make it an optional parameter, implies explicit selection.
    print(f"{Colors.YELLOW}Please specify a provider to login.{Colors.RESET}")
    print("Available providers:")
    print(f"  {Colors.GREEN}--codex{Colors.RESET}       Login to OpenAI Codex subscription")
    print(f"  {Colors.GREEN}--gemini{Colors.RESET}      Login to Google Gemini subscription")
    print(f"  {Colors.GREEN}--antigravity{Colors.RESET} Login to Google Antigravity subscription")
    sys.exit(1)


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
