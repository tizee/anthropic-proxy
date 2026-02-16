"""
Daemon and PID management for anthropic-proxy server.
Handles starting, stopping, and monitoring background server instances.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

from .config_manager import DEFAULT_LOG_DIR

logger = logging.getLogger(__name__)

# PID file location
PID_FILE = DEFAULT_LOG_DIR / "anthropic-proxy.pid"
# Log file for daemon output
DAEMON_LOG_FILE = DEFAULT_LOG_DIR / "daemon.log"
# Shutdown timeout in seconds
SHUTDOWN_TIMEOUT = 10


class ProcessInfo(NamedTuple):
    """Information about a running process."""

    pid: int
    port: int | None
    uptime: str | None
    command: str


def get_pid_file() -> Path:
    """Get the path to the PID file.

    Returns:
        Path to PID file
    """
    return PID_FILE


def read_pid() -> int | None:
    """Read PID from file.

    Returns:
        PID if file exists and contains valid integer, None otherwise
    """
    pid_file = get_pid_file()
    if not pid_file.exists():
        return None

    try:
        pid_text = pid_file.read_text(encoding="utf-8").strip()
        return int(pid_text) if pid_text else None
    except (OSError, ValueError) as e:
        logger.warning(f"Failed to read PID file {pid_file}: {e}")
        return None


def write_pid(pid: int) -> None:
    """Write PID to file.

    Args:
        pid: Process ID to write
    """
    pid_file = get_pid_file()
    try:
        # Ensure log directory exists
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(pid), encoding="utf-8")
    except OSError as e:
        logger.error(f"Failed to write PID file {pid_file}: {e}")
        raise


def remove_pid_file() -> None:
    """Remove the PID file if it exists."""
    pid_file = get_pid_file()
    try:
        if pid_file.exists():
            pid_file.unlink()
    except OSError as e:
        logger.warning(f"Failed to remove PID file {pid_file}: {e}")


def is_running(pid: int) -> bool:
    """Check if a process with the given PID is running.

    Args:
        pid: Process ID to check

    Returns:
        True if process is running, False otherwise
    """
    try:
        # Send signal 0 to check if process exists
        # This doesn't actually send a signal, just checks liveness
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def get_process_info(pid: int) -> ProcessInfo | None:
    """Get information about a running process.

    Args:
        pid: Process ID to query

    Returns:
        ProcessInfo if process is running, None otherwise
    """
    if not is_running(pid):
        return None

    try:
        import psutil

        process = psutil.Process(pid)
        create_time = process.create_time()
        uptime_seconds = time.time() - create_time

        # Format uptime nicely
        hours, remainder = divmod(int(uptime_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h {minutes}m {seconds}s"

        # Try to extract port from command line
        port = None
        try:
            cmdline = process.cmdline()
            for i, arg in enumerate(cmdline):
                if arg == "--port" and i + 1 < len(cmdline):
                    port = int(cmdline[i + 1])
                    break
        except (psutil.AccessDenied, psutil.NoSuchProcess, ValueError):
            pass

        # Get command
        try:
            command = " ".join(process.cmdline())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            command = f"python (pid {pid})"

        return ProcessInfo(
            pid=pid,
            port=port,
            uptime=uptime_str,
            command=command,
        )
    except ImportError:
        # psutil not available, return basic info
        return ProcessInfo(
            pid=pid,
            port=None,
            uptime=None,
            command=f"python (pid {pid})",
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def kill_process(pid: int, timeout: int = SHUTDOWN_TIMEOUT) -> bool:
    """Kill a process gracefully, then forcefully if needed.

    Args:
        pid: Process ID to kill
        timeout: Seconds to wait for graceful shutdown

    Returns:
        True if process was killed, False otherwise
    """
    try:
        # First try SIGTERM for graceful shutdown
        os.kill(pid, signal.SIGTERM)

        # Wait for process to exit
        for _ in range(timeout * 10):  # Check 10 times per second
            time.sleep(0.1)
            if not is_running(pid):
                return True

        # If still running, use SIGKILL
        logger.warning(f"Process {pid} did not exit gracefully, using SIGKILL")
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        return not is_running(pid)

    except (OSError, ProcessLookupError) as e:
        logger.debug(f"Process {pid} already exited: {e}")
        return True


def clear_logs(config_path: Path | None = None) -> None:
    """Clear daemon.log and server.log files if cleanup_logs_on_start is enabled.

    Args:
        config_path: Path to config.json. If None, uses default location.
    """
    # Check if cleanup is enabled
    try:
        from .config import load_config_from_file

        file_config = load_config_from_file(config_path)
        cleanup_enabled = file_config.get("cleanup_logs_on_start", True)
        if not cleanup_enabled:
            logger.debug("Log cleanup disabled by config")
            return
    except Exception:
        # Default to enabled if config read fails
        pass

    log_files = [DAEMON_LOG_FILE]

    # Add server.log from config if available
    try:
        from .config import load_config_from_file

        file_config = load_config_from_file(config_path)
        log_path_str = file_config.get("log_file_path")
        if log_path_str:
            log_files.append(Path(log_path_str).expanduser())
    except Exception:
        pass

    for log_file in log_files:
        try:
            if log_file.exists():
                log_file.unlink()
                logger.debug(f"Cleared log file: {log_file}")
        except OSError as e:
            logger.debug(f"Failed to clear log file {log_file}: {e}")


def start_daemon(
    host: str,
    port: int,
    models_path: Path,
    config_path: Path,
) -> int:
    """Start the server as a background daemon process.

    Args:
        host: Host to bind to
        port: Port to bind to
        models_path: Path to models.yaml
        config_path: Path to config.json

    Returns:
        PID of the started process

    Raises:
        RuntimeError: If daemon fails to start
    """
    # Ensure log directory exists
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Clear log files on start (respects cleanup_logs_on_start setting)
    clear_logs(config_path)

    # Build command to run server module directly
    cmd = [
        sys.executable,
        "-m",
        "anthropic_proxy.server",
        "--host",
        str(host),
        "--port",
        str(port),
        "--models",
        str(models_path),
        "--config",
        str(config_path),
    ]

    # Build environment with config paths
    env = os.environ.copy()

    # Open log file for daemon output
    with DAEMON_LOG_FILE.open("a", encoding="utf-8") as log_file:
        try:
            process = subprocess.Popen(
                cmd,
                stdin=None,
                stdout=log_file,
                stderr=log_file,
                env=env,
                start_new_session=True,  # Detach from parent process
            )

            # Give it a moment to start
            time.sleep(0.5)

            # Check if process is still running
            if not is_running(process.pid):
                raise RuntimeError(
                    f"Daemon failed to start. Check {DAEMON_LOG_FILE} for details."
                )

            # Write PID file
            write_pid(process.pid)

            logger.info(f"Started daemon with PID {process.pid}")
            return process.pid

        except Exception as e:
            # Clean up if something went wrong
            if "process" in locals() and is_running(process.pid):
                kill_process(process.pid)
            raise RuntimeError(f"Failed to start daemon: {e}") from e


def stop_daemon() -> bool:
    """Stop the running daemon.

    Returns:
        True if daemon was stopped or was not running, False on failure
    """
    pid = read_pid()
    if pid is None:
        # No PID file, consider "stopped"
        return True

    if not is_running(pid):
        # PID file exists but process is dead - clean up
        remove_pid_file()
        return True

    success = kill_process(pid)
    if success:
        remove_pid_file()
        logger.info(f"Stopped daemon with PID {pid}")

    return success


def get_daemon_status() -> ProcessInfo | None:
    """Get status of the daemon.

    Returns:
        ProcessInfo if daemon is running, None otherwise
    """
    pid = read_pid()
    if pid is None:
        return None

    info = get_process_info(pid)
    if info is None:
        # Process is dead but PID file exists - clean up
        remove_pid_file()

    return info
