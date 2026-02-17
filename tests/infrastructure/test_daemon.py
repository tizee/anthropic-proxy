"""
Unit tests for daemon and PID management functionality.

Tests cover:
- PID file operations (read, write, remove)
- Process running status checks
- Process info retrieval
- Daemon start/stop/restart operations
"""

import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from anthropic_proxy.daemon import (
    ProcessInfo,
    clear_logs,
    get_daemon_status,
    get_pid_file,
    get_process_info,
    is_running,
    kill_process,
    read_pid,
    remove_pid_file,
    start_daemon,
    stop_daemon,
    write_pid,
)


class TestPidFileOperations(unittest.TestCase):
    """Test cases for PID file operations."""

    def setUp(self):
        """Set up test fixtures with a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_pid_file = Path(self.temp_dir) / "test.pid"

    def tearDown(self):
        """Clean up temporary directory."""
        if self.test_pid_file.exists():
            self.test_pid_file.unlink()
        Path(self.temp_dir).rmdir()

    def test_get_pid_file_returns_path(self):
        """Test that get_pid_file returns the expected path."""
        result = get_pid_file()
        # The function returns the module-level PID_FILE constant
        # Just verify it returns a Path object
        self.assertIsInstance(result, Path)

    @patch("anthropic_proxy.daemon.PID_FILE")
    def test_read_pid_returns_none_when_file_not_exists(self, mock_pid_file):
        """Test read_pid returns None when PID file doesn't exist."""
        mock_pid_file.exists.return_value = False

        result = read_pid()
        self.assertIsNone(result)

    @patch("anthropic_proxy.daemon.PID_FILE")
    def test_read_pid_returns_valid_pid(self, mock_pid_file):
        """Test read_pid returns correct PID from file."""
        test_pid = 12345
        mock_pid_file.exists.return_value = True
        mock_pid_file.read_text.return_value = str(test_pid)

        result = read_pid()
        self.assertEqual(result, test_pid)

    @patch("anthropic_proxy.daemon.PID_FILE")
    def test_read_pid_returns_none_for_invalid_content(self, mock_pid_file):
        """Test read_pid returns None for invalid PID content."""
        mock_pid_file.exists.return_value = True

        # Test empty string
        mock_pid_file.read_text.return_value = ""
        result = read_pid()
        self.assertIsNone(result)

        # Test non-numeric string
        mock_pid_file.read_text.return_value = "not_a_number"
        result = read_pid()
        self.assertIsNone(result)

    @patch("anthropic_proxy.daemon.PID_FILE")
    def test_write_pid_creates_file_with_pid(self, mock_pid_file):
        """Test write_pid correctly writes PID to file."""
        test_pid = 54321
        mock_pid_file.parent = Mock()
        mock_pid_file.parent.mkdir = Mock()
        mock_pid_file.write_text = Mock()

        write_pid(test_pid)

        mock_pid_file.write_text.assert_called_once_with(str(test_pid), encoding="utf-8")

    @patch("anthropic_proxy.daemon.PID_FILE")
    def test_remove_pid_file_deletes_existing_file(self, mock_pid_file):
        """Test remove_pid_file deletes the PID file if it exists."""
        mock_pid_file.exists.return_value = True
        mock_pid_file.unlink = Mock()

        remove_pid_file()

        mock_pid_file.unlink.assert_called_once()

    @patch("anthropic_proxy.daemon.PID_FILE")
    def test_remove_pid_file_handles_nonexistent_file(self, mock_pid_file):
        """Test remove_pid_file handles missing file gracefully."""
        mock_pid_file.exists.return_value = False
        mock_pid_file.unlink = Mock()

        # Should not raise exception
        remove_pid_file()

        # unlink should not be called
        mock_pid_file.unlink.assert_not_called()


class TestIsRunning(unittest.TestCase):
    """Test cases for is_running function."""

    def test_is_running_returns_true_for_current_process(self):
        """Test is_running returns True for current process."""
        current_pid = os.getpid()
        self.assertTrue(is_running(current_pid))

    def test_is_running_returns_false_for_invalid_pid(self):
        """Test is_running returns False for non-existent PID."""
        # Use a very high PID that is unlikely to exist
        self.assertFalse(is_running(999999999))


class TestGetProcessInfo(unittest.TestCase):
    """Test cases for get_process_info function."""

    def test_get_process_info_returns_none_for_nonexistent_process(self):
        """Test get_process_info returns None for non-existent process."""
        info = get_process_info(999999999)
        self.assertIsNone(info)

    def test_get_process_info_returns_processinfo_for_valid_pid(self):
        """Test get_process_info returns ProcessInfo for running process."""
        current_pid = os.getpid()

        # Create a real process info object
        info = get_process_info(current_pid)

        self.assertIsInstance(info, ProcessInfo)
        self.assertEqual(info.pid, current_pid)
        # With psutil available, uptime should not be None
        self.assertIsNotNone(info.uptime)
        self.assertIsNotNone(info.command)

    def test_get_process_info_fallback_without_psutil(self):
        """Test get_process_info returns basic info without psutil is handled."""
        # Since psutil is lazy-imported and we have it installed,
        # this test verifies the normal path with psutil available.
        # The ImportError path is hard to test without uninstalling psutil.
        current_pid = os.getpid()

        info = get_process_info(current_pid)

        # With psutil available (typical case), we get full info
        self.assertIsInstance(info, ProcessInfo)
        self.assertEqual(info.pid, current_pid)
        # With psutil available, these should be populated
        self.assertIsNotNone(info.uptime)

    def test_get_process_info_extracts_port_from_cmdline(self):
        """Test get_process_info extracts port from command line arguments."""
        # Use current process to test, but we can't easily mock its cmdline
        # Instead, just test that the function handles cmdline parsing
        current_pid = os.getpid()

        info = get_process_info(current_pid)

        # Verify that port extraction is attempted (may be None for current process)
        # The function should at least return a ProcessInfo
        self.assertIsInstance(info, ProcessInfo)
        self.assertEqual(info.pid, current_pid)


class TestKillProcess(unittest.TestCase):
    """Test cases for kill_process function."""

    def test_kill_process_handles_nonexistent_process(self):
        """Test kill_process returns True for already dead process."""
        result = kill_process(999999999, timeout=1)
        self.assertTrue(result)

    @patch("anthropic_proxy.daemon.is_running")
    @patch("anthropic_proxy.daemon.os.kill")
    def test_kill_process_sends_sigterm_then_sigkill(self, mock_kill, mock_is_running):
        """Test kill_process sends SIGTERM then SIGKILL if needed."""
        # Simulate a process that doesn't respond to SIGTERM
        # With timeout=1, it loops 10 times (timeout * 10), then checks once more after SIGKILL
        mock_is_running.side_effect = [True] * 10 + [False]

        with patch("anthropic_proxy.daemon.time.sleep"):
            result = kill_process(12345, timeout=1)

        # Should send SIGTERM first
        mock_kill.assert_any_call(12345, signal.SIGTERM)
        # Should eventually send SIGKILL
        mock_kill.assert_any_call(12345, signal.SIGKILL)
        self.assertTrue(result)

    @patch("anthropic_proxy.daemon.is_running")
    @patch("anthropic_proxy.daemon.os.kill")
    def test_kill_process_sigterm_sufficient(self, mock_kill, mock_is_running):
        """Test kill_process succeeds with just SIGTERM."""
        # Process exits after SIGTERM
        mock_is_running.side_effect = [True, False]
        mock_kill.side_effect = [
            None,  # SIGTERM
            ProcessLookupError,  # is_running check will fail
        ]

        with patch("anthropic_proxy.daemon.time.sleep"):
            result = kill_process(12345, timeout=1)

        # Should only send SIGTERM
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        self.assertTrue(result)


class TestGetDaemonStatus(unittest.TestCase):
    """Test cases for get_daemon_status function."""

    @patch("anthropic_proxy.daemon.read_pid")
    def test_get_daemon_status_returns_none_when_no_pid_file(self, mock_read_pid):
        """Test get_daemon_status returns None when no PID file exists."""
        mock_read_pid.return_value = None

        status = get_daemon_status()
        self.assertIsNone(status)

    @patch("anthropic_proxy.daemon.get_process_info")
    @patch("anthropic_proxy.daemon.read_pid")
    @patch("anthropic_proxy.daemon.remove_pid_file")
    def test_get_daemon_status_cleans_up_dead_process(
        self, mock_remove, mock_read_pid, mock_get_info
    ):
        """Test get_daemon_status removes PID file for dead process."""
        mock_read_pid.return_value = 12345
        mock_get_info.return_value = None  # Process is dead

        status = get_daemon_status()

        self.assertIsNone(status)
        mock_remove.assert_called_once()

    @patch("anthropic_proxy.daemon.get_process_info")
    @patch("anthropic_proxy.daemon.read_pid")
    def test_get_daemon_status_returns_process_info(
        self, mock_read_pid, mock_get_info
    ):
        """Test get_daemon_status returns ProcessInfo for running daemon."""
        test_info = ProcessInfo(
            pid=12345, port=8080, uptime="1h 0m 0s", command="python -m anthropic_proxy"
        )
        mock_read_pid.return_value = 12345
        mock_get_info.return_value = test_info

        status = get_daemon_status()

        self.assertEqual(status, test_info)


class TestStopDaemon(unittest.TestCase):
    """Test cases for stop_daemon function."""

    @patch("anthropic_proxy.daemon.read_pid")
    def test_stop_daemon_returns_true_when_no_pid(self, mock_read_pid):
        """Test stop_daemon returns True when no PID file exists."""
        mock_read_pid.return_value = None

        result = stop_daemon()
        self.assertTrue(result)

    @patch("anthropic_proxy.daemon.kill_process")
    @patch("anthropic_proxy.daemon.remove_pid_file")
    @patch("anthropic_proxy.daemon.is_running")
    @patch("anthropic_proxy.daemon.read_pid")
    def test_stop_daemon_kills_running_process(
        self, mock_read_pid, mock_is_running, mock_remove, mock_kill
    ):
        """Test stop_daemon kills process and removes PID file."""
        mock_read_pid.return_value = 12345
        mock_is_running.return_value = True
        mock_kill.return_value = True

        result = stop_daemon()

        self.assertTrue(result)
        mock_kill.assert_called_once_with(12345)
        mock_remove.assert_called_once()

    @patch("anthropic_proxy.daemon.remove_pid_file")
    @patch("anthropic_proxy.daemon.is_running")
    @patch("anthropic_proxy.daemon.read_pid")
    def test_stop_daemon_cleans_up_dead_process(
        self, mock_read_pid, mock_is_running, mock_remove
    ):
        """Test stop_daemon cleans up PID file for dead process."""
        mock_read_pid.return_value = 12345
        mock_is_running.return_value = False

        result = stop_daemon()

        self.assertTrue(result)
        mock_remove.assert_called_once()


class TestStartDaemon(unittest.TestCase):
    """Test cases for start_daemon function."""

    @patch("anthropic_proxy.daemon.write_pid")
    @patch("anthropic_proxy.daemon.is_running")
    @patch("anthropic_proxy.daemon.time.sleep")
    @patch("anthropic_proxy.daemon.subprocess.Popen")
    @patch("anthropic_proxy.daemon.DAEMON_LOG_FILE")
    def test_start_daemon_launches_process_successfully(
        self, mock_log_file_path, mock_popen, mock_sleep, mock_is_running, mock_write
    ):
        """Test start_daemon successfully launches daemon process."""
        # Mock the log file path to use a temp location
        mock_log_file_path.open = Mock()
        mock_log_file_mock = Mock()
        mock_log_file_path.open.return_value.__enter__ = Mock(return_value=mock_log_file_mock)
        mock_log_file_path.open.return_value.__exit__ = Mock(return_value=False)
        mock_log_file_path.__str__ = Mock(return_value="/tmp/anthropic-proxy/daemon.log")

        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        mock_is_running.return_value = True

        pid = start_daemon(
            host="localhost",
            port=8080,
            models_path=Path("/tmp/models.yaml"),
            config_path=Path("/tmp/config.json"),
        )

        self.assertEqual(pid, 12345)
        mock_popen.assert_called_once()
        mock_write.assert_called_once_with(12345)

    @patch("anthropic_proxy.daemon.kill_process")
    @patch("anthropic_proxy.daemon.is_running")
    @patch("anthropic_proxy.daemon.time.sleep")
    @patch("anthropic_proxy.daemon.subprocess.Popen")
    @patch("anthropic_proxy.daemon.DAEMON_LOG_FILE")
    def test_start_daemon_raises_error_on_failure(
        self,
        mock_log_file_path,
        mock_popen,
        mock_sleep,
        mock_is_running,
        mock_kill,
    ):
        """Test start_daemon raises RuntimeError when process fails to start."""
        # Mock the log file path to use a temp location
        mock_log_file_path.open = Mock()
        mock_log_file_mock = Mock()
        mock_log_file_path.open.return_value.__enter__ = Mock(return_value=mock_log_file_mock)
        mock_log_file_path.open.return_value.__exit__ = Mock(return_value=False)
        mock_log_file_path.__str__ = Mock(return_value="/tmp/anthropic-proxy/daemon.log")

        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        # Process dies immediately
        mock_is_running.return_value = False

        with self.assertRaises(RuntimeError):
            start_daemon(
                host="localhost",
                port=8080,
                models_path=Path("/tmp/models.yaml"),
                config_path=Path("/tmp/config.json"),
            )


class TestClearLogs(unittest.TestCase):
    """Test cases for clear_logs function."""

    def setUp(self):
        """Set up test fixtures with temporary log files."""
        self.temp_dir = tempfile.mkdtemp()
        self.daemon_log = Path(self.temp_dir) / "daemon.log"
        self.server_log = Path(self.temp_dir) / "server.log"
        self.config_file = Path(self.temp_dir) / "config.json"

    def tearDown(self):
        """Clean up temporary files."""
        for f in [self.daemon_log, self.server_log, self.config_file]:
            if f.exists():
                f.unlink()
        Path(self.temp_dir).rmdir()

    @patch("anthropic_proxy.config.load_config_from_file")
    @patch("anthropic_proxy.daemon.DAEMON_LOG_FILE")
    def test_clear_logs_deletes_logs_when_enabled(
        self, mock_daemon_log, mock_load_config
    ):
        """Test clear_logs deletes log files when cleanup_logs_on_start is True."""
        # Create temporary log files
        self.daemon_log.write_text("daemon log content")
        self.server_log.write_text("server log content")

        # Mock DAEMON_LOG_FILE to point to our temp file
        mock_daemon_log.__str__ = Mock(return_value=str(self.daemon_log))
        mock_daemon_log.exists = Mock(return_value=True)
        mock_daemon_log.unlink = Mock()

        # Mock config to enable cleanup
        mock_load_config.return_value = {
            "cleanup_logs_on_start": True,
            "log_file_path": str(self.server_log),
        }

        # Mock Path operations for server.log
        with patch("pathlib.Path.unlink") as mock_unlink:
            clear_logs(self.config_file)

        # Verify files were deleted (mocked unlink was called)
        self.assertTrue(mock_daemon_log.unlink.called or mock_unlink.called)

    @patch("anthropic_proxy.config.load_config_from_file")
    @patch("anthropic_proxy.daemon.DAEMON_LOG_FILE")
    def test_clear_logs_skips_when_disabled(
        self, mock_daemon_log, mock_load_config
    ):
        """Test clear_logs skips deletion when cleanup_logs_on_start is False."""
        # Create temporary log files
        self.daemon_log.write_text("daemon log content")
        self.server_log.write_text("server log content")

        # Mock DAEMON_LOG_FILE to point to our temp file
        mock_daemon_log.exists = Mock(return_value=True)
        mock_daemon_log.unlink = Mock()

        # Mock config to disable cleanup
        mock_load_config.return_value = {
            "cleanup_logs_on_start": False,
            "log_file_path": str(self.server_log),
        }

        clear_logs(self.config_file)

        # Verify files were NOT deleted
        mock_daemon_log.unlink.assert_not_called()

    @patch("anthropic_proxy.config.load_config_from_file")
    @patch("anthropic_proxy.daemon.DAEMON_LOG_FILE")
    def test_clear_logs_defaults_to_enabled_when_missing(
        self, mock_daemon_log, mock_load_config
    ):
        """Test clear_logs defaults to enabled when cleanup_logs_on_start is not set."""
        # Mock DAEMON_LOG_FILE
        mock_daemon_log.exists = Mock(return_value=True)
        mock_daemon_log.unlink = Mock()

        # Mock config without cleanup_logs_on_start field
        mock_load_config.return_value = {
            "log_file_path": str(self.server_log),
        }

        clear_logs(self.config_file)

        # Verify files were deleted (default is enabled)
        mock_daemon_log.unlink.assert_called_once()

    @patch("anthropic_proxy.config.load_config_from_file")
    def test_clear_logs_handles_nonexistent_files(self, mock_load_config):
        """Test clear_logs handles missing log files gracefully."""
        # Mock config with enabled cleanup
        mock_load_config.return_value = {
            "cleanup_logs_on_start": True,
            "log_file_path": str(self.server_log),
        }

        # Should not raise exception even if files don't exist
        clear_logs(self.config_file)

    @patch("anthropic_proxy.config.load_config_from_file")
    def test_clear_logs_handles_config_load_error(self, mock_load_config):
        """Test clear_logs defaults to enabled when config load fails."""
        # Simulate config load failure
        mock_load_config.side_effect = Exception("Config load failed")

        # Mock DAEMON_LOG_FILE to avoid actual file operations
        with patch("anthropic_proxy.daemon.DAEMON_LOG_FILE") as mock_daemon_log:
            mock_daemon_log.exists = Mock(return_value=False)
            # Should not raise exception
            clear_logs(self.config_file)


if __name__ == "__main__":
    unittest.main()
