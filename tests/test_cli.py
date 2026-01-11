"""
Unit tests for CLI functionality.

Tests cover:
- Argument parsing and subcommands
- Utility functions (redact_api_keys, print_config)
- Command handlers (init, start, stop, restart, status)
"""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from anthropic_proxy.cli import (
    _get_host_port,
    build_parser,
    cmd_init,
    cmd_restart,
    cmd_start,
    cmd_status,
    cmd_stop,
    parse_args,
    print_config,
    redact_api_keys,
)
from anthropic_proxy.daemon import ProcessInfo


class TestRedactApiKeys(unittest.TestCase):
    """Test cases for redact_api_keys function."""

    def test_redact_api_keys_removes_api_key_field(self):
        """Test redact_api_keys replaces api_key values with REDACTED."""
        models_data = [
            {"model_id": "model1", "api_key": "secret-key-1"},
            {"model_id": "model2", "api_key": "secret-key-2"},
            {"model_id": "model3", "api_key": "secret-key-3"},
        ]

        result = redact_api_keys(models_data, show_keys=False)

        self.assertEqual(len(result), 3)
        for model in result:
            self.assertEqual(model["api_key"], "<REDACTED>")
            # Ensure original data wasn't modified
            self.assertIn(model["model_id"], ["model1", "model2", "model3"])

    def test_redact_api_keys_preserves_data_when_show_keys_true(self):
        """Test redact_api_keys preserves api_key when show_keys=True."""
        models_data = [
            {"model_id": "model1", "api_key": "secret-key-1"},
            {"model_id": "model2", "api_key": "secret-key-2"},
        ]

        result = redact_api_keys(models_data, show_keys=True)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["api_key"], "secret-key-1")
        self.assertEqual(result[1]["api_key"], "secret-key-2")

    def test_redact_api_keys_handles_models_without_api_key(self):
        """Test redact_api_keys handles models without api_key field."""
        models_data = [
            {"model_id": "model1", "api_base": "https://api.example.com"},
            {"model_id": "model2", "api_key": "secret-key"},
        ]

        result = redact_api_keys(models_data, show_keys=False)

        self.assertEqual(len(result), 2)
        self.assertNotIn("api_key", result[0])
        self.assertEqual(result[1]["api_key"], "<REDACTED>")

    def test_redact_api_keys_does_not_modify_original_list(self):
        """Test redact_api_keys creates a new list, not modifying original."""
        models_data = [
            {"model_id": "model1", "api_key": "secret-key"},
        ]

        result = redact_api_keys(models_data, show_keys=False)

        # Original should still have the api_key
        self.assertEqual(models_data[0]["api_key"], "secret-key")
        # Result should have redacted key
        self.assertEqual(result[0]["api_key"], "<REDACTED>")


class TestGetHostPort(unittest.TestCase):
    """Test cases for _get_host_port function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_get_host_port_from_args_only(self):
        """Test _get_host_port returns host/port from args when provided."""
        args = argparse.Namespace(host="0.0.0.0", port=9000, config=None)
        args.config = None

        with patch("anthropic_proxy.cli.load_config_file", return_value={}):
            with patch("anthropic_proxy.cli.config") as mock_config:
                mock_config.host = "127.0.0.1"
                mock_config.port = 8080
                host, port = _get_host_port(args)

        self.assertEqual(host, "0.0.0.0")
        self.assertEqual(port, 9000)

    def test_get_host_port_from_config_file(self):
        """Test _get_host_port reads from config file when args not provided."""
        config_path = Path(self.temp_dir) / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump({"host": "0.0.0.0", "port": 9000}, f)

        args = argparse.Namespace(host=None, port=None, config=config_path)

        with patch("anthropic_proxy.cli.config") as mock_config:
            mock_config.host = "127.0.0.1"
            mock_config.port = 8080
            host, port = _get_host_port(args)

        self.assertEqual(host, "0.0.0.0")
        self.assertEqual(port, 9000)

    def test_get_host_port_falls_back_to_defaults(self):
        """Test _get_host_port falls back to config defaults."""
        args = argparse.Namespace(host=None, port=None, config=None)

        with patch("anthropic_proxy.cli.load_config_file", return_value={}):
            with patch("anthropic_proxy.cli.config") as mock_config:
                mock_config.host = "127.0.0.1"
                mock_config.port = 8080
                host, port = _get_host_port(args)

        self.assertEqual(host, "127.0.0.1")
        self.assertEqual(port, 8080)

    def test_get_host_port_args_override_config(self):
        """Test _get_host_port args override config file values."""
        config_path = Path(self.temp_dir) / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump({"host": "0.0.0.0", "port": 9000}, f)

        args = argparse.Namespace(host="localhost", port=7000, config=config_path)

        with patch("anthropic_proxy.cli.config") as mock_config:
            mock_config.host = "127.0.0.1"
            mock_config.port = 8080
            host, port = _get_host_port(args)

        self.assertEqual(host, "localhost")
        self.assertEqual(port, 7000)


class TestBuildParser(unittest.TestCase):
    """Test cases for build_parser function."""

    def test_build_parser_creates_parser(self):
        """Test build_parser returns an ArgumentParser."""
        parser = build_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_build_parser_has_global_options(self):
        """Test build_parser has expected global options."""
        parser = build_parser()

        # Test that --models option exists and parses correctly
        args = parser.parse_args(["--models", "/path/to/models.yaml"])
        self.assertEqual(args.models, Path("/path/to/models.yaml"))

        # Test that --config option exists and parses correctly
        args = parser.parse_args(["--config", "/path/to/config.json"])
        self.assertEqual(args.config, Path("/path/to/config.json"))

        # Test that --print-config option exists
        args = parser.parse_args(["--print-config"])
        self.assertTrue(args.print_config)

    def test_build_parser_has_init_subcommand(self):
        """Test build_parser has init subcommand with --force option."""
        parser = build_parser()

        # Test init subcommand
        args = parser.parse_args(["init"])
        self.assertEqual(args.command, "init")
        self.assertFalse(args.force)

        # Test init --force
        args = parser.parse_args(["init", "--force"])
        self.assertEqual(args.command, "init")
        self.assertTrue(args.force)

    def test_build_parser_has_start_subcommand(self):
        """Test build_parser has start subcommand with options."""
        parser = build_parser()
        args = parser.parse_args(["start", "--host", "0.0.0.0", "--port", "9000"])

        self.assertEqual(args.command, "start")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)

    def test_build_parser_has_stop_subcommand(self):
        """Test build_parser has stop subcommand."""
        parser = build_parser()
        args = parser.parse_args(["stop"])

        self.assertEqual(args.command, "stop")

    def test_build_parser_has_restart_subcommand(self):
        """Test build_parser has restart subcommand with options."""
        parser = build_parser()
        args = parser.parse_args(["restart", "--port", "9000"])

        self.assertEqual(args.command, "restart")
        self.assertEqual(args.port, 9000)

    def test_build_parser_has_status_subcommand(self):
        """Test build_parser has status subcommand."""
        parser = build_parser()
        args = parser.parse_args(["status"])

        self.assertEqual(args.command, "status")


class TestCmdStart(unittest.TestCase):
    """Test cases for cmd_start function."""

    @patch("anthropic_proxy.cli.daemon.start_daemon")
    @patch("anthropic_proxy.cli.daemon.get_daemon_status")
    @patch("anthropic_proxy.cli.initialize_config")
    def test_cmd_start_starts_daemon_when_not_running(
        self, mock_init, mock_status, mock_start
    ):
        """Test cmd_start starts daemon when server is not running."""
        mock_status.return_value = None
        mock_start.return_value = 12345

        args = argparse.Namespace(
            host="localhost", port=8080, models=None, config=None
        )

        with patch("anthropic_proxy.cli._get_host_port", return_value=("localhost", 8080)):
            with patch("builtins.print"):
                cmd_start(args)

        mock_start.assert_called_once()

    @patch("anthropic_proxy.cli.daemon.start_daemon")
    @patch("anthropic_proxy.cli.daemon.get_daemon_status")
    @patch("anthropic_proxy.cli.initialize_config")
    def test_cmd_start_exits_when_already_running(
        self, mock_init, mock_status, mock_start
    ):
        """Test cmd_start exits with error when server is already running."""
        mock_status.return_value = ProcessInfo(
            pid=12345, port=8080, uptime="1h", command="python"
        )

        args = argparse.Namespace(
            host="localhost", port=8080, models=None, config=None
        )

        with patch("anthropic_proxy.cli._get_host_port", return_value=("localhost", 8080)):
            with self.assertRaises(SystemExit) as cm:
                cmd_start(args)

        self.assertEqual(cm.exception.code, 1)
        mock_start.assert_not_called()

    @patch("anthropic_proxy.cli.daemon.start_daemon")
    @patch("anthropic_proxy.cli.daemon.get_daemon_status")
    @patch("anthropic_proxy.cli.initialize_config")
    def test_cmd_start_handles_runtime_error(
        self, mock_init, mock_status, mock_start
    ):
        """Test cmd_start handles RuntimeError from start_daemon."""
        mock_status.return_value = None
        mock_start.side_effect = RuntimeError("Failed to start")

        args = argparse.Namespace(
            host="localhost", port=8080, models=None, config=None
        )

        with patch("anthropic_proxy.cli._get_host_port", return_value=("localhost", 8080)):
            with self.assertRaises(SystemExit) as cm:
                cmd_start(args)

        self.assertEqual(cm.exception.code, 1)


class TestCmdStop(unittest.TestCase):
    """Test cases for cmd_stop function."""

    @patch("anthropic_proxy.cli.daemon.stop_daemon")
    @patch("anthropic_proxy.cli.daemon.get_daemon_status")
    def test_cmd_stop_stops_running_daemon(self, mock_status, mock_stop):
        """Test cmd_stop stops the daemon when running."""
        mock_status.return_value = ProcessInfo(
            pid=12345, port=8080, uptime="1h", command="python"
        )
        mock_stop.return_value = True

        args = argparse.Namespace()

        with patch("builtins.print"):
            cmd_stop(args)

        mock_stop.assert_called_once()

    @patch("anthropic_proxy.cli.daemon.stop_daemon")
    @patch("anthropic_proxy.cli.daemon.get_daemon_status")
    def test_cmd_stop_exits_when_not_running(self, mock_status, mock_stop):
        """Test cmd_stop exits gracefully when daemon is not running."""
        mock_status.return_value = None

        args = argparse.Namespace()

        with self.assertRaises(SystemExit) as cm:
            with patch("builtins.print"):
                cmd_stop(args)

        self.assertEqual(cm.exception.code, 0)
        mock_stop.assert_not_called()


class TestCmdRestart(unittest.TestCase):
    """Test cases for cmd_restart function."""

    @patch("anthropic_proxy.cli.cmd_start")
    @patch("anthropic_proxy.cli.daemon.stop_daemon")
    @patch("anthropic_proxy.cli.daemon.get_daemon_status")
    def test_cmd_restart_stops_then_starts(self, mock_status, mock_stop, mock_start):
        """Test cmd_restart stops then starts the daemon."""
        mock_status.return_value = ProcessInfo(
            pid=12345, port=8080, uptime="1h", command="python"
        )
        mock_stop.return_value = True

        args = argparse.Namespace(
            host="localhost", port=8080, models=None, config=None
        )

        with patch("builtins.print"):
            cmd_restart(args)

        mock_stop.assert_called_once()
        mock_start.assert_called_once_with(args)

    @patch("anthropic_proxy.cli.cmd_start")
    @patch("anthropic_proxy.cli.daemon.get_daemon_status")
    def test_cmd_restart_starts_when_not_running(self, mock_status, mock_start):
        """Test cmd_restart starts daemon when it was not running."""
        mock_status.return_value = None

        args = argparse.Namespace(
            host="localhost", port=8080, models=None, config=None
        )

        with patch("builtins.print"):
            cmd_restart(args)

        mock_start.assert_called_once_with(args)


class TestCmdStatus(unittest.TestCase):
    """Test cases for cmd_status function."""

    @patch("anthropic_proxy.cli.daemon.get_daemon_status")
    def test_cmd_status_prints_running_status(self, mock_status):
        """Test cmd_status prints running status when daemon is up."""
        mock_status.return_value = ProcessInfo(
            pid=12345, port=8080, uptime="1h 30m", command="python -m anthropic_proxy"
        )

        args = argparse.Namespace()

        # When running, cmd_status doesn't exit - just prints and returns
        with patch("builtins.print"):
            cmd_status(args)

        # Should not raise SystemExit
        mock_status.assert_called_once()

    @patch("anthropic_proxy.cli.daemon.get_daemon_status")
    def test_cmd_status_prints_stopped_status(self, mock_status):
        """Test cmd_status prints stopped status when daemon is down."""
        mock_status.return_value = None

        args = argparse.Namespace()

        with self.assertRaises(SystemExit) as cm:
            with patch("builtins.print"):
                cmd_status(args)

        self.assertEqual(cm.exception.code, 0)


class TestCmdInit(unittest.TestCase):
    """Test cases for cmd_init function."""

    def _make_args(self, force=False):
        """Create args namespace for cmd_init."""
        args = argparse.Namespace()
        args.force = force
        return args

    @patch("anthropic_proxy.config_manager.ensure_log_dir")
    @patch("anthropic_proxy.config_manager.create_default_config_file")
    @patch("anthropic_proxy.config_manager.create_default_models_file")
    @patch("anthropic_proxy.cli.DEFAULT_MODELS_FILE")
    @patch("anthropic_proxy.cli.DEFAULT_CONFIG_FILE")
    def test_cmd_init_creates_files_when_not_exist(
        self, mock_config_file, mock_models_file, mock_create_models, mock_create_config, mock_log_dir
    ):
        """Test cmd_init creates files when they don't exist."""
        mock_models_file.exists.return_value = False
        mock_config_file.exists.return_value = False

        with patch("builtins.print"):
            cmd_init(self._make_args(force=False))

        mock_create_models.assert_called_once_with(force=False)
        mock_create_config.assert_called_once_with(force=False)

    @patch("anthropic_proxy.config_manager.ensure_log_dir")
    @patch("anthropic_proxy.config_manager.create_default_config_file")
    @patch("anthropic_proxy.config_manager.create_default_models_file")
    @patch("anthropic_proxy.cli.DEFAULT_MODELS_FILE")
    @patch("anthropic_proxy.cli.DEFAULT_CONFIG_FILE")
    def test_cmd_init_skips_when_files_exist(
        self, mock_config_file, mock_models_file, mock_create_models, mock_create_config, mock_log_dir
    ):
        """Test cmd_init skips creation when files exist and no force."""
        mock_models_file.exists.return_value = True
        mock_config_file.exists.return_value = True

        with patch("builtins.print"):
            cmd_init(self._make_args(force=False))

        # Neither file should be created when both exist
        mock_create_models.assert_not_called()
        mock_create_config.assert_not_called()

    @patch("anthropic_proxy.config_manager.ensure_log_dir")
    @patch("anthropic_proxy.config_manager.create_default_config_file")
    @patch("anthropic_proxy.config_manager.create_default_models_file")
    @patch("anthropic_proxy.cli.DEFAULT_MODELS_FILE")
    @patch("anthropic_proxy.cli.DEFAULT_CONFIG_FILE")
    def test_cmd_init_force_only_resets_config(
        self, mock_config_file, mock_models_file, mock_create_models, mock_create_config, mock_log_dir
    ):
        """Test cmd_init --force only resets config.json, never models.yaml."""
        mock_models_file.exists.return_value = True
        mock_config_file.exists.return_value = True

        with patch("builtins.print"):
            cmd_init(self._make_args(force=True))

        # Models file is NEVER overwritten (even with force)
        mock_create_models.assert_not_called()
        # Config file is reset with force
        mock_create_config.assert_called_once_with(force=True)


class TestPrintConfig(unittest.TestCase):
    """Test cases for print_config function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_print_config_handles_nonexistent_files(self):
        """Test print_config handles missing config files gracefully."""
        models_path = Path(self.temp_dir) / "nonexistent_models.yaml"
        config_path = Path(self.temp_dir) / "nonexistent_config.json"

        with patch("builtins.print"):
            print_config(models_path, config_path, show_api_keys=False)

        # Should not raise exception


class TestParseArgs(unittest.TestCase):
    """Test cases for parse_args function."""

    def test_parse_args_returns_namespace(self):
        """Test parse_args returns argparse.Namespace."""
        with patch("sys.argv", ["anthropic-proxy", "init"]):
            args = parse_args()
            self.assertIsInstance(args, argparse.Namespace)

    def test_parse_args_parses_init_subcommand(self):
        """Test parse_args correctly parses init subcommand."""
        with patch("sys.argv", ["anthropic-proxy", "init"]):
            args = parse_args()
            self.assertEqual(args.command, "init")
            self.assertFalse(args.force)

        with patch("sys.argv", ["anthropic-proxy", "init", "--force"]):
            args = parse_args()
            self.assertEqual(args.command, "init")
            self.assertTrue(args.force)

    def test_parse_args_parses_start_command(self):
        """Test parse_args correctly parses start subcommand."""
        with patch("sys.argv", ["anthropic-proxy", "start", "--port", "9000"]):
            args = parse_args()
            self.assertEqual(args.command, "start")
            self.assertEqual(args.port, 9000)


if __name__ == "__main__":
    unittest.main()
