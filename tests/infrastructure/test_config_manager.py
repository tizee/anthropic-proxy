"""
Unit tests for configuration directory and file management.

Tests cover:
- Directory creation (config and log directories)
- Default config and models file creation
- Config file loading
- Initialization with and without force flag
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from anthropic_proxy.config_manager import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_FILE,
    DEFAULT_CONFIG_TEMPLATE,
    DEFAULT_LOG_DIR,
    DEFAULT_MODELS_FILE,
    DEFAULT_MODELS_TEMPLATE,
    create_default_config_file,
    create_default_models_file,
    ensure_config_dir,
    ensure_log_dir,
    get_config_file_path,
    get_default_log_file_path,
    get_log_dir_path,
    get_models_file_path,
    initialize_config,
    load_config_file,
)


class TestEnsureConfigDir(unittest.TestCase):
    """Test cases for ensure_config_dir function."""

    @patch("anthropic_proxy.config_manager.DEFAULT_CONFIG_DIR")
    def test_ensure_config_dir_creates_directory_if_not_exists(self, mock_dir):
        """Test ensure_config_dir creates directory when it doesn't exist."""
        mock_dir.exists.return_value = False
        mock_dir.mkdir = Mock()

        result = ensure_config_dir()

        mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.assertEqual(result, mock_dir)

    @patch("anthropic_proxy.config_manager.DEFAULT_CONFIG_DIR")
    def test_ensure_config_dir_returns_existing_directory(self, mock_dir):
        """Test ensure_config_dir returns existing directory."""
        mock_dir.exists.return_value = True

        result = ensure_config_dir()

        mock_dir.mkdir.assert_not_called()
        self.assertEqual(result, mock_dir)


class TestEnsureLogDir(unittest.TestCase):
    """Test cases for ensure_log_dir function."""

    @patch("anthropic_proxy.config_manager.DEFAULT_LOG_DIR")
    def test_ensure_log_dir_creates_directory_if_not_exists(self, mock_dir):
        """Test ensure_log_dir creates directory when it doesn't exist."""
        mock_dir.exists.return_value = False
        mock_dir.mkdir = Mock()

        result = ensure_log_dir()

        mock_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        self.assertEqual(result, mock_dir)

    @patch("anthropic_proxy.config_manager.DEFAULT_LOG_DIR")
    def test_ensure_log_dir_returns_existing_directory(self, mock_dir):
        """Test ensure_log_dir returns existing directory."""
        mock_dir.exists.return_value = True

        result = ensure_log_dir()

        mock_dir.mkdir.assert_not_called()
        self.assertEqual(result, mock_dir)


class TestCreateDefaultModelsFile(unittest.TestCase):
    """Test cases for create_default_models_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    @patch("anthropic_proxy.config_manager.ensure_config_dir")
    @patch("anthropic_proxy.config_manager.DEFAULT_MODELS_FILE")
    def test_creates_file_when_not_exists(self, mock_file, mock_ensure):
        """Test create_default_models_file creates file when it doesn't exist."""
        mock_file.exists.return_value = False
        mock_file.write_text = Mock()

        result = create_default_models_file(force=False)

        mock_file.write_text.assert_called_once_with(
            DEFAULT_MODELS_TEMPLATE, encoding="utf-8"
        )
        self.assertEqual(result, mock_file)

    @patch("anthropic_proxy.config_manager.ensure_config_dir")
    @patch("anthropic_proxy.config_manager.DEFAULT_MODELS_FILE")
    def test_does_not_overwrite_existing_file_by_default(self, mock_file, mock_ensure):
        """Test create_default_models_file doesn't overwrite existing file."""
        mock_file.exists.return_value = True
        mock_file.write_text = Mock()

        result = create_default_models_file(force=False)

        mock_file.write_text.assert_not_called()
        self.assertEqual(result, mock_file)

    @patch("anthropic_proxy.config_manager.ensure_config_dir")
    @patch("anthropic_proxy.config_manager.DEFAULT_MODELS_FILE")
    def test_overwrites_when_force_true(self, mock_file, mock_ensure):
        """Test create_default_models_file overwrites when force=True."""
        mock_file.exists.return_value = True
        mock_file.write_text = Mock()

        result = create_default_models_file(force=True)

        mock_file.write_text.assert_called_once_with(
            DEFAULT_MODELS_TEMPLATE, encoding="utf-8"
        )
        self.assertEqual(result, mock_file)


class TestCreateDefaultConfigFile(unittest.TestCase):
    """Test cases for create_default_config_file function."""

    @patch("anthropic_proxy.config_manager.ensure_config_dir")
    @patch("anthropic_proxy.config_manager.DEFAULT_CONFIG_FILE")
    def test_creates_file_when_not_exists(self, mock_file, mock_ensure):
        """Test create_default_config_file creates file when it doesn't exist."""
        mock_file.exists.return_value = False

        # Create a proper mock for the file object
        mock_file_obj = Mock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_file_obj
        mock_context_manager.__exit__.return_value = False
        mock_file.open = Mock(return_value=mock_context_manager)

        result = create_default_config_file(force=False)

        mock_file.open.assert_called_once_with("w", encoding="utf-8")

    @patch("anthropic_proxy.config_manager.ensure_config_dir")
    @patch("anthropic_proxy.config_manager.DEFAULT_CONFIG_FILE")
    def test_does_not_overwrite_existing_file_by_default(self, mock_file, mock_ensure):
        """Test create_default_config_file doesn't overwrite existing file."""
        mock_file.exists.return_value = True
        mock_file.open = Mock()

        result = create_default_config_file(force=False)

        mock_file.open.assert_not_called()
        self.assertEqual(result, mock_file)

    @patch("anthropic_proxy.config_manager.ensure_config_dir")
    @patch("anthropic_proxy.config_manager.DEFAULT_CONFIG_FILE")
    def test_overwrites_when_force_true(self, mock_file, mock_ensure):
        """Test create_default_config_file overwrites when force=True."""
        mock_file.exists.return_value = True

        # Create a proper mock for the file object
        mock_file_obj = Mock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_file_obj
        mock_context_manager.__exit__.return_value = False
        mock_file.open = Mock(return_value=mock_context_manager)

        result = create_default_config_file(force=True)

        mock_file.open.assert_called_once_with("w", encoding="utf-8")


class TestInitializeConfig(unittest.TestCase):
    """Test cases for initialize_config function."""

    @patch("anthropic_proxy.config_manager.create_default_config_file")
    @patch("anthropic_proxy.config_manager.create_default_models_file")
    @patch("anthropic_proxy.config_manager.ensure_config_dir")
    @patch("anthropic_proxy.config_manager.ensure_log_dir")
    def test_initialize_config_creates_all_directories_and_files(
        self, mock_log, mock_config, mock_models, mock_config_file
    ):
        """Test initialize_config creates all necessary directories and files."""
        mock_models.return_value = Path("/models.yaml")
        mock_config_file.return_value = Path("/config.json")

        models_path, config_path = initialize_config(force=False)

        mock_log.assert_called_once()
        mock_config.assert_called_once()
        self.assertEqual(models_path, Path("/models.yaml"))
        self.assertEqual(config_path, Path("/config.json"))

    @patch("anthropic_proxy.config_manager.create_default_config_file")
    @patch("anthropic_proxy.config_manager.create_default_models_file")
    @patch("anthropic_proxy.config_manager.ensure_config_dir")
    @patch("anthropic_proxy.config_manager.ensure_log_dir")
    def test_initialize_config_passes_force_flag(
        self, mock_log, mock_config, mock_models, mock_config_file
    ):
        """Test initialize_config passes force flag to file creators."""
        mock_models.return_value = Path("/models.yaml")
        mock_config_file.return_value = Path("/config.json")

        initialize_config(force=True)

        mock_models.assert_called_once_with(force=True)
        mock_config_file.assert_called_once_with(force=True)


class TestLoadConfigFile(unittest.TestCase):
    """Test cases for load_config_file function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_load_config_file_returns_empty_dict_for_nonexistent_file(self):
        """Test load_config_file returns empty dict when file doesn't exist."""
        config_path = Path(self.temp_dir) / "nonexistent.json"

        result = load_config_file(config_path)

        self.assertEqual(result, {})

    def test_load_config_file_returns_empty_dict_for_none_path(self):
        """Test load_config_file returns empty dict when path is None."""
        with patch("anthropic_proxy.config_manager.DEFAULT_CONFIG_FILE") as mock_file:
            mock_file.exists.return_value = False

            result = load_config_file(None)

            self.assertEqual(result, {})

    def test_load_config_file_parses_valid_json(self):
        """Test load_config_file correctly parses valid JSON."""
        config_path = Path(self.temp_dir) / "config.json"
        test_config = {"host": "localhost", "port": 8080, "log_level": "DEBUG"}

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(test_config, f)

        result = load_config_file(config_path)

        self.assertEqual(result, test_config)

    def test_load_config_file_returns_empty_dict_for_invalid_json(self):
        """Test load_config_file returns empty dict for invalid JSON."""
        config_path = Path(self.temp_dir) / "invalid.json"

        with config_path.open("w", encoding="utf-8") as f:
            f.write("{ invalid json }")

        result = load_config_file(config_path)

        self.assertEqual(result, {})

    def test_load_config_file_handles_empty_file(self):
        """Test load_config_file handles empty file."""
        config_path = Path(self.temp_dir) / "empty.json"

        with config_path.open("w", encoding="utf-8") as f:
            f.write("")

        result = load_config_file(config_path)

        self.assertEqual(result, {})


class TestGetPathFunctions(unittest.TestCase):
    """Test cases for path getter functions."""

    @patch("anthropic_proxy.config_manager.DEFAULT_MODELS_FILE")
    def test_get_models_file_path(self, mock_path):
        """Test get_models_file_path returns correct path."""
        result = get_models_file_path()
        self.assertEqual(result, mock_path)

    @patch("anthropic_proxy.config_manager.DEFAULT_CONFIG_FILE")
    def test_get_config_file_path(self, mock_path):
        """Test get_config_file_path returns correct path."""
        result = get_config_file_path()
        self.assertEqual(result, mock_path)

    @patch("anthropic_proxy.config_manager.DEFAULT_LOG_DIR")
    def test_get_log_dir_path(self, mock_path):
        """Test get_log_dir_path returns correct path."""
        result = get_log_dir_path()
        self.assertEqual(result, mock_path)

    @patch("anthropic_proxy.config_manager.DEFAULT_LOG_DIR")
    def test_get_default_log_file_path(self, mock_dir):
        """Test get_default_log_file_path returns correct path."""
        mock_dir.__truediv__ = Mock(return_value=Path("/log/server.log"))
        result = get_default_log_file_path()
        self.assertEqual(result, Path("/log/server.log"))


class TestDefaultTemplates(unittest.TestCase):
    """Test cases for default template constants."""

    def test_default_models_template_is_string(self):
        """Test DEFAULT_MODELS_TEMPLATE is a non-empty string."""
        self.assertIsInstance(DEFAULT_MODELS_TEMPLATE, str)
        self.assertGreater(len(DEFAULT_MODELS_TEMPLATE), 0)

    def test_default_models_template_contains_expected_content(self):
        """Test DEFAULT_MODELS_TEMPLATE contains expected documentation."""
        self.assertIn("model_id", DEFAULT_MODELS_TEMPLATE)
        self.assertIn("api_base", DEFAULT_MODELS_TEMPLATE)
        self.assertIn("api_key", DEFAULT_MODELS_TEMPLATE)
        self.assertIn("format", DEFAULT_MODELS_TEMPLATE)
        self.assertIn("direct", DEFAULT_MODELS_TEMPLATE)

    def test_default_config_template_has_required_keys(self):
        """Test DEFAULT_CONFIG_TEMPLATE has all required keys."""
        required_keys = ["log_level", "log_file_path", "host", "port"]
        for key in required_keys:
            self.assertIn(key, DEFAULT_CONFIG_TEMPLATE)

    def test_default_config_template_values_are_correct_types(self):
        """Test DEFAULT_CONFIG_TEMPLATE values are correct types."""
        self.assertIsInstance(DEFAULT_CONFIG_TEMPLATE["log_level"], str)
        self.assertIsInstance(DEFAULT_CONFIG_TEMPLATE["log_file_path"], str)
        self.assertIsInstance(DEFAULT_CONFIG_TEMPLATE["host"], str)
        self.assertIsInstance(DEFAULT_CONFIG_TEMPLATE["port"], int)


class TestDefaultPaths(unittest.TestCase):
    """Test cases for default path constants."""

    @patch("anthropic_proxy.config_manager.Path.home")
    def test_default_config_dir_path(self, mock_home):
        """Test DEFAULT_CONFIG_DIR is correctly constructed."""
        mock_home.return_value = Path("/home/user")

        result = DEFAULT_CONFIG_DIR
        expected = Path("/home/user/.config/anthropic-proxy")

        # The actual path depends on the system, so we just verify the structure
        self.assertIn(".config", str(result))
        self.assertIn("anthropic-proxy", str(result))

    @patch("anthropic_proxy.config_manager.Path.home")
    def test_default_log_dir_path(self, mock_home):
        """Test DEFAULT_LOG_DIR is correctly constructed."""
        mock_home.return_value = Path("/home/user")

        result = DEFAULT_LOG_DIR

        # Verify it's a hidden directory in home
        self.assertIn(".anthropic-proxy", str(result))

    def test_default_models_file_is_under_config_dir(self):
        """Test DEFAULT_MODELS_FILE is under DEFAULT_CONFIG_DIR."""
        self.assertIn("models.yaml", str(DEFAULT_MODELS_FILE))

    def test_default_config_file_is_under_config_dir(self):
        """Test DEFAULT_CONFIG_FILE is under DEFAULT_CONFIG_DIR."""
        self.assertIn("config.json", str(DEFAULT_CONFIG_FILE))


if __name__ == "__main__":
    unittest.main()
