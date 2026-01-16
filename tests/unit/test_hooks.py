import os
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from anthropic_proxy.hook import HookManager, request_hook_spec, response_hook_spec


class TestHookManager(unittest.TestCase):
    def setUp(self):
        """Set up a new HookManager instance for each test."""
        self.hook_manager = HookManager()

    def test_initialization(self):
        """Verify that the HookManager initializes with empty hook lists."""
        self.assertEqual(self.hook_manager.request_hooks, [])
        self.assertEqual(self.hook_manager.response_hooks, [])

    def test_register_hooks_with_both_hooks(self):
        """Test registering a module that contains both request and response hooks."""
        mock_module = Mock()
        mock_module.request_hook = lambda x: x
        mock_module.response_hook = lambda x: x
        setattr(mock_module, request_hook_spec, mock_module.request_hook)
        setattr(mock_module, response_hook_spec, mock_module.response_hook)

        self.hook_manager.register_hooks(mock_module)

        self.assertEqual(len(self.hook_manager.request_hooks), 1)
        self.assertEqual(len(self.hook_manager.response_hooks), 1)
        self.assertIn(mock_module.request_hook, self.hook_manager.request_hooks)
        self.assertIn(mock_module.response_hook, self.hook_manager.response_hooks)

    def test_register_hooks_with_only_request_hook(self):
        """Test registering a module with only a request hook."""
        mock_module = Mock(spec=[request_hook_spec])
        mock_module.request_hook = lambda x: x

        self.hook_manager.register_hooks(mock_module)

        self.assertEqual(len(self.hook_manager.request_hooks), 1)
        self.assertEqual(len(self.hook_manager.response_hooks), 0)

    def test_register_hooks_with_no_hooks(self):
        """Test registering a module with no hooks."""
        mock_module = Mock(spec=[])
        self.hook_manager.register_hooks(mock_module)
        self.assertEqual(len(self.hook_manager.request_hooks), 0)
        self.assertEqual(len(self.hook_manager.response_hooks), 0)

    @patch("anthropic_proxy.hook.importlib.import_module")
    @patch("anthropic_proxy.hook.pkgutil.iter_modules")
    def test_load_plugins_successfully(self, mock_iter_modules, mock_import_module):
        """Verify that load_plugins correctly discovers and registers hooks."""
        mock_package = Mock()
        mock_package.__name__ = "fake_plugins"
        mock_package.__path__ = ["/fake/path"]
        mock_iter_modules.return_value = [(None, "test_plugin", False)]

        mock_plugin_module = Mock(spec=[request_hook_spec])
        mock_plugin_module.request_hook = lambda x: x
        mock_import_module.return_value = mock_plugin_module

        self.hook_manager.load_plugins(mock_package)

        mock_iter_modules.assert_called_once_with(["/fake/path"])
        mock_import_module.assert_called_once_with("fake_plugins.test_plugin")
        self.assertEqual(len(self.hook_manager.request_hooks), 1)

    @patch("anthropic_proxy.hook.logger")
    def test_load_plugins_package_without_path(self, mock_logger):
        """Test that load_plugins handles packages without a __path__ attribute."""
        mock_package = Mock(spec=["__name__"])
        mock_package.__name__ = "no_path_package"

        self.hook_manager.load_plugins(mock_package)

        mock_logger.warning.assert_called_once_with(
            "Package no_path_package does not have a __path__."
        )

    @patch("anthropic_proxy.hook.importlib.import_module")
    @patch("anthropic_proxy.hook.pkgutil.iter_modules")
    def test_load_plugins_import_error(self, mock_iter_modules, mock_import_module):
        """Test that load_plugins handles import errors gracefully."""
        mock_import_module.side_effect = ImportError("Test import failed")
        mock_package = Mock()
        mock_package.__name__ = "fake_plugins"
        mock_package.__path__ = ["/fake/path"]
        mock_iter_modules.return_value = [
            (None, "failing_plugin", False),
        ]

        # The try-except block in load_plugins should catch the ImportError
        # and the test should pass without crashing.
        self.hook_manager.load_plugins(mock_package)

        # Verify that no hooks were loaded due to the import error.
        self.assertEqual(len(self.hook_manager.request_hooks), 0)
        self.assertEqual(len(self.hook_manager.response_hooks), 0)

    def test_trigger_request_hooks(self):
        """Test that request hooks are triggered and modify the payload."""
        initial_payload = {"data": 0}
        expected_payload = {"data": 1}

        def request_hook(payload):
            payload["data"] += 1
            return payload

        self.hook_manager.request_hooks.append(request_hook)
        processed_payload = self.hook_manager.trigger_request_hooks(initial_payload)

        self.assertEqual(processed_payload, expected_payload)

    def test_trigger_response_hooks(self):
        """Test that response hooks are triggered and modify the payload."""
        initial_payload = {"status": "processing"}
        expected_payload = {"status": "completed"}

        def response_hook(payload):
            payload["status"] = "completed"
            return payload

        self.hook_manager.response_hooks.append(response_hook)
        processed_payload = self.hook_manager.trigger_response_hooks(initial_payload)

        self.assertEqual(processed_payload, expected_payload)

    def test_trigger_hooks_chaining(self):
        """Test that multiple hooks are chained correctly."""
        initial_payload = {"value": "A"}

        def hook1(payload):
            payload["value"] += "B"
            return payload

        def hook2(payload):
            payload["value"] += "C"
            return payload

        self.hook_manager.request_hooks.extend([hook1, hook2])
        processed_payload = self.hook_manager.trigger_request_hooks(initial_payload)

        self.assertEqual(processed_payload["value"], "ABC")

    def test_trigger_with_no_hooks(self):
        """Test that the payload is unchanged when no hooks are registered."""
        initial_payload = {"data": "unchanged"}
        processed_payload = self.hook_manager.trigger_request_hooks(initial_payload)
        self.assertEqual(processed_payload, initial_payload)


class TestFilterToolsPlugin(unittest.TestCase):
    def setUp(self):
        """Set up test environment with real filter tools plugin."""
        self.hook_manager = HookManager()
        from anthropic_proxy.plugins.filter_tools import request_hook, response_hook

        self.request_hook = request_hook
        self.response_hook = response_hook

    def test_request_hook_filters_claude_format_tools(self):
        """Test filtering tools in Claude format."""
        initial_payload = {
            "tools": [
                {"name": "WebSearch"},
                {"name": "SomeOtherTool"},
            ]
        }
        expected_payload = {"tools": [{"name": "SomeOtherTool"}]}

        result = self.request_hook(initial_payload)
        self.assertEqual(result, expected_payload)

    def test_request_hook_filters_openai_format_tools(self):
        """Test filtering tools in OpenAI format."""
        initial_payload = {
            "tools": [
                {"function": {"name": "NotebookEdit"}},
                {"function": {"name": "NotebookRead"}},
                {"function": {"name": "ValidTool"}},
            ]
        }
        expected_payload = {"tools": [{"function": {"name": "ValidTool"}}]}

        result = self.request_hook(initial_payload)
        self.assertEqual(result, expected_payload)

    def test_request_hook_preserves_unknown_format_tools(self):
        """Test that tools with unknown format are preserved."""
        initial_payload = {
            "tools": [
                {"unknown_key": "UnknownTool"},
            ]
        }

        result = self.request_hook(initial_payload)
        self.assertEqual(result, initial_payload)

    def test_request_hook_returns_unchanged_when_no_tools(self):
        """Test payload remains unchanged when no tools key exists."""
        initial_payload = {"prompt": "Hello, world!"}
        result = self.request_hook(initial_payload)
        self.assertEqual(result, initial_payload)

    def test_response_hook_adds_flag(self):
        """Test response hook adds 'hook_applied' flag."""
        initial_payload = {"key": "value"}
        expected_payload = {"key": "value", "hook_applied": True}

        result = self.response_hook(initial_payload)
        self.assertEqual(result, expected_payload)


if __name__ == "__main__":
    unittest.main()
