import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from anthropic_proxy.hook import hook_manager, load_all_plugins


class TestHookSystem(unittest.TestCase):
    def setUp(self):
        """Set up the test environment by loading all plugins."""
        # Reset hooks before each test to ensure isolation
        hook_manager.request_hooks = []
        hook_manager.response_hooks = []
        load_all_plugins()

    def test_request_hook_filters_tools(self):
        """Verify that the request hook correctly filters out specified tools."""
        # Initial payload with a list of tools
        initial_payload = {
            "tools": [
                {"name": "WebSearch"},
                {"name": "WebFetch"},
                {"name": "SomeOtherTool"},
            ]
        }

        # Expected payload after the hook is applied
        expected_payload = {"tools": [{"name": "SomeOtherTool"}]}

        # Trigger the request hooks
        processed_payload = hook_manager.trigger_request_hooks(initial_payload)

        # Assert that the tools were filtered as expected
        self.assertEqual(processed_payload, expected_payload)
        self.assertNotIn({"name": "WebSearch"}, processed_payload["tools"])
        self.assertNotIn({"name": "WebFetch"}, processed_payload["tools"])
        self.assertIn({"name": "SomeOtherTool"}, processed_payload["tools"])

    def test_response_hook_modifies_payload(self):
        """Verify that the response hook correctly modifies the payload."""
        # Initial payload for the response
        initial_payload = {"some_data": "initial_value"}

        # Expected payload after the hook is applied
        expected_payload = {"some_data": "initial_value", "hook_applied": True}

        # Trigger the response hooks
        processed_payload = hook_manager.trigger_response_hooks(initial_payload)

        # Assert that the payload was modified as expected
        self.assertEqual(processed_payload, expected_payload)
        self.assertTrue(processed_payload.get("hook_applied"))

    def test_hooks_are_loaded(self):
        """Verify that the hooks are loaded into the hook_manager."""
        self.assertGreater(
            len(hook_manager.request_hooks), 0, "Request hooks should be loaded"
        )
        self.assertGreater(
            len(hook_manager.response_hooks), 0, "Response hooks should be loaded"
        )


if __name__ == "__main__":
    unittest.main()
