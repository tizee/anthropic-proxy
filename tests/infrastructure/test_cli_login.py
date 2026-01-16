import unittest
from unittest.mock import patch, MagicMock
import argparse
import sys
import os

# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.cli import cmd_login, parse_args

class TestCliLogin(unittest.TestCase):
    def test_login_codex_flag(self):
        """Test login command with --codex flag calls codex_auth.login()."""
        with patch("sys.argv", ["anthropic-proxy", "login", "--codex"]):
            args = parse_args()
            self.assertTrue(args.codex)
            
            with patch("anthropic_proxy.codex.codex_auth.login") as mock_login:
                cmd_login(args)
                mock_login.assert_called_once()

    def test_login_no_flag(self):
        """Test login command without flags prints help/error."""
        with patch("sys.argv", ["anthropic-proxy", "login"]):
            args = parse_args()
            self.assertFalse(args.codex)
            
            with patch("builtins.print") as mock_print:
                with self.assertRaises(SystemExit) as cm:
                    cmd_login(args)
                
                self.assertEqual(cm.exception.code, 1)
                # Verify it printed something about specifying a provider
                mock_print.assert_any_call("Available providers:")

if __name__ == "__main__":
    unittest.main()
