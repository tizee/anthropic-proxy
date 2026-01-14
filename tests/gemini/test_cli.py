import unittest
from unittest.mock import patch

from anthropic_proxy.cli import cmd_login, parse_args


class TestGeminiCLI(unittest.TestCase):
    def test_cli_login_gemini(self):
        with patch("sys.argv", ["anthropic-proxy", "login", "--gemini"]):
            args = parse_args()
            self.assertTrue(args.gemini)

            with patch("anthropic_proxy.gemini.gemini_auth.login") as mock_login:
                cmd_login(args)
                mock_login.assert_called_once()


if __name__ == "__main__":
    unittest.main()
