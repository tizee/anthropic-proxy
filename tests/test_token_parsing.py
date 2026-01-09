"""Unit tests for token parsing functionality."""

import unittest

from anthropic_proxy.config import parse_token_value


class TestTokenParsing(unittest.TestCase):
    """Test cases for token value parsing."""

    def test_uppercase_k_notation(self):
        """Test parsing token values with uppercase K notation."""
        self.assertEqual(parse_token_value("16K"), 16000)
        self.assertEqual(parse_token_value("66K"), 66000)
        self.assertEqual(parse_token_value("8K"), 8000)
        self.assertEqual(parse_token_value("256K"), 256000)
        self.assertEqual(parse_token_value("1050K"), 1050000)
        self.assertEqual(parse_token_value("200K"), 200000)

    def test_lowercase_k_notation(self):
        """Test parsing token values with lowercase k notation (backward compatibility)."""
        self.assertEqual(parse_token_value("16k"), 16000)
        self.assertEqual(parse_token_value("66k"), 66000)
        self.assertEqual(parse_token_value("8k"), 8000)

    def test_integer_values(self):
        """Test parsing integer token values."""
        self.assertEqual(parse_token_value(16000), 16000)
        self.assertEqual(parse_token_value(8000), 8000)
        self.assertEqual(parse_token_value(0), 0)
        self.assertEqual(parse_token_value(1000000), 1000000)

    def test_string_integer_values(self):
        """Test parsing string representations of integers."""
        self.assertEqual(parse_token_value("16000"), 16000)
        self.assertEqual(parse_token_value("8000"), 8000)
        self.assertEqual(parse_token_value("0"), 0)
        self.assertEqual(parse_token_value("1000000"), 1000000)

    def test_decimal_k_notation(self):
        """Test parsing decimal values with K notation."""
        self.assertEqual(parse_token_value("0.5K"), 500)
        self.assertEqual(parse_token_value("1.5K"), 1500)
        self.assertEqual(parse_token_value("2.75K"), 2750)

    def test_invalid_values(self):
        """Test parsing invalid token values."""
        self.assertEqual(parse_token_value(""), None)
        self.assertEqual(parse_token_value("abc"), None)
        self.assertEqual(parse_token_value("16x"), None)
        self.assertEqual(parse_token_value(""), None)
        self.assertEqual(parse_token_value(None), None)

    def test_custom_default_value(self):
        """Test that custom default values are returned for invalid inputs."""
        self.assertEqual(parse_token_value("abc", default_value=1000), 1000)
        self.assertEqual(parse_token_value(None, default_value=200000), 200000)
        self.assertEqual(parse_token_value("", default_value=5000), 5000)


if __name__ == '__main__':
    unittest.main()
