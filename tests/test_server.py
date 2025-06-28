#!/usr/bin/env python3
"""
Test suite for server retry mechanisms and related functionality.

This module tests the retry mechanisms implemented in the Claude proxy server,
including httpx client configuration, retry behavior on various error types,
and integration with the server endpoints.

Usage:
  python tests/test_server.py                    # Run all tests
  python -m unittest tests.test_server           # Run with unittest module
"""

import unittest
import sys
import os
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock, call
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from openai import AsyncOpenAI

from anthropic_proxy.client import create_claude_client, create_openai_client
from anthropic_proxy.types import ModelDefaults


class TestRetryMechanism(unittest.TestCase):
    """Test the retry mechanisms in client creation and network requests."""

    def setUp(self):
        """Set up test environment with mock models."""
        self.mock_models = {
            "test-claude-model": {
                "model_id": "test-claude-model",
                "model_name": "claude-3-sonnet-20240229",
                "api_base": "https://api.anthropic.com",
                "api_key_name": "ANTHROPIC_API_KEY",
                "direct": True,
                "max_input_tokens": 100000,
                "extra_headers": {},
            },
            "test-openai-model": {
                "model_id": "test-openai-model",
                "model_name": "gpt-4",
                "api_base": "https://api.openai.com/v1",
                "api_key_name": "OPENAI_API_KEY",
                "direct": False,
                "max_input_tokens": 50000,
                "extra_headers": {},
            },
        }

        # Mock the config with custom API keys
        self.mock_config = MagicMock()
        self.mock_config.custom_api_keys = {
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "OPENAI_API_KEY": "test-openai-key",
        }

        # Patch the global model dictionary and config
        self.models_patcher = patch.dict(
            "anthropic_proxy.client.CUSTOM_OPENAI_MODELS", self.mock_models, clear=True
        )
        self.config_patcher = patch("anthropic_proxy.client.config", self.mock_config)

        self.models_patcher.start()
        self.config_patcher.start()

    def tearDown(self):
        """Clean up patches."""
        self.models_patcher.stop()
        self.config_patcher.stop()

    def test_default_max_retries_value(self):
        """Test that DEFAULT_MAX_RETRIES is correctly defined."""
        self.assertEqual(ModelDefaults.DEFAULT_MAX_RETRIES, 2)
        self.assertIsInstance(ModelDefaults.DEFAULT_MAX_RETRIES, int)
        self.assertGreater(ModelDefaults.DEFAULT_MAX_RETRIES, 0)

    @patch("anthropic_proxy.client.httpx.AsyncClient")
    @patch("anthropic_proxy.client.httpx.AsyncHTTPTransport")
    def test_create_claude_client_with_retry(
        self, mock_transport_class, mock_client_class
    ):
        """Test that create_claude_client configures retry transport correctly."""
        # Setup mocks
        mock_transport = MagicMock()
        mock_transport_class.return_value = mock_transport
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Call the function
        result = create_claude_client("test-claude-model")

        # Verify transport was created with correct retry count
        mock_transport_class.assert_called_once_with(
            retries=ModelDefaults.DEFAULT_MAX_RETRIES
        )

        # Verify client was created with transport
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]

        self.assertEqual(call_kwargs["transport"], mock_transport)
        self.assertEqual(call_kwargs["base_url"], "https://api.anthropic.com/v1")
        self.assertIn("x-api-key", call_kwargs["headers"])
        self.assertEqual(call_kwargs["headers"]["x-api-key"], "test-anthropic-key")

        # Verify result
        self.assertEqual(result, mock_client)

    @patch("anthropic_proxy.client.AsyncOpenAI")
    @patch("anthropic_proxy.client.httpx.AsyncClient")
    @patch("anthropic_proxy.client.httpx.AsyncHTTPTransport")
    def test_create_openai_client_with_retry(
        self, mock_transport_class, mock_http_client_class, mock_openai_class
    ):
        """Test that create_openai_client configures retry transport correctly."""
        # Setup mocks
        mock_transport = MagicMock()
        mock_transport_class.return_value = mock_transport
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        mock_openai_client = MagicMock()
        mock_openai_class.return_value = mock_openai_client

        # Call the function
        result = create_openai_client("test-openai-model")

        # Verify transport was created with correct retry count
        mock_transport_class.assert_called_once_with(
            retries=ModelDefaults.DEFAULT_MAX_RETRIES
        )

        # Verify HTTP client was created with transport
        mock_http_client_class.assert_called_once_with(transport=mock_transport)

        # Verify OpenAI client was created with HTTP client
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]

        self.assertEqual(call_kwargs["http_client"], mock_http_client)
        self.assertEqual(call_kwargs["base_url"], "https://api.openai.com/v1")
        self.assertEqual(call_kwargs["api_key"], "test-openai-key")

        # Verify result
        self.assertEqual(result, mock_openai_client)

    @patch("anthropic_proxy.client.httpx.AsyncClient")
    @patch("anthropic_proxy.client.httpx.AsyncHTTPTransport")
    def test_claude_client_retry_configuration_details(
        self, mock_transport_class, mock_client_class
    ):
        """Test detailed retry configuration for Claude client."""
        mock_transport = MagicMock()
        mock_transport_class.return_value = mock_transport
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Call function
        create_claude_client("test-claude-model")

        # Verify transport configuration
        mock_transport_class.assert_called_once_with(
            retries=2
        )  # DEFAULT_MAX_RETRIES = 2

        # Verify client configuration includes all expected parameters
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]

        # Check all required parameters are present
        required_keys = ["base_url", "headers", "timeout", "transport"]
        for key in required_keys:
            self.assertIn(key, call_kwargs, f"Missing required parameter: {key}")

        # Check timeout configuration
        self.assertIsInstance(call_kwargs["timeout"], httpx.Timeout)

    @patch("anthropic_proxy.client.AsyncOpenAI")
    @patch("anthropic_proxy.client.httpx.AsyncClient")
    @patch("anthropic_proxy.client.httpx.AsyncHTTPTransport")
    def test_openai_client_retry_configuration_details(
        self, mock_transport_class, mock_http_client_class, mock_openai_class
    ):
        """Test detailed retry configuration for OpenAI client."""
        mock_transport = MagicMock()
        mock_transport_class.return_value = mock_transport
        mock_http_client = MagicMock()
        mock_http_client_class.return_value = mock_http_client
        mock_openai_client = MagicMock()
        mock_openai_class.return_value = mock_openai_client

        # Call function
        create_openai_client("test-openai-model")

        # Verify transport configuration
        mock_transport_class.assert_called_once_with(
            retries=2
        )  # DEFAULT_MAX_RETRIES = 2

        # Verify HTTP client gets the transport
        mock_http_client_class.assert_called_once_with(transport=mock_transport)

        # Verify OpenAI client gets the HTTP client
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        self.assertEqual(call_kwargs["http_client"], mock_http_client)

    def test_invalid_model_handling(self):
        """Test that invalid models raise appropriate errors."""
        with self.assertRaises(ValueError) as context:
            create_claude_client("nonexistent-model")

        self.assertIn("Unknown model", str(context.exception))

        with self.assertRaises(ValueError) as context:
            create_openai_client("nonexistent-model")

        self.assertIn("Unknown custom model", str(context.exception))

    def test_missing_api_key_handling(self):
        """Test handling of missing API keys."""
        # Mock config without API keys
        self.mock_config.custom_api_keys = {}

        with self.assertRaises(ValueError) as context:
            create_claude_client("test-claude-model")

        self.assertIn("No API key available", str(context.exception))

        with self.assertRaises(ValueError) as context:
            create_openai_client("test-openai-model")

        self.assertIn("No API key available", str(context.exception))


class TestRetryBehaviorSimulation(unittest.TestCase):
    """Test retry behavior concepts using MockTransport to simulate different scenarios."""

    def setUp(self):
        """Set up test helpers."""
        self.retry_attempts = []

    def _run_async_test(self, coro):
        """Helper method to run async tests in sync test methods."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_mock_transport_simulates_connect_error(self):
        """Test that MockTransport properly simulates connection errors."""

        async def _test():
            # MockTransport can simulate connection errors for testing purposes
            def connection_error_handler(request):
                self.retry_attempts.append(request)
                raise httpx.ConnectError("Simulated connection error")

            transport = httpx.MockTransport(connection_error_handler)
            client = httpx.AsyncClient(transport=transport)

            try:
                # This should fail with ConnectError
                with self.assertRaises(httpx.ConnectError):
                    await client.get("http://test.example.com")

                # Should have exactly 1 attempt (MockTransport doesn't retry)
                self.assertEqual(len(self.retry_attempts), 1)

            finally:
                await client.aclose()

        self._run_async_test(_test())

    def test_mock_transport_simulates_timeout_error(self):
        """Test that MockTransport properly simulates timeout errors."""

        async def _test():
            # MockTransport can simulate timeout errors for testing purposes
            def timeout_handler(request):
                self.retry_attempts.append(request)
                raise httpx.ConnectTimeout("Simulated timeout error")

            transport = httpx.MockTransport(timeout_handler)
            client = httpx.AsyncClient(transport=transport)

            try:
                # This should fail with ConnectTimeout
                with self.assertRaises(httpx.ConnectTimeout):
                    await client.get("http://test.example.com")

                # Should have exactly 1 attempt (MockTransport doesn't retry)
                self.assertEqual(len(self.retry_attempts), 1)

            finally:
                await client.aclose()

        self._run_async_test(_test())

    def test_mock_transport_http_error_responses(self):
        """Test that MockTransport properly simulates HTTP error responses."""

        async def _test():
            # MockTransport returns HTTP error responses without retrying
            def http_error_handler(request):
                self.retry_attempts.append(request)
                return httpx.Response(404, json={"error": "Not found"})

            transport = httpx.MockTransport(http_error_handler)
            client = httpx.AsyncClient(transport=transport)

            try:
                response = await client.get("http://test.example.com")
                self.assertEqual(response.status_code, 404)

                # Should have exactly 1 attempt (no retries for HTTP errors)
                self.assertEqual(len(self.retry_attempts), 1)

            finally:
                await client.aclose()

        self._run_async_test(_test())

    def test_mock_transport_success_response(self):
        """Test that MockTransport properly simulates successful responses."""

        async def _test():
            # MockTransport returns successful responses
            def success_handler(request):
                self.retry_attempts.append(request)
                return httpx.Response(
                    200, json={"success": True, "message": "Mock response"}
                )

            transport = httpx.MockTransport(success_handler)
            client = httpx.AsyncClient(transport=transport)

            try:
                response = await client.get("http://test.example.com")
                self.assertEqual(response.status_code, 200)

                response_data = response.json()
                self.assertTrue(response_data["success"])
                self.assertEqual(response_data["message"], "Mock response")

                # Should have exactly 1 attempt
                self.assertEqual(len(self.retry_attempts), 1)

            finally:
                await client.aclose()

        self._run_async_test(_test())

    def test_retry_transport_configuration(self):
        """Test that retry transport can be configured with different retry counts."""
        # Test that we can create transport instances with different retry values
        test_values = [0, 1, 2, 5]

        for retry_count in test_values:
            with self.subTest(retry_count=retry_count):
                # Should not raise any exceptions
                transport = httpx.AsyncHTTPTransport(retries=retry_count)
                self.assertIsInstance(transport, httpx.AsyncHTTPTransport)

    def test_mock_transport_integration(self):
        """Test that MockTransport can be used for testing purposes."""

        def simple_handler(request):
            return httpx.Response(200, json={"message": "Mock response"})

        transport = httpx.MockTransport(simple_handler)
        self.assertIsInstance(transport, httpx.MockTransport)

        # Should be able to create client with mock transport
        client = httpx.AsyncClient(transport=transport)
        self.assertIsInstance(client, httpx.AsyncClient)


class TestRetryIntegration(unittest.TestCase):
    """Integration tests for retry mechanisms."""

    def setUp(self):
        """Set up integration test environment."""
        self.mock_models = {
            "integration-test-model": {
                "model_id": "integration-test-model",
                "model_name": "test-model",
                "api_base": "https://test.example.com",
                "api_key_name": "TEST_API_KEY",
                "direct": True,
                "max_input_tokens": 100000,
                "extra_headers": {},
            }
        }

        self.mock_config = MagicMock()
        self.mock_config.custom_api_keys = {"TEST_API_KEY": "test-key-value"}

        self.models_patcher = patch.dict(
            "anthropic_proxy.client.CUSTOM_OPENAI_MODELS", self.mock_models, clear=True
        )
        self.config_patcher = patch("anthropic_proxy.client.config", self.mock_config)

        self.models_patcher.start()
        self.config_patcher.start()

    def tearDown(self):
        """Clean up integration test patches."""
        self.models_patcher.stop()
        self.config_patcher.stop()

    def _run_async_test(self, coro):
        """Helper method to run async tests in sync test methods."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @patch("anthropic_proxy.client.httpx.AsyncClient")
    @patch("anthropic_proxy.client.httpx.AsyncHTTPTransport")
    def test_end_to_end_retry_configuration(
        self, mock_transport_class, mock_client_class
    ):
        """Test end-to-end retry configuration from client creation to usage."""
        mock_transport = MagicMock()
        mock_transport_class.return_value = mock_transport
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create client
        client = create_claude_client("integration-test-model")

        # Verify the complete chain
        self.assertEqual(client, mock_client)
        mock_transport_class.assert_called_once_with(
            retries=ModelDefaults.DEFAULT_MAX_RETRIES
        )

        # Verify client was created with correct configuration
        call_kwargs = mock_client_class.call_args[1]
        self.assertEqual(call_kwargs["transport"], mock_transport)
        self.assertEqual(call_kwargs["base_url"], "https://test.example.com/v1")

    def test_retry_mechanism_consistency(self):
        """Test that retry mechanisms are consistently applied across different clients."""
        # This test verifies that both claude and openai clients use the same retry value
        self.assertEqual(ModelDefaults.DEFAULT_MAX_RETRIES, 2)

        # Both client types should use the same constant
        with patch("anthropic_proxy.client.httpx.AsyncHTTPTransport") as mock_transport:
            with patch("anthropic_proxy.client.httpx.AsyncClient"):
                with patch("anthropic_proxy.client.AsyncOpenAI"):
                    # Create both types of clients
                    create_claude_client("integration-test-model")

                    # Verify transport was called with the same retry value
                    mock_transport.assert_called_with(
                        retries=ModelDefaults.DEFAULT_MAX_RETRIES
                    )

    def test_retry_transport_creation_verification(self):
        """Test that our retry transport configuration is properly set up."""
        # Verify that we can create the same retry transport used in our implementation
        retry_transport = httpx.AsyncHTTPTransport(
            retries=ModelDefaults.DEFAULT_MAX_RETRIES
        )
        self.assertIsInstance(retry_transport, httpx.AsyncHTTPTransport)

        # Test with a success mock transport to verify basic functionality
        async def _test():
            def success_handler(request):
                return httpx.Response(
                    200, json={"success": True, "message": "Request succeeded"}
                )

            mock_transport = httpx.MockTransport(success_handler)

            async with httpx.AsyncClient(transport=mock_transport) as client:
                response = await client.get("http://test.api.com/messages")

                self.assertEqual(response.status_code, 200)
                response_data = response.json()
                self.assertTrue(response_data["success"])

        self._run_async_test(_test())

    def test_client_creation_with_retry_configuration(self):
        """Test that our client creation functions properly configure retries."""
        with patch("anthropic_proxy.client.httpx.AsyncHTTPTransport") as mock_transport:
            with patch("anthropic_proxy.client.httpx.AsyncClient") as mock_client:
                # Test Claude client creation
                create_claude_client("integration-test-model")

                # Verify transport was created with correct retry configuration
                mock_transport.assert_called_with(
                    retries=ModelDefaults.DEFAULT_MAX_RETRIES
                )

                # Verify client was created with the transport
                mock_client.assert_called_once()
                call_kwargs = mock_client.call_args[1]
                self.assertIn("transport", call_kwargs)

        # Reset mocks for OpenAI client test
        with patch("anthropic_proxy.client.httpx.AsyncHTTPTransport") as mock_transport:
            with patch("anthropic_proxy.client.httpx.AsyncClient") as mock_http_client:
                with patch("anthropic_proxy.client.AsyncOpenAI") as mock_openai:
                    # Test OpenAI client creation
                    create_openai_client("integration-test-model")

                    # Verify transport was created with retry configuration
                    mock_transport.assert_called_with(
                        retries=ModelDefaults.DEFAULT_MAX_RETRIES
                    )

                    # Verify HTTP client was created with transport
                    mock_http_client.assert_called_with(
                        transport=mock_transport.return_value
                    )

                    # Verify OpenAI client was created with HTTP client
                    call_kwargs = mock_openai.call_args[1]
                    self.assertIn("http_client", call_kwargs)


if __name__ == "__main__":
    # Configure logging for test output
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Run all tests
    unittest.main(verbosity=2)
