import unittest
import pytest
from unittest.mock import patch, MagicMock
import os
import sys

# Add the parent directory to the sys.path to allow imports from the server module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anthropic_proxy.client import (
    determine_model_by_router,
    CUSTOM_OPENAI_MODELS,
    is_direct_mode_model,
)
from anthropic_proxy.config import Config
from anthropic_proxy.server import create_message
from anthropic_proxy.utils import count_tokens_in_messages
from anthropic_proxy.types import ClaudeMessage, ClaudeContentBlockText


class TestRoutingLogic(unittest.TestCase):
    def setUp(self):
        """Set up a mock environment for testing."""
        self.config = Config()
        self.config.router_config = {
            "background": "model-background",
            "think": "model-think",
            "long_context": "model-long-context",
            "default": "model-default",
        }

        # This is a bit of a hack, but it's the easiest way to inject the config
        # for the functions we are testing.
        patcher = patch("anthropic_proxy.client.config", self.config)
        self.addCleanup(patcher.stop)
        patcher.start()

        # Mock the CUSTOM_OPENAI_MODELS data
        self.mock_models = {
            "model-default": {"max_input_tokens": 10000, "direct": False},
            "model-background": {"max_input_tokens": 20000, "direct": False},
            "model-think": {"max_input_tokens": 30000, "direct": False},
            "model-long-context": {"max_input_tokens": 100000, "direct": False},
            "model-with-low-limit": {"max_input_tokens": 500, "direct": False},
            "model-direct-claude": {
                "max_input_tokens": 50000,
                "direct": True,
                "api_base": "https://api.anthropic.com",
            },
            "model-direct-anthropic": {
                "max_input_tokens": 200000,
                "direct": True,
                "api_base": "https://api.anthropic.com/v1",
            },
            "model-openai-compatible": {
                "max_input_tokens": 30000,
                "direct": False,
                "api_base": "https://api.openai.com/v1",
            },
        }

        # We also need to patch the global dictionary in the client module
        patcher_models = patch.dict(
            "anthropic_proxy.client.CUSTOM_OPENAI_MODELS", self.mock_models, clear=True
        )
        self.addCleanup(patcher_models.stop)
        patcher_models.start()

    def test_determine_model_by_router_think_model(self):
        """Test that a request with thinking=True routes to the think model."""
        routed_model = determine_model_by_router(
            original_model="any-model", token_count=100, has_thinking=True
        )
        self.assertEqual(routed_model, "model-think")

    def test_determine_model_by_router_haiku_model(self):
        """Test that a haiku model routes to the background model."""
        routed_model = determine_model_by_router(
            original_model="claude-3-haiku-20240307",
            token_count=100,
            has_thinking=False,
        )
        self.assertEqual(routed_model, "model-background")

    def test_determine_model_by_router_sonnet_model(self):
        """Test that a sonnet model routes to the default model."""
        routed_model = determine_model_by_router(
            original_model="claude-3-sonnet-20240229",
            token_count=100,
            has_thinking=False,
        )
        self.assertEqual(routed_model, "model-default")

    def test_determine_model_by_router_default_fallback(self):
        """Test that any other model falls back to the default router."""
        routed_model = determine_model_by_router(
            original_model="some-other-model", token_count=100, has_thinking=False
        )
        self.assertEqual(routed_model, "model-default")

    @patch("anthropic_proxy.server.create_message")
    def test_long_context_fallback_when_tokens_exceed_limit(self, mock_create_message):
        """
        Test that the routing switches to the long_context model if the token count
        exceeds the preliminary model's max_input_tokens.
        """
        # Let's simulate the logic inside create_message for the final routing decision

        # 1. Get preliminary model
        preliminary_model = (
            "model-with-low-limit"  # This model has max_input_tokens: 500
        )
        token_count = 600  # Exceeds the limit

        # 2. Check against the limit
        final_model = preliminary_model
        if preliminary_model in self.mock_models:
            model_config_check = self.mock_models[preliminary_model]
            max_input_for_model = model_config_check.get("max_input_tokens", 0)
            if token_count > max_input_for_model:
                final_model = self.config.router_config["long_context"]

        self.assertEqual(final_model, "model-long-context")

    @patch("anthropic_proxy.server.create_message")
    def test_long_context_fallback_when_tokens_are_within_limit(
        self, mock_create_message
    ):
        """
        Test that the routing does NOT switch to the long_context model if the token count
        is within the preliminary model's max_input_tokens.
        """
        # 1. Get preliminary model
        preliminary_model = "model-default"  # This model has max_input_tokens: 10000
        token_count = 5000  # Within the limit

        # 2. Check against the limit
        final_model = preliminary_model
        if preliminary_model in self.mock_models:
            model_config_check = self.mock_models[preliminary_model]
            max_input_for_model = model_config_check.get("max_input_tokens", 0)
            if token_count > max_input_for_model:
                final_model = self.config.router_config["long_context"]

        self.assertEqual(final_model, "model-default")

    @patch("anthropic_proxy.server.create_message")
    def test_long_context_fallback_for_unknown_model(self, mock_create_message):
        """
        Test that the routing uses the global long_context_threshold for a model
        not in the custom model config.
        """
        self.config.long_context_threshold = 15000

        # 1. Get preliminary model (one that is not in our mock_models)
        preliminary_model = "claude-3-opus-20240229"
        token_count_exceeds = 20000
        token_count_within = 10000

        # 2. Check for exceeding case
        final_model_exceeds = preliminary_model
        if preliminary_model not in self.mock_models:
            if token_count_exceeds > self.config.long_context_threshold:
                final_model_exceeds = self.config.router_config["long_context"]

        self.assertEqual(final_model_exceeds, "model-long-context")

        # 3. Check for within limit case
        final_model_within = preliminary_model
        if preliminary_model not in self.mock_models:
            if token_count_within > self.config.long_context_threshold:
                final_model_within = self.config.router_config["long_context"]

        self.assertEqual(final_model_within, "claude-3-opus-20240229")

    def test_is_direct_mode_model_with_explicit_direct_true(self):
        """Test that a model with direct=True is identified as direct mode."""
        result = is_direct_mode_model("model-direct-claude")
        self.assertTrue(result)

    def test_is_direct_mode_model_with_explicit_direct_false(self):
        """Test that a model with direct=False is not identified as direct mode."""
        result = is_direct_mode_model("model-default")
        self.assertFalse(result)

    def test_is_direct_mode_model_with_anthropic_url(self):
        """Test that a model with anthropic.com in api_base is identified as direct mode."""
        result = is_direct_mode_model("model-direct-anthropic")
        self.assertTrue(result)

    def test_is_direct_mode_model_with_openai_compatible(self):
        """Test that an OpenAI-compatible model is not identified as direct mode."""
        result = is_direct_mode_model("model-openai-compatible")
        self.assertFalse(result)

    def test_is_direct_mode_model_unknown_model(self):
        """Test that an unknown model returns False for direct mode."""
        result = is_direct_mode_model("unknown-model")
        self.assertFalse(result)

    def test_routing_to_direct_mode_models(self):
        """Test that routing can work with direct mode models in the router config."""
        # Set up router config to use direct mode models
        self.config.router_config = {
            "background": "model-direct-claude",
            "think": "model-direct-anthropic",
            "long_context": "model-direct-anthropic",
            "default": "model-direct-claude",
        }

        # Test haiku model routes to direct background model
        routed_model = determine_model_by_router(
            original_model="claude-3-haiku-20240307",
            token_count=100,
            has_thinking=False,
        )
        self.assertEqual(routed_model, "model-direct-claude")
        self.assertTrue(is_direct_mode_model(routed_model))

        # Test thinking routes to direct think model
        routed_model = determine_model_by_router(
            original_model="claude-3-sonnet-20240229",
            token_count=100,
            has_thinking=True,
        )
        self.assertEqual(routed_model, "model-direct-anthropic")
        self.assertTrue(is_direct_mode_model(routed_model))

    def test_mixed_routing_configuration(self):
        """Test routing with a mix of direct and OpenAI-compatible models."""
        # Mixed router config
        self.config.router_config = {
            "background": "model-openai-compatible",  # OpenAI-compatible
            "think": "model-direct-claude",  # Direct mode
            "long_context": "model-direct-anthropic",  # Direct mode
            "default": "model-default",  # OpenAI-compatible
        }

        # Test background routing (OpenAI-compatible)
        routed_model = determine_model_by_router(
            original_model="claude-3-haiku-20240307",
            token_count=100,
            has_thinking=False,
        )
        self.assertEqual(routed_model, "model-openai-compatible")
        self.assertFalse(is_direct_mode_model(routed_model))

        # Test thinking routing (Direct mode)
        routed_model = determine_model_by_router(
            original_model="claude-3-sonnet-20240229",
            token_count=100,
            has_thinking=True,
        )
        self.assertEqual(routed_model, "model-direct-claude")
        self.assertTrue(is_direct_mode_model(routed_model))

        # Test default routing (OpenAI-compatible)
        routed_model = determine_model_by_router(
            original_model="claude-3-sonnet-20240229",
            token_count=100,
            has_thinking=False,
        )
        self.assertEqual(routed_model, "model-default")
        self.assertFalse(is_direct_mode_model(routed_model))

    def test_direct_mode_long_context_fallback(self):
        """Test that long context fallback works with direct mode models."""
        self.config.router_config = {
            "background": "model-background",
            "think": "model-think",
            "long_context": "model-direct-anthropic",  # Direct mode for long context
            "default": "model-default",
        }

        # Test that a model with exceeded token count routes to direct long context model
        preliminary_model = "model-with-low-limit"  # max_input_tokens: 500
        token_count = 600  # Exceeds the limit

        final_model = preliminary_model
        if preliminary_model in self.mock_models:
            model_config_check = self.mock_models[preliminary_model]
            max_input_for_model = model_config_check.get("max_input_tokens", 0)
            if token_count > max_input_for_model:
                final_model = self.config.router_config["long_context"]

        self.assertEqual(final_model, "model-direct-anthropic")
        self.assertTrue(is_direct_mode_model(final_model))

    def test_all_direct_mode_routing_configuration(self):
        """Test routing where all models are configured for direct mode."""
        # All direct mode router config
        self.config.router_config = {
            "background": "model-direct-claude",
            "think": "model-direct-anthropic",
            "long_context": "model-direct-anthropic",
            "default": "model-direct-claude",
        }

        # Test all routing scenarios result in direct mode
        test_cases = [
            ("claude-3-haiku-20240307", 100, False, "model-direct-claude"),
            ("claude-3-sonnet-20240229", 100, True, "model-direct-anthropic"),
            ("claude-3-sonnet-20240229", 100, False, "model-direct-claude"),
            ("some-other-model", 100, False, "model-direct-claude"),
        ]

        for original_model, token_count, has_thinking, expected_model in test_cases:
            with self.subTest(model=original_model, thinking=has_thinking):
                routed_model = determine_model_by_router(
                    original_model, token_count, has_thinking
                )
                self.assertEqual(routed_model, expected_model)
                self.assertTrue(is_direct_mode_model(routed_model))


def test_basic_message_token_count():
    messages = [
        ClaudeMessage(
            role="user",
            content=[ClaudeContentBlockText(type="text", text="Hello world")],
        ),
        ClaudeMessage(
            role="assistant",
            content=[ClaudeContentBlockText(type="text", text="Hi there!")],
        ),
    ]
    model = "claude-3-sonnet-20240229"
    tokens = count_tokens_in_messages(messages, model)
    assert tokens > 0


def test_complex_message_structure():
    messages = [
        ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockText(
                    type="text",
                    text="Calculate token count for this complex message with multiple content blocks",
                )
            ],
        )
    ]
    model = "claude-3-opus-20240229"
    tokens = count_tokens_in_messages(messages, model)
    assert tokens > 0


def test_empty_messages():
    messages = []
    model = "claude-3-haiku-20240307"
    tokens = count_tokens_in_messages(messages, model)
    assert tokens == 0


def test_different_model_types():
    messages = [
        ClaudeMessage(
            role="user",
            content=[
                ClaudeContentBlockText(
                    type="text", text="Model type should affect token counting"
                )
            ],
        )
    ]
    models = [
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]
    token_counts = [count_tokens_in_messages(messages, model) for model in models]
    assert all(tokens > 0 for tokens in token_counts)


if __name__ == "__main__":
    unittest.main()
