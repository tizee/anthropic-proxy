import argparse
import asyncio
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Add the parent directory to the path so we can import from server
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import necessary components from the new package structure
from anthropic_proxy.client import (
    CUSTOM_OPENAI_MODELS,
    initialize_custom_models,
)
from anthropic_proxy.config import config as server_config  # rename to avoid conflict
from anthropic_proxy.types import ClaudeMessagesRequest

# --- Test Configuration ---
PROXY_URL = "http://127.0.0.1:8082/v1/messages"


def parse_streaming_response_tokens(
    response_text: str, is_anthropic_format: bool = False
) -> dict[str, int]:
    """Parse streaming response to extract token usage information."""
    input_tokens = 0
    output_tokens = 0

    try:
        if is_anthropic_format:
            # Parse Anthropic format streaming response
            # Look for message_stop or message_delta events which contain usage
            # Try exact pattern first
            usage_pattern = (
                r'"usage":\s*{\s*"input_tokens":\s*(\d+),\s*"output_tokens":\s*(\d+)'
            )
            matches = re.findall(usage_pattern, response_text)
            if matches:
                input_tokens, output_tokens = map(
                    int, matches[-1]
                )  # Take the last usage info
            else:
                # Alternative pattern - sometimes tokens are in different order
                alt_pattern = r'"input_tokens":\s*(\d+).*?"output_tokens":\s*(\d+)'
                matches = re.findall(alt_pattern, response_text, re.DOTALL)
                if matches:
                    input_tokens, output_tokens = map(int, matches[-1])
                else:
                    # Try reversed order pattern
                    rev_pattern = r'"output_tokens":\s*(\d+).*?"input_tokens":\s*(\d+)'
                    matches = re.findall(rev_pattern, response_text, re.DOTALL)
                    if matches:
                        output_tokens, input_tokens = map(int, matches[-1])
        else:
            # Parse OpenAI format streaming response
            # Look for usage in the final chunk
            usage_pattern = r'"usage":\s*{\s*"prompt_tokens":\s*(\d+),\s*"completion_tokens":\s*(\d+)'
            matches = re.findall(usage_pattern, response_text)
            if matches:
                input_tokens, output_tokens = map(int, matches[-1])
            else:
                # Alternative pattern
                alt_pattern = r'"prompt_tokens":\s*(\d+).*?"completion_tokens":\s*(\d+)'
                matches = re.findall(alt_pattern, response_text, re.DOTALL)
                if matches:
                    input_tokens, output_tokens = map(int, matches[-1])
    except Exception as e:
        print(f"Warning: Could not parse token usage from response: {e}")
        # Debug output for troubleshooting
        if len(response_text) > 500:
            print(f"Response sample (last 500 chars): ...{response_text[-500:]}")

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class TokenizedTestResult:
    """Enhanced test result class that includes token-based metrics."""

    def __init__(self):
        self.ttfc = 0.0  # Time to first chunk
        self.total_duration = 0.0
        self.total_chunks = 0
        self.total_content_length = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.first_token_time = None
        self.tokens_per_second = 0.0
        self.cost = 0.0
        self.error = None
        self.response_text = ""

    def calculate_derived_metrics(self):
        """Calculate derived metrics like tokens per second."""
        if self.total_duration > 0:
            # Calculate tokens per second based on output tokens and generation time
            generation_time = (
                self.total_duration - self.ttfc
            ) / 1000  # Convert to seconds
            if generation_time > 0 and self.output_tokens > 0:
                self.tokens_per_second = self.output_tokens / generation_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for easy display."""
        return {
            "Time to First Chunk (ms)": self.ttfc,
            "Total Duration (ms)": self.total_duration,
            "Total Chunks": self.total_chunks,
            "Content Length (bytes)": self.total_content_length,
            "Input Tokens": self.input_tokens,
            "Output Tokens": self.output_tokens,
            "Total Tokens": self.total_tokens,
            "Tokens per Second": self.tokens_per_second,
            "Estimated Cost ($)": self.cost,
            "Average Speed (bytes/sec)": (
                self.total_content_length / (self.total_duration / 1000)
            )
            if self.total_duration > 0
            else 0,
            "error": self.error,
        }


async def run_tokenized_test(
    name: str,
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    model_id: str,
) -> TokenizedTestResult:
    """Runs a performance test with token analysis."""
    print(f"--- Running Test: {name} ---")
    result = TokenizedTestResult()
    start_time = time.perf_counter()
    first_chunk_time = None
    response_chunks = []

    try:
        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=60
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                result.total_chunks += 1
                result.total_content_length += len(chunk)
                response_chunks.append(chunk.decode("utf-8", errors="ignore"))

        end_time = time.perf_counter()

        # Basic timing metrics
        result.ttfc = (first_chunk_time - start_time) * 1000 if first_chunk_time else -1
        result.total_duration = (end_time - start_time) * 1000

        # Parse the full response text for token information
        result.response_text = "".join(response_chunks)
        # Determine format based on URL - proxy endpoints return Anthropic format, direct returns OpenAI format
        is_anthropic_format = (
            "v1/messages" in url or "test_conversion" in url
        )  # Proxy endpoints
        token_info = parse_streaming_response_tokens(
            result.response_text, is_anthropic_format
        )

        result.input_tokens = token_info["input_tokens"]
        result.output_tokens = token_info["output_tokens"]
        result.total_tokens = token_info["total_tokens"]

        # Calculate derived metrics
        result.calculate_derived_metrics()

        # Calculate cost using the same model configuration for both direct and proxy tests
        # This ensures fair cost comparison since both tests use the same underlying model
        if result.total_tokens > 0 and model_id in CUSTOM_OPENAI_MODELS:
            model_config = CUSTOM_OPENAI_MODELS[model_id]
            # Use new per-million token pricing with backward compatibility
            input_cost_per_million = model_config.get("input_cost_per_million_tokens")
            output_cost_per_million = model_config.get("output_cost_per_million_tokens")

            # Backward compatibility: fall back to old per-token pricing if new format not available
            if input_cost_per_million is None:
                input_cost_per_token = model_config.get(
                    "input_cost_per_token", 0.000001
                )
                input_cost_per_million = input_cost_per_token * 1_000_000
            if output_cost_per_million is None:
                output_cost_per_token = model_config.get(
                    "output_cost_per_token", 0.000002
                )
                output_cost_per_million = output_cost_per_token * 1_000_000

            result.cost = (result.input_tokens / 1_000_000 * input_cost_per_million) + (
                result.output_tokens / 1_000_000 * output_cost_per_million
            )

    except httpx.RequestError as e:
        print(
            f"ERROR in {name}: Could not connect to {e.request.url}. Is the server running?"
        )
        result.error = str(e)
    except httpx.HTTPStatusError as e:
        print(
            f"ERROR in {name}: Received status {e.response.status_code}. Response: {e.response.text}"
        )
        result.error = str(e)

    # Display results
    print(f"Results for {name}:")
    if result.error:
        print(f"  Error: {result.error}")
    else:
        result_dict = result.to_dict()
        for key, value in result_dict.items():
            if key != "error":
                if isinstance(value, int | float):
                    if key == "Estimated Cost ($)":
                        print(f"  {key}: ${value:.6f}")
                    elif "Token" in key or key == "Total Chunks":
                        print(f"  {key}: {int(value)}")
                    else:
                        print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
    print("-" * 50)

    return result


async def main(args):
    """
    Third-party Model Proxy Performance Test

    This test measures the performance overhead introduced by the proxy server when
    routing requests to third-party models (DeepSeek, Gemini, etc.).

    Test Flow:
    1. Direct Test: Send OpenAI-format request directly to model's API endpoint
    2. Proxy Test: Send Anthropic-format request to proxy, which translates and forwards

    Both tests use the same model and content to ensure fair comparison.
    """
    # Initialize model configurations from server.py
    # This loads models.yaml and populates CUSTOM_OPENAI_MODELS
    initialize_custom_models()

    model_id = args.model_id
    if model_id not in CUSTOM_OPENAI_MODELS:
        print(f"🔴 Error: Model ID '{model_id}' not found in models.yaml.")
        print(f"Available models: {list(CUSTOM_OPENAI_MODELS.keys())}")
        sys.exit(1)

    print(f"🔧 Testing third-party model: {model_id}")
    print(f"📍 Model endpoint: {CUSTOM_OPENAI_MODELS[model_id]['api_base']}")
    model_config = CUSTOM_OPENAI_MODELS[model_id]

    # --- Test Payloads ---
    # Proxy test: Send Anthropic-format request to proxy server
    # The proxy will translate this to OpenAI format and forward to the model
    proxy_payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Tell me a very short story about a robot."}
        ],
        "max_tokens": 150,
        "stream": True,
    }

    # Direct test: Send OpenAI-format request directly to the model's API
    # This bypasses the proxy to measure baseline performance
    claude_request = ClaudeMessagesRequest(**proxy_payload)
    direct_payload = claude_request.to_openai_request()
    direct_payload["model"] = model_config[
        "model_name"
    ]  # Use the model_name for direct API

    # --- Test Execution ---
    print("\n" + "=" * 60)
    print("PROXY PERFORMANCE TEST SETUP")
    print("=" * 60)
    print("📋 Test Configuration:")
    print(f"   • Third-party Model: {model_config['model_name']}")
    print(f"   • Direct API Endpoint: {model_config['api_base']}")
    print(f"   • Proxy Server: {PROXY_URL}")
    print("   • Request Content: Same for both tests")
    print("   • Streaming Mode: Enabled")

    # Debug: Show actual payloads for comparison
    print("\n🔍 PAYLOAD COMPARISON:")
    print("   Direct (OpenAI format):")
    print(f"     • model: {direct_payload['model']}")
    print(f"     • messages: {len(direct_payload['messages'])} messages")
    print(f"     • max_tokens: {direct_payload['max_tokens']}")
    print("   Proxy (Anthropic format):")
    print(f"     • model: {proxy_payload['model']}")
    print(f"     • messages: {len(proxy_payload['messages'])} messages")
    print(f"     • max_tokens: {proxy_payload['max_tokens']}")
    print("\n⚠️  Prerequisites:")
    print("   1. Proxy server must be running: `make run`")
    print("   2. API key must be configured for the model")
    print("=" * 60)

    # Test 1: Baseline - Direct connection to third-party model's API
    api_key_name = model_config.get("api_key_name", "OPENAI_API_KEY")
    api_key = server_config.custom_api_keys.get(api_key_name)
    if not api_key:
        print(f"🔴 Error: No API key found for {api_key_name}")
        sys.exit(1)

    direct_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    direct_api_url = f"{model_config['api_base']}/chat/completions"

    print("\n🎯 Test 1: Direct API Call (Baseline Performance)")
    async with httpx.AsyncClient() as direct_client:
        direct_results = await run_tokenized_test(
            "DIRECT API",
            direct_client,
            direct_api_url,
            direct_payload,
            direct_headers,
            model_id,
        )

    # Wait a moment before the next test
    await asyncio.sleep(2)

    # Test 2: Via proxy server (measures translation overhead)
    # Use test_conversion endpoint to bypass routing and test the specified model directly
    test_conversion_url = "http://127.0.0.1:8082/v1/messages/test_conversion"
    print("\n🎯 Test 2: Via Proxy Server (Measures Translation Overhead)")
    print("   Using test_conversion endpoint to bypass model routing")
    async with httpx.AsyncClient() as proxy_client:
        proxy_results = await run_tokenized_test(
            "PROXY SERVER",
            proxy_client,
            test_conversion_url,
            proxy_payload,
            {"Content-Type": "application/json"},
            model_id,
        )

    # --- Proxy Performance Analysis ---
    print("\n" + "=" * 60)
    print("PROXY SERVER PERFORMANCE ANALYSIS")
    print("=" * 60)

    if not direct_results.error and not proxy_results.error:
        # Calculate proxy overhead metrics
        overhead_ttfc = proxy_results.ttfc - direct_results.ttfc
        overhead_duration = proxy_results.total_duration - direct_results.total_duration

        # Token throughput impact
        direct_tps = direct_results.tokens_per_second
        proxy_tps = proxy_results.tokens_per_second
        throughput_loss = (
            ((direct_tps - proxy_tps) / direct_tps * 100) if direct_tps > 0 else 0
        )

        # Translation overhead analysis
        response_size_overhead = (
            proxy_results.total_content_length - direct_results.total_content_length
        )
        translation_overhead_percent = (
            (response_size_overhead / direct_results.total_content_length * 100)
            if direct_results.total_content_length > 0
            else 0
        )

        # Token consistency check (should be identical for same model/request)
        token_input_diff = proxy_results.input_tokens - direct_results.input_tokens
        token_output_diff = proxy_results.output_tokens - direct_results.output_tokens
        tokens_consistent = token_input_diff == 0 and token_output_diff == 0

        # Calculate percentage changes for latency
        ttfc_percent_change = (
            (overhead_ttfc / direct_results.ttfc * 100)
            if direct_results.ttfc > 0
            else 0
        )
        duration_percent_change = (
            (overhead_duration / direct_results.total_duration * 100)
            if direct_results.total_duration > 0
            else 0
        )

        print("🔍 PROXY TRANSLATION OVERHEAD:")
        print("   ⏱️  Latency Added (vs. Direct API):")
        print(
            f"      • Time to First Chunk: {overhead_ttfc:+.2f} ms ({ttfc_percent_change:+.1f}%)"
        )
        print(
            f"      • Total Request Time: {overhead_duration:+.2f} ms ({duration_percent_change:+.1f}%)"
        )

        print("\n   🚀 Throughput Impact:")
        print(f"      • Direct API: {direct_tps:.2f} tokens/sec")
        print(f"      • Via Proxy: {proxy_tps:.2f} tokens/sec")
        print(f"      • Performance Loss: {throughput_loss:.1f}%")

        print("\n   📦 Response Format Overhead:")
        print(f"      • Direct (OpenAI): {direct_results.total_content_length:,} bytes")
        print(
            f"      • Proxy (Anthropic): {proxy_results.total_content_length:,} bytes"
        )
        print(
            f"      • Format Overhead: {response_size_overhead:+,} bytes ({translation_overhead_percent:+.1f}%)"
        )

        print("\n   🎯 Token Accuracy:")
        print(
            f"      • Input Tokens: {direct_results.input_tokens} (direct) vs {proxy_results.input_tokens} (proxy) [{token_input_diff:+d}]"
        )
        print(
            f"      • Output Tokens: {direct_results.output_tokens} (direct) vs {proxy_results.output_tokens} (proxy) [{token_output_diff:+d}]"
        )
        print(
            f"      • Consistency: {'✅ Perfect Match' if tokens_consistent else '⚠️  Tokens Differ'}"
        )

        print("\n   💰 Cost Impact:")
        print(f"      • Same Model Used: Both tests use {model_config['model_name']}")
        print("      • Cost Difference: Proportional to token differences only")
        if not tokens_consistent:
            cost_diff = proxy_results.cost - direct_results.cost
            print(f"      • Additional Cost: ${cost_diff:+.6f}")

        # Proxy efficiency assessment
        if overhead_ttfc < 50 and throughput_loss < 5:
            efficiency = "🟢 EXCELLENT"
            summary = "Proxy adds minimal overhead."
            criteria = "TTFC Overhead < 50ms AND Throughput Loss < 5%"
        elif overhead_ttfc < 100 and throughput_loss < 10:
            efficiency = "🟡 GOOD"
            summary = "Acceptable proxy performance."
            criteria = "TTFC Overhead < 100ms AND Throughput Loss < 10%"
        elif overhead_ttfc < 200 and throughput_loss < 20:
            efficiency = "🟠 FAIR"
            summary = "Some optimization opportunities."
            criteria = "TTFC Overhead < 200ms AND Throughput Loss < 20%"
        else:
            efficiency = "🔴 POOR"
            summary = "Significant proxy overhead detected."
            criteria = "TTFC Overhead >= 200ms OR Throughput Loss >= 20%"

        print(f"\n🏆 PROXY EFFICIENCY RATING: {efficiency}")
        print(f"   {summary}")
        print(f"   - Criteria: {criteria}")
        print(
            f"   - Actuals: TTFC Overhead = {overhead_ttfc:.2f}ms, Throughput Loss = {throughput_loss:.1f}%"
        )

        # Actionable insights
        print("\n💡 KEY INSIGHTS:")
        if overhead_ttfc > 100:
            print(
                f"   • High TTFC overhead ({overhead_ttfc:.0f}ms) - Check network latency or processing delays"
            )
        if throughput_loss > 15:
            print(
                f"   • Significant throughput loss ({throughput_loss:.1f}%) - Consider request processing optimizations"
            )
        if translation_overhead_percent > 30:
            print(
                f"   • Large format overhead ({translation_overhead_percent:.1f}%) - Anthropic format is verbose"
            )
        if not tokens_consistent:
            print("   • Token inconsistency detected - May indicate translation issues")
        if overhead_ttfc < 50 and throughput_loss < 10:
            print("   • Excellent proxy performance - Translation overhead is minimal")

    else:
        print("❌ Could not complete analysis due to test errors:")
        if direct_results.error:
            print(f"   📍 Direct API Error: {direct_results.error}")
        if proxy_results.error:
            print(f"   🔄 Proxy Server Error: {proxy_results.error}")

    print("\n" + "=" * 60)
    print("✅ Proxy performance analysis complete")
    print("   Use these metrics to optimize your proxy server configuration")
    print("=" * 60)


if __name__ == "__main__":
    # Don't run the test if there are no args.
    if len(sys.argv) == 1:
        print("Third-Party Model Proxy Performance Test")
        print("=" * 50)
        print(
            "Usage: python performance_test.py --model_id <model_id_from_models.yaml>"
        )
        print("       python performance_test.py -m <model_id_from_models.yaml>")
        print("\n🎯 SPEED & PERFORMANCE ANALYSIS:")
        print("  🚀 Token throughput (tokens/sec) - Critical for user experience")
        print("  ⏱️  Latency overhead (TTFC, total duration)")
        print("  📊 Translation efficiency (format conversion cost)")
        print("  🎯 Token consistency (accuracy verification)")
        print("  💰 Cost impact analysis")
        print("\n📈 Measures proxy server overhead when routing to third-party models")
        print("   (DeepSeek, Gemini, etc.) vs direct API calls")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Token-based Performance Analysis for Claude Proxy Server"
    )
    parser.add_argument(
        "--model_id",
        "-m",
        type=str,
        required=True,
        help="The model ID to test (must be in models.yaml)",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
