import asyncio
import time
import json
import httpx
import argparse
import sys
import os
from typing import Dict, Any, AsyncGenerator

# Add the parent directory to the path so we can import from server
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import necessary components from your server.py
from server import (
    initialize_custom_models,
    create_openai_client,
    CUSTOM_OPENAI_MODELS,
    config as server_config, # rename to avoid conflict
)
from models import ClaudeMessagesRequest


# --- Test Configuration ---
PROXY_URL = "http://127.0.0.1:8082/v1/messages"


async def run_test(name: str, client: httpx.AsyncClient, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """Runs a single performance test against a given URL and collects metrics."""
    print(f"--- Running Test: {name} ---")
    start_time = time.perf_counter()
    first_chunk_time = None
    total_chunks = 0
    total_content_length = 0

    try:
        async with client.stream("POST", url, json=payload, headers=headers, timeout=60) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                total_chunks += 1
                total_content_length += len(chunk)

        end_time = time.perf_counter()

        ttfc = (first_chunk_time - start_time) * 1000 if first_chunk_time else -1
        total_duration = (end_time - start_time) * 1000
        avg_speed = (total_content_length / (total_duration / 1000)) if total_duration > 0 else 0

        results = {
            "Time to First Chunk (ms)": ttfc,
            "Total Duration (ms)": total_duration,
            "Average Speed (bytes/sec)": avg_speed,
            "Total Chunks": total_chunks,
            "error": None
        }

    except httpx.RequestError as e:
        print(f"ERROR in {name}: Could not connect to {e.request.url}. Is the server running?")
        results = {"error": str(e)}
    except httpx.HTTPStatusError as e:
        print(f"ERROR in {name}: Received status {e.response.status_code}. Response: {e.response.text}")
        results = {"error": str(e)}

    print(f"Results for {name}:")
    if results.get("error"):
        print(f"  Error: {results['error']}")
    else:
        for key, value in results.items():
            if key != "error":
                print(f"  {key}: {value:.2f}")
    print("-" * 25)

    return results


async def main(args):
    """Main function to orchestrate the performance tests."""
    # Initialize model configurations from server.py
    # This loads models.yaml and populates CUSTOM_OPENAI_MODELS
    initialize_custom_models()

    model_id = args.model_id
    if model_id not in CUSTOM_OPENAI_MODELS:
        print(f"🔴 Error: Model ID '{model_id}' not found in models.yaml.")
        print(f"Available models: {list(CUSTOM_OPENAI_MODELS.keys())}")
        sys.exit(1)

    print(f"🔧 Testing with model: {model_id}")
    model_config = CUSTOM_OPENAI_MODELS[model_id]

    # --- Payloads ---
    # For the proxy, we send an Anthropic-formatted request
    proxy_payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Tell me a very short story about a robot."}],
        "max_tokens": 150,
        "stream": True,
    }

    # For the direct test, we convert the Anthropic request to the OpenAI format
    # This reuses the same logic as the proxy server itself for a fair comparison
    claude_request = ClaudeMessagesRequest(**proxy_payload)
    direct_payload = claude_request.to_openai_request()
    direct_payload["model"] = model_config["model_name"] # Use the specific model_name for the direct API

    # --- Test Execution ---
    print("\nIMPORTANT:")
    print("1. Please ensure the main proxy server is running in another terminal via `make run`.")
    print(f"2. Direct test will connect to: {model_config['api_base']}")

    # Test 1: Direct connection to the model's API
    direct_client = create_openai_client(model_id)
    direct_api_url = f"{model_config['api_base']}/chat/completions"
    direct_results = await run_test("Direct Connection", direct_client, direct_api_url, direct_payload, {})

    # Wait a moment before the next test
    await asyncio.sleep(2)

    # Test 2: Connection via the proxy server
    async with httpx.AsyncClient() as proxy_client:
        proxy_results = await run_test("Proxy Connection", proxy_client, PROXY_URL, proxy_payload, {"Content-Type": "application/json"})

    # --- Comparison ---
    print("\n--- Performance Comparison ---")
    if not direct_results.get("error") and not proxy_results.get("error"):
        overhead_ttfc = proxy_results["Time to First Chunk (ms)"] - direct_results["Time to First Chunk (ms)"]
        overhead_duration = proxy_results["Total Duration (ms)"] - direct_results["Total Duration (ms)"]
        
        direct_speed = direct_results["Average Speed (bytes/sec)"]
        proxy_speed = proxy_results["Average Speed (bytes/sec)"]
        speed_diff_percent = ((direct_speed - proxy_speed) / direct_speed) * 100 if direct_speed > 0 else 0

        print(f"Proxy Overhead (Time to First Chunk): {overhead_ttfc:.2f} ms")
        print(f"Proxy Overhead (Total Duration): {overhead_duration:.2f} ms")
        print(f"Proxy Speed Reduction: {speed_diff_percent:.2f}% slower than direct")
    else:
        print("Could not generate comparison due to errors in one of the tests.")

    print("\nDone.")

if __name__ == "__main__":
    # Don't run the test if there are no args.
    if len(sys.argv) == 1:
        print("Usage: python performance_test.py --model_id <model_id_from_models.yaml>")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Proxy Server Performance Test")
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to test (must be in models.yaml)")
    args = parser.parse_args()

    asyncio.run(main(args))
