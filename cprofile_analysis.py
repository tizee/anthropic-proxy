#!/usr/bin/env python3
"""
cProfile-based performance analysis for Claude Code Proxy.
Uses Python's standard profiling tools to identify bottlenecks.
"""

import asyncio
import cProfile
import pstats
import io
import json
import os
import time
import yaml
from typing import Dict, Any
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pstats import SortKey

# Load environment variables
load_dotenv()


def load_model_config(model_id: str) -> Dict[str, Any]:
    """Load model configuration from models.yaml"""
    config_file = "models.yaml"
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Models config file not found: {config_file}")
    
    try:
        with open(config_file) as file:
            models = yaml.safe_load(file)
        
        if not models:
            raise ValueError(f"No models found in config file: {config_file}")
        
        # Find the specific model
        for model in models:
            if model.get("model_id") == model_id:
                return model
        
        raise ValueError(f"Model '{model_id}' not found in config")
        
    except Exception as e:
        raise RuntimeError(f"Error loading model config: {e}")


async def direct_api_call():
    """Direct API call for baseline performance"""
    model_id = "gemini-2.5-flash-lite-preview-06-17"
    
    # Load model configuration
    model_config = load_model_config(model_id)
    
    # Get API key
    api_key_name = model_config.get("api_key_name")
    api_key = os.environ.get(api_key_name)
    
    # Create OpenAI client
    api_base = model_config.get("api_base")
    model_name = model_config.get("model_name", model_id)
    extra_headers = model_config.get("extra_headers", {})
    
    client = AsyncOpenAI(
        base_url=api_base,
        api_key=api_key,
        default_headers=extra_headers
    )
    
    messages = [
        {
            "role": "user", 
            "content": "Write a simple Python function to calculate the factorial of a number. Explain your approach briefly."
        }
    ]
    
    try:
        # Get max_tokens from config
        max_tokens = model_config.get("max_tokens", 1000)
        
        # Create streaming request
        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            store=False,
            stream_options={"include_usage": True}
        )
        
        accumulated_content = ""
        usage_info = None
        
        async for chunk in stream:
            # Check for content
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                if choice.delta and choice.delta.content:
                    accumulated_content += choice.delta.content
            
            # Check for usage
            if chunk.usage:
                usage_info = chunk.usage
        
        await client.close()
        
        output_tokens = usage_info.completion_tokens if usage_info else len(accumulated_content) // 4
        return {
            "output_tokens": output_tokens,
            "content_length": len(accumulated_content),
            "success": True
        }
        
    except Exception as e:
        await client.close()
        return {"success": False, "error": str(e)}


async def proxy_api_call():
    """API call via proxy for comparison"""
    model_id = "gemini-2.5-flash-lite-preview-06-17"
    
    try:
        # Load model configuration for max_tokens
        model_config = load_model_config(model_id)
        max_tokens = model_config.get("max_tokens", 1000)
        
    except Exception as e:
        max_tokens = 1000
    
    url = "http://127.0.0.1:8082/v1/messages"
    request_data = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write a simple Python function to calculate the factorial of a number. Explain your approach briefly."
                    }
                ]
            }
        ],
        "stream": True
    }
    
    content_chunks = 0
    usage_info = None
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status_code != 200:
                    return {"success": False, "error": f"HTTP {response.status_code}"}
                
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        # Parse SSE format
                        lines = chunk.strip().split('\n')
                        
                        for line in lines:
                            if line.startswith('data: '):
                                current_data = line[6:].strip()
                                
                                if current_data == "[DONE]":
                                    break
                                
                                try:
                                    data = json.loads(current_data)
                                    event_type = data.get("type", "unknown")
                                    
                                    # Count content chunks
                                    if (event_type == "content_block_delta" and 
                                        data.get("delta", {}).get("type") == "text_delta"):
                                        content_chunks += 1
                                    
                                    # Extract usage info
                                    elif event_type == "message_delta" and "usage" in data:
                                        usage_info = data["usage"]
                                
                                except json.JSONDecodeError:
                                    pass
        
        output_tokens = usage_info.get("output_tokens", 0) if usage_info else 0
        return {
            "output_tokens": output_tokens,
            "content_chunks": content_chunks,
            "success": True
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_with_profiler(async_func, profile_name):
    """Run an async function with cProfile"""
    print(f"\n🔍 Profiling {profile_name}...")
    
    # Create profiler
    pr = cProfile.Profile()
    
    # Profile the execution
    pr.enable()
    start_time = time.perf_counter()
    result = asyncio.run(async_func())
    end_time = time.perf_counter()
    pr.disable()
    
    total_time = end_time - start_time
    
    # Save profile data
    profile_file = f"{profile_name}_profile.prof"
    pr.dump_stats(profile_file)
    
    # Generate text report
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    
    print(f"⏱️  Total execution time: {total_time:.3f}s")
    if result.get("success"):
        print(f"📊 Output tokens: {result.get('output_tokens', 0)}")
        if total_time > 0:
            throughput = result.get('output_tokens', 0) / total_time
            print(f"🚀 Token throughput: {throughput:.1f} tokens/s")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    # Print top 20 functions by cumulative time
    print(f"\n📈 Top 20 functions by cumulative time ({profile_name}):")
    print("=" * 80)
    ps.sort_stats(SortKey.CUMULATIVE).print_stats(20)
    
    # Print top 10 functions by internal time
    print(f"\n⚡ Top 10 functions by internal time ({profile_name}):")
    print("=" * 80)
    ps.sort_stats(SortKey.TIME).print_stats(10)
    
    # Save detailed report
    report_file = f"{profile_name}_report.txt"
    with open(report_file, 'w') as f:
        # Full stats sorted by cumulative time
        ps_file = pstats.Stats(pr, stream=f)
        ps_file.sort_stats(SortKey.CUMULATIVE).print_stats()
        f.write("\n" + "="*80 + "\n")
        f.write("TOP FUNCTIONS BY INTERNAL TIME:\n")
        f.write("="*80 + "\n")
        ps_file.sort_stats(SortKey.TIME).print_stats()
    
    print(f"📄 Detailed report saved to: {report_file}")
    print(f"📊 Profile data saved to: {profile_file}")
    
    return result, total_time, profile_file


def analyze_profile_comparison(direct_file, proxy_file):
    """Compare two profile files and identify differences"""
    print(f"\n🔬 COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    # Load both profiles
    direct_stats = pstats.Stats(direct_file)
    proxy_stats = pstats.Stats(proxy_file)
    
    # Get function statistics
    direct_functions = direct_stats.get_stats_profile()
    proxy_functions = proxy_stats.get_stats_profile()
    
    # Find functions that exist in proxy but not in direct (proxy overhead)
    proxy_only_functions = []
    for func_name in proxy_functions.func_profiles:
        if func_name not in direct_functions.func_profiles:
            func_profile = proxy_functions.func_profiles[func_name]
            if func_profile.cumtime > 0.001:  # Only significant functions
                proxy_only_functions.append((func_name, func_profile.cumtime))
    
    # Sort by cumulative time
    proxy_only_functions.sort(key=lambda x: x[1], reverse=True)
    
    print(f"🎯 Functions unique to proxy (potential overhead sources):")
    print("-" * 80)
    for func_name, cumtime in proxy_only_functions[:15]:
        # Clean up function name for display
        if hasattr(func_name, '__name__'):
            display_name = func_name.__name__
        else:
            display_name = str(func_name)
        print(f"  {cumtime:8.4f}s  {display_name}")
    
    # Find functions with significantly higher cumulative time in proxy
    print(f"\n📊 Functions with significantly higher time in proxy:")
    print("-" * 80)
    time_differences = []
    for func_name in direct_functions.func_profiles:
        if func_name in proxy_functions.func_profiles:
            direct_time = direct_functions.func_profiles[func_name].cumtime
            proxy_time = proxy_functions.func_profiles[func_name].cumtime
            if direct_time > 0 and proxy_time > direct_time * 2:  # 2x or more difference
                ratio = proxy_time / direct_time
                time_differences.append((func_name, direct_time, proxy_time, ratio))
    
    time_differences.sort(key=lambda x: x[3], reverse=True)  # Sort by ratio
    for func_name, direct_time, proxy_time, ratio in time_differences[:10]:
        if hasattr(func_name, '__name__'):
            display_name = func_name.__name__
        else:
            display_name = str(func_name)
        print(f"  {ratio:6.1f}x  {direct_time:6.3f}s → {proxy_time:6.3f}s  {display_name}")


def main():
    """Main analysis function"""
    print("🧪 cProfile Performance Analysis")
    print("=" * 80)
    
    # Test direct API call
    direct_result, direct_time, direct_profile = run_with_profiler(
        direct_api_call, "direct_api"
    )
    
    # Small delay between tests
    time.sleep(2)
    
    # Test proxy API call
    proxy_result, proxy_time, proxy_profile = run_with_profiler(
        proxy_api_call, "proxy_api"
    )
    
    # Performance comparison
    if direct_result.get("success") and proxy_result.get("success"):
        print(f"\n🏁 PERFORMANCE COMPARISON")
        print("=" * 80)
        
        if direct_time > 0:
            slowdown_ratio = proxy_time / direct_time
            print(f"⏱️  Direct API: {direct_time:.3f}s")
            print(f"⏱️  Proxy API:  {proxy_time:.3f}s")
            print(f"🐌 Slowdown:   {slowdown_ratio:.1f}x")
        
        direct_throughput = direct_result.get('output_tokens', 0) / direct_time if direct_time > 0 else 0
        proxy_throughput = proxy_result.get('output_tokens', 0) / proxy_time if proxy_time > 0 else 0
        
        if direct_throughput > 0 and proxy_throughput > 0:
            throughput_ratio = direct_throughput / proxy_throughput
            print(f"🚀 Direct throughput: {direct_throughput:.1f} tokens/s")
            print(f"🚀 Proxy throughput:  {proxy_throughput:.1f} tokens/s")
            print(f"📉 Throughput loss:   {throughput_ratio:.1f}x")
        
        # Comparative analysis
        analyze_profile_comparison(direct_profile, proxy_profile)
    
    print(f"\n✅ Analysis complete!")
    print("📁 Generated files:")
    print("   - direct_api_profile.prof")
    print("   - direct_api_report.txt") 
    print("   - proxy_api_profile.prof")
    print("   - proxy_api_report.txt")
    print("\n💡 Use 'python -m pstats <profile_file>' for interactive analysis")


if __name__ == "__main__":
    main()