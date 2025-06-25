#!/usr/bin/env python3
"""
Direct profiling of server.py create_message workflow.
Tests the proxy server's core logic without HTTP overhead.
"""

import asyncio
import cProfile
import pstats
import io
import time
from typing import Dict, Any
from pstats import SortKey
from fastapi import Request
from unittest.mock import Mock

# Import server components
from server import create_message, initialize_custom_models
from models import ClaudeMessagesRequest


def create_mock_request(request_data: Dict[str, Any]) -> Request:
    """Create a mock FastAPI Request object"""
    import json
    
    mock_request = Mock(spec=Request)
    
    # Mock the body method to return the JSON data
    async def mock_body():
        return json.dumps(request_data).encode('utf-8')
    
    mock_request.body = mock_body
    mock_request.url.path = "/v1/messages"
    
    return mock_request


async def test_server_workflow():
    """Test the complete server workflow for create_message"""
    
    # Initialize custom models (this happens on server startup)
    initialize_custom_models()
    
    # Create a test request matching the format expected by ClaudeMessagesRequest
    request_data = {
        "model": "gemini-2.5-flash-lite-preview-06-17",
        "max_tokens": 1000,
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
    
    # Create the Pydantic request model
    claude_request = ClaudeMessagesRequest(**request_data)
    
    # Create mock FastAPI request
    raw_request = create_mock_request(request_data)
    
    # Call the server's create_message function directly
    try:
        response = await create_message(claude_request, raw_request)
        
        # If it's a streaming response, consume it to measure full processing time
        if hasattr(response, 'body_iterator'):
            content_chunks = 0
            async for chunk in response.body_iterator:
                if chunk:
                    content_chunks += 1
            
            return {
                "success": True,
                "content_chunks": content_chunks,
                "response_type": "streaming"
            }
        else:
            return {
                "success": True,
                "response_type": "non-streaming",
                "response": response
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


async def test_non_streaming_workflow():
    """Test non-streaming workflow for comparison"""
    
    # Initialize custom models
    initialize_custom_models()
    
    # Create a non-streaming test request
    request_data = {
        "model": "gemini-2.5-flash-lite-preview-06-17", 
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write a simple Python function to calculate the factorial of a number."
                    }
                ]
            }
        ],
        "stream": False
    }
    
    # Create the Pydantic request model
    claude_request = ClaudeMessagesRequest(**request_data)
    
    # Create mock FastAPI request
    raw_request = create_mock_request(request_data)
    
    try:
        response = await create_message(claude_request, raw_request)
        return {
            "success": True,
            "response_type": "non-streaming",
            "response": response
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def run_with_profiler(async_func, profile_name):
    """Run an async function with cProfile and detailed analysis"""
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
    profile_file = f"{profile_name}_server_profile.prof"
    pr.dump_stats(profile_file)
    
    print(f"⏱️  Total execution time: {total_time:.3f}s")
    print(f"✅ Result: {result}")
    
    # Generate analysis
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    
    # Print top functions by cumulative time
    print(f"\n📈 Top 15 functions by cumulative time ({profile_name}):")
    print("=" * 90)
    ps.sort_stats(SortKey.CUMULATIVE).print_stats(15)
    
    # Print functions with high internal time
    print(f"\n⚡ Top 10 functions by internal time ({profile_name}):")
    print("=" * 90)
    ps.sort_stats(SortKey.TIME).print_stats(10)
    
    # Look for specific bottlenecks
    print(f"\n🔍 JSON/serialization functions ({profile_name}):")
    print("-" * 60)
    ps.sort_stats(SortKey.CUMULATIVE).print_stats('.*json.*')
    
    print(f"\n🔍 Model conversion functions ({profile_name}):")
    print("-" * 60)
    ps.sort_stats(SortKey.CUMULATIVE).print_stats('.*convert.*|.*models\.py.*')
    
    print(f"\n🔍 OpenAI SDK functions ({profile_name}):")
    print("-" * 60)
    ps.sort_stats(SortKey.CUMULATIVE).print_stats('.*openai.*')
    
    # Save detailed report
    report_file = f"{profile_name}_server_report.txt"
    with open(report_file, 'w') as f:
        ps_file = pstats.Stats(pr, stream=f)
        ps_file.sort_stats(SortKey.CUMULATIVE).print_stats()
        f.write("\n" + "="*80 + "\n")
        f.write("FUNCTIONS BY INTERNAL TIME:\n")
        f.write("="*80 + "\n")
        ps_file.sort_stats(SortKey.TIME).print_stats()
        f.write("\n" + "="*80 + "\n")
        f.write("JSON/SERIALIZATION FUNCTIONS:\n") 
        f.write("="*80 + "\n")
        ps_file.sort_stats(SortKey.CUMULATIVE).print_stats('.*json.*')
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL CONVERSION FUNCTIONS:\n")
        f.write("="*80 + "\n")
        ps_file.sort_stats(SortKey.CUMULATIVE).print_stats('.*convert.*|.*models\.py.*')
    
    print(f"📄 Detailed report saved to: {report_file}")
    print(f"📊 Profile data saved to: {profile_file}")
    
    return result, total_time, profile_file


def analyze_server_bottlenecks(streaming_profile, non_streaming_profile):
    """Analyze differences between streaming and non-streaming workflows"""
    print(f"\n🔬 SERVER WORKFLOW ANALYSIS")
    print("=" * 80)
    
    # Load both profiles
    streaming_stats = pstats.Stats(streaming_profile)
    non_streaming_stats = pstats.Stats(non_streaming_profile)
    
    # Get function statistics
    streaming_functions = streaming_stats.get_stats_profile()
    non_streaming_functions = non_streaming_stats.get_stats_profile()
    
    # Find streaming-specific overhead
    streaming_only = []
    for func_name in streaming_functions.func_profiles:
        if func_name not in non_streaming_functions.func_profiles:
            func_profile = streaming_functions.func_profiles[func_name]
            if func_profile.cumtime > 0.001:  # Only significant functions
                streaming_only.append((func_name, func_profile.cumtime))
    
    streaming_only.sort(key=lambda x: x[1], reverse=True)
    
    print(f"🎯 Streaming-specific functions (overhead sources):")
    print("-" * 80)
    for func_name, cumtime in streaming_only[:10]:
        display_name = str(func_name).split('.')[-1] if '.' in str(func_name) else str(func_name)
        print(f"  {cumtime:8.4f}s  {display_name}")
    
    # Find functions with significantly different times
    print(f"\n📊 Functions with different performance patterns:")
    print("-" * 80)
    time_differences = []
    for func_name in non_streaming_functions.func_profiles:
        if func_name in streaming_functions.func_profiles:
            non_streaming_time = non_streaming_functions.func_profiles[func_name].cumtime
            streaming_time = streaming_functions.func_profiles[func_name].cumtime
            if non_streaming_time > 0.001:  # Only significant functions
                if streaming_time > non_streaming_time * 1.5:  # 50% or more difference
                    ratio = streaming_time / non_streaming_time if non_streaming_time > 0 else float('inf')
                    time_differences.append((func_name, non_streaming_time, streaming_time, ratio))
    
    time_differences.sort(key=lambda x: x[3], reverse=True)
    for func_name, non_streaming_time, streaming_time, ratio in time_differences[:8]:
        display_name = str(func_name).split('.')[-1] if '.' in str(func_name) else str(func_name)
        print(f"  {ratio:6.1f}x  {non_streaming_time:6.3f}s → {streaming_time:6.3f}s  {display_name}")


def main():
    """Main profiling function"""
    print("🧪 SERVER WORKFLOW cProfile ANALYSIS")
    print("=" * 80)
    
    # Test streaming workflow
    streaming_result, streaming_time, streaming_profile = run_with_profiler(
        test_server_workflow, "streaming"
    )
    
    # Small delay between tests
    print("\n" + "⏳ Waiting 2 seconds between tests...")
    time.sleep(2)
    
    # Test non-streaming workflow
    non_streaming_result, non_streaming_time, non_streaming_profile = run_with_profiler(
        test_non_streaming_workflow, "non_streaming"
    )
    
    # Performance comparison
    if (streaming_result.get("success") and non_streaming_result.get("success")):
        print(f"\n🏁 WORKFLOW PERFORMANCE COMPARISON")
        print("=" * 80)
        
        if non_streaming_time > 0:
            slowdown_ratio = streaming_time / non_streaming_time
            print(f"⏱️  Non-streaming: {non_streaming_time:.3f}s")
            print(f"⏱️  Streaming:     {streaming_time:.3f}s")
            print(f"🐌 Streaming overhead: {slowdown_ratio:.1f}x")
        
        # Detailed bottleneck analysis
        analyze_server_bottlenecks(streaming_profile, non_streaming_profile)
    else:
        print(f"\n❌ One or both workflows failed:")
        print(f"   Streaming: {streaming_result}")
        print(f"   Non-streaming: {non_streaming_result}")
    
    print(f"\n✅ Server workflow analysis complete!")
    print("📁 Generated files:")
    print("   - streaming_server_profile.prof")
    print("   - streaming_server_report.txt")
    print("   - non_streaming_server_profile.prof") 
    print("   - non_streaming_server_report.txt")
    print("\n💡 Use 'python -m pstats <profile_file>' for interactive analysis")


if __name__ == "__main__":
    main()