#!/usr/bin/env python3
"""
Test the performance improvement from the optimized streaming conversion.
"""

import asyncio
import time
from profile_server_workflow import run_with_profiler, test_server_workflow

async def quick_performance_test():
    """Quick test to verify optimization works"""
    print("🚀 Testing optimized server performance...")
    
    start_time = time.perf_counter()
    result = await test_server_workflow()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    
    print(f"⏱️  Execution time: {total_time:.3f}s")
    print(f"✅ Result: {result}")
    
    if result.get("success"):
        content_chunks = result.get("content_chunks", 0)
        if total_time > 0 and content_chunks > 0:
            throughput = content_chunks / total_time
            print(f"📦 Content chunks: {content_chunks}")
            print(f"🚀 Chunk throughput: {throughput:.1f} chunks/s")
    
    return total_time

async def main():
    """Run performance tests"""
    print("🧪 OPTIMIZATION VERIFICATION TEST")
    print("=" * 60)
    
    # Run 3 quick tests to get average performance
    times = []
    for i in range(3):
        print(f"\n🔄 Test run {i+1}/3:")
        test_time = await quick_performance_test()
        times.append(test_time)
        
        # Wait between tests
        if i < 2:
            await asyncio.sleep(2)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"🕐 Average time: {avg_time:.3f}s")
    print(f"🕐 Best time:    {min_time:.3f}s")
    print(f"🕐 Worst time:   {max_time:.3f}s")
    
    # Compare with baseline
    baseline_time = 24.0  # Previous unoptimized time
    if avg_time < baseline_time:
        improvement = baseline_time / avg_time
        print(f"🎉 IMPROVEMENT: {improvement:.1f}x faster than baseline!")
        print(f"⚡ Time saved: {baseline_time - avg_time:.1f}s per request")
    else:
        print(f"⚠️  Performance regression detected")
    
    return avg_time

if __name__ == "__main__":
    asyncio.run(main())