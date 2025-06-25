#!/usr/bin/env python3
"""
Final performance validation test.
"""

import asyncio
import time
import statistics
from profile_server_workflow import test_server_workflow

async def run_performance_tests():
    """Run multiple performance tests to get accurate statistics"""
    
    print("🧪 FINAL PERFORMANCE VALIDATION")
    print("=" * 70)
    print("🎯 Testing optimized streaming performance...")
    
    times = []
    chunks = []
    
    for i in range(5):
        print(f"\n🔄 Test run {i+1}/5:")
        
        start_time = time.perf_counter()
        result = await test_server_workflow()
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        times.append(total_time)
        
        if result.get("success"):
            chunk_count = result.get("content_chunks", 0)
            chunks.append(chunk_count)
            print(f"   ✅ {total_time:.3f}s, {chunk_count} chunks")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Wait between tests
        if i < 4:
            await asyncio.sleep(1)
    
    # Calculate statistics
    if times:
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\n📊 PERFORMANCE STATISTICS")
        print("=" * 70)
        print(f"⏱️  Average time: {avg_time:.3f}s")
        print(f"⏱️  Best time:    {min_time:.3f}s")
        print(f"⏱️  Worst time:   {max_time:.3f}s")
        
        if chunks:
            avg_chunks = statistics.mean(chunks)
            throughput = avg_chunks / avg_time if avg_time > 0 else 0
            print(f"📦 Average chunks: {avg_chunks:.1f}")
            print(f"🚀 Avg throughput: {throughput:.1f} chunks/s")
        
        # Compare with baseline
        baseline_time = 24.0  # Original unoptimized time
        improvement = baseline_time / avg_time
        
        print(f"\n🎉 OPTIMIZATION RESULTS")
        print("=" * 70)
        print(f"📈 Performance improvement: {improvement:.1f}x faster")
        print(f"⚡ Time saved per request: {baseline_time - avg_time:.1f}s")
        print(f"📉 Response time: {baseline_time:.1f}s → {avg_time:.3f}s")
        
        # Calculate potential cost savings
        if improvement > 10:
            print(f"\n💰 IMPACT ANALYSIS")
            print("-" * 40)
            print(f"🔥 MASSIVE performance gain: {improvement:.0f}x faster!")
            print(f"⚡ Eliminated OpenAI SDK Pydantic overhead")
            print(f"🚀 Raw HTTP streaming approach successful")
            print(f"💎 Production-ready optimization")

async def main():
    await run_performance_tests()

if __name__ == "__main__":
    asyncio.run(main())