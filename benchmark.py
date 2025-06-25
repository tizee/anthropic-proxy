#!/usr/bin/env python3
"""
Performance benchmark script for Claude Code Proxy.
Tests streaming and non-streaming performance with different model providers.
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any
import httpx
import argparse
import sys
from datetime import datetime


class ProxyBenchmark:
    """Benchmark the Claude Code Proxy performance"""

    def __init__(self, base_url: str = "http://127.0.0.1:8082"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        self.results: Dict[str, List[float]] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    def sample_request(self, model: str, stream: bool = True, thinking: bool = False) -> Dict[str, Any]:
        """Generate a sample request for testing"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Write a simple Python function to calculate the factorial of a number. Explain your approach briefly."
                    }
                ]
            }
        ]

        request = {
            "model": model,
            "max_tokens": 1000,
            "messages": messages,
            "stream": stream
        }

        if thinking:
            request["thinking"] = {"type": "enabled"}

        return request

    async def benchmark_streaming_request(self, model: str, thinking: bool = False) -> Dict[str, Any]:
        """Benchmark a streaming request"""
        request_data = self.sample_request(model, stream=True, thinking=thinking)

        # 关键时间节点
        request_start_time = time.perf_counter()
        first_chunk_time = None
        first_content_time = None
        stream_end_time = None

        # 统计信息
        chunk_count = 0
        content_chunks = 0  # 只计算实际内容chunk

        # 从response中提取的usage信息
        usage_info = None

        try:
            # Add a small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)

            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:

                if response.status_code != 200:
                    response_body = await response.aread()
                    raise Exception(f"HTTP {response.status_code}: {response_body.decode() if response_body else 'No response body'}")

                async for chunk in response.aiter_text():
                    chunk_count += 1

                    # 记录第一个chunk的时间
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()

                    # 解析chunk以获取usage信息和内容信息 - 支持SSE格式
                    if chunk.strip():
                        # 解析Server-Sent Events格式
                        lines = chunk.strip().split('\n')
                        current_event = None
                        current_data = None
                        
                        for line in lines:
                            if line.startswith('event: '):
                                current_event = line[7:].strip()
                            elif line.startswith('data: '):
                                current_data = line[6:].strip()
                                
                                if current_data == "[DONE]":
                                    break
                                
                                try:
                                    data = json.loads(current_data)
                                    
                                    # 检查是否是内容delta
                                    if (data.get("type") == "content_block_delta" and
                                        data.get("delta", {}).get("type") == "text_delta"):
                                        content_chunks += 1
                                        if first_content_time is None:
                                            first_content_time = time.perf_counter()

                                    # 提取usage信息 - message_delta事件中包含usage
                                    elif data.get("type") == "message_delta":
                                        # 根据SSE格式，usage在data的顶层
                                        if "usage" in data:
                                            usage_info = data["usage"]

                                except json.JSONDecodeError:
                                    pass
                        
                        if current_data == "[DONE]":
                            break

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "thinking": thinking
            }

        stream_end_time = time.perf_counter()

        # 计算关键指标
        total_duration = stream_end_time - request_start_time
        time_to_first_chunk = first_chunk_time - request_start_time if first_chunk_time else 0
        time_to_first_content = first_content_time - request_start_time if first_content_time else 0

        # 从usage中获取token信息
        output_tokens = usage_info.get("output_tokens", 0) if usage_info else 0

        result = {
            "success": True,
            "model": model,
            "thinking": thinking,
            "stream": True,
            "request_start_time": request_start_time,
            "stream_end_time": stream_end_time,
            "total_duration": total_duration,
            "time_to_first_chunk": time_to_first_chunk,
            "time_to_first_content": time_to_first_content,
            "chunk_count": chunk_count,
            "content_chunks": content_chunks,
            "output_tokens": output_tokens,
            "chunks_per_sec": chunk_count / total_duration if total_duration > 0 else 0,
            "content_chunks_per_sec": content_chunks / total_duration if total_duration > 0 else 0,
            "tokens_per_sec": output_tokens / total_duration if total_duration > 0 else 0,
        }

        if usage_info:
            result["usage"] = usage_info

        return result

    async def benchmark_non_streaming_request(self, model: str, thinking: bool = False) -> Dict[str, Any]:
        """Benchmark a non-streaming request"""
        request_data = self.sample_request(model, stream=False, thinking=thinking)

        # 关键时间节点
        request_start_time = time.perf_counter()
        response_received_time = None

        try:
            # Add a small delay to avoid overwhelming the server
            await asyncio.sleep(0.1)

            response = await self.client.post(
                f"{self.base_url}/v1/messages",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )

            response_received_time = time.perf_counter()

            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

            result = response.json()

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "thinking": thinking
            }

        request_end_time = time.perf_counter()

        # 计算关键指标
        total_duration = request_end_time - request_start_time
        response_time = response_received_time - request_start_time if response_received_time else 0

        # 从response中直接获取usage信息
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        return {
            "success": True,
            "model": model,
            "thinking": thinking,
            "stream": False,
            "request_start_time": request_start_time,
            "response_received_time": response_received_time,
            "request_end_time": request_end_time,
            "total_duration": total_duration,
            "response_time": response_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tokens_per_sec": output_tokens / total_duration if total_duration > 0 else 0,
            "usage": usage,
        }

    async def run_benchmark_suite(self, models: List[str], iterations: int = 3) -> Dict[str, Any]:
        """Run a complete benchmark suite"""
        print(f"🚀 Starting Claude Code Proxy Benchmark")
        print(f"📊 Testing {len(models)} models with {iterations} iterations each")
        print(f"🔗 Proxy URL: {self.base_url}")
        print("-" * 60)

        all_results = []

        for model in models:
            print(f"\n📋 Testing model: {model}")

            # Test streaming
            streaming_results = []
            for i in range(iterations):
                print(f"  🔄 Streaming test {i+1}/{iterations}...", end=" ")
                result = await self.benchmark_streaming_request(model)
                streaming_results.append(result)
                if result["success"]:
                    tokens_info = f"{result.get('tokens_per_sec', 0):.1f} tokens/s" if result.get('output_tokens', 0) > 0 else "N/A"
                    ttfc = result.get('time_to_first_chunk', 0)
                    print(f"✅ {result['total_duration']:.2f}s ({tokens_info}, TTFC: {ttfc:.3f}s)")
                else:
                    print(f"❌ {result['error']}")

            # Test non-streaming
            non_streaming_results = []
            for i in range(iterations):
                print(f"  📄 Non-streaming test {i+1}/{iterations}...", end=" ")
                result = await self.benchmark_non_streaming_request(model)
                non_streaming_results.append(result)
                if result["success"]:
                    tokens_info = f"{result.get('tokens_per_sec', 0):.1f} tokens/s"
                    response_time = result.get('response_time', 0)
                    print(f"✅ {result['total_duration']:.2f}s ({tokens_info}, RT: {response_time:.2f}s)")
                else:
                    print(f"❌ {result['error']}")

            all_results.extend(streaming_results)
            all_results.extend(non_streaming_results)

        return {
            "timestamp": datetime.now().isoformat(),
            "proxy_url": self.base_url,
            "models": models,
            "iterations": iterations,
            "results": all_results
        }

    def analyze_results(self, benchmark_data: Dict[str, Any]) -> None:
        """Analyze and display benchmark results"""
        results = benchmark_data["results"]
        successful_results = [r for r in results if r["success"]]

        if not successful_results:
            print("❌ No successful results to analyze!")
            return

        print(f"\n📈 BENCHMARK RESULTS ANALYSIS")
        print("=" * 60)

        # Group by model and stream type
        by_model_stream = {}
        for result in successful_results:
            key = f"{result['model']}-{'stream' if result['stream'] else 'non-stream'}"
            if key not in by_model_stream:
                by_model_stream[key] = []
            by_model_stream[key].append(result)

        for key, group_results in by_model_stream.items():
            if key.endswith('-stream'):
                model = key[:-7]  # Remove '-stream'
                stream_type = 'stream'
            elif key.endswith('-non-stream'):
                model = key[:-11]  # Remove '-non-stream'
                stream_type = 'non-stream'
            else:
                model, stream_type = key.split('-', 1)

            print(f"\n🎯 {model} ({stream_type})")
            print("-" * 40)

            durations = [r['total_duration'] for r in group_results]
            if stream_type == "stream":
                # 流式请求的关键指标
                token_throughput = [r.get('tokens_per_sec', 0) for r in group_results if r.get('tokens_per_sec', 0) > 0]
                ttfc = [r.get('time_to_first_chunk', 0) for r in group_results]
                ttfct = [r.get('time_to_first_content', 0) for r in group_results]  # Time to first content token
                content_chunk_rates = [r.get('content_chunks_per_sec', 0) for r in group_results]
                output_tokens = [r.get('output_tokens', 0) for r in group_results]

                print(f"⏱️  Total Duration: avg={statistics.mean(durations):.2f}s, "
                      f"min={min(durations):.2f}s, max={max(durations):.2f}s")

                if token_throughput:
                    print(f"🚀 Token Throughput: avg={statistics.mean(token_throughput):.1f} tokens/s, "
                          f"min={min(token_throughput):.1f}, max={max(token_throughput):.1f}")

                if ttfc:
                    print(f"⚡ Time to First Chunk: avg={statistics.mean(ttfc):.3f}s, "
                          f"min={min(ttfc):.3f}s, max={max(ttfc):.3f}s")

                if ttfct and any(t > 0 for t in ttfct):
                    print(f"📝 Time to First Content: avg={statistics.mean([t for t in ttfct if t > 0]):.3f}s")

                if content_chunk_rates:
                    print(f"📦 Content Chunk Rate: avg={statistics.mean(content_chunk_rates):.1f}/s")

                if output_tokens:
                    print(f"📊 Output Tokens: avg={statistics.mean(output_tokens):.0f}, "
                          f"min={min(output_tokens)}, max={max(output_tokens)}")

            else:
                # 非流式请求的关键指标
                token_throughput = [r.get('tokens_per_sec', 0) for r in group_results if r.get('tokens_per_sec', 0) > 0]
                response_times = [r.get('response_time', 0) for r in group_results]
                output_tokens = [r.get('output_tokens', 0) for r in group_results]
                input_tokens = [r.get('input_tokens', 0) for r in group_results]

                print(f"⏱️  Total Duration: avg={statistics.mean(durations):.2f}s, "
                      f"min={min(durations):.2f}s, max={max(durations):.2f}s")

                if response_times:
                    print(f"📡 Response Time: avg={statistics.mean(response_times):.2f}s, "
                          f"min={min(response_times):.2f}s, max={max(response_times):.2f}s")

                if token_throughput:
                    print(f"🚀 Token Throughput: avg={statistics.mean(token_throughput):.1f} tokens/s, "
                          f"min={min(token_throughput):.1f}, max={max(token_throughput):.1f}")

                if output_tokens:
                    print(f"📊 Output Tokens: avg={statistics.mean(output_tokens):.0f}, "
                          f"min={min(output_tokens)}, max={max(output_tokens)}")

                if input_tokens:
                    print(f"📥 Input Tokens: avg={statistics.mean(input_tokens):.0f}, "
                          f"min={min(input_tokens)}, max={max(input_tokens)}")

        # Error summary
        failed_results = [r for r in results if not r["success"]]
        if failed_results:
            print(f"\n❌ ERRORS ({len(failed_results)} failures)")
            print("-" * 40)
            error_counts = {}
            for result in failed_results:
                error = result['error']
                if error not in error_counts:
                    error_counts[error] = 0
                error_counts[error] += 1

            for error, count in error_counts.items():
                print(f"  {count}x: {error}")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Claude Code Proxy performance")
    parser.add_argument("--url", "-u", default="http://127.0.0.1:8082",
                       help="Proxy server URL (default: http://127.0.0.1:8082)")
    parser.add_argument("--models", "-m", nargs="+",
                       default=["doubao-seed-1.6", "gemini-2.5-flash-lite-preview-06-17"],
                       help="Models to test (default: doubao-seed-1.6 gemini-2.5-flash-lite-preview-06-17)")
    parser.add_argument("--iterations", "-i", type=int, default=3,
                       help="Number of iterations per test (default: 3)")
    parser.add_argument("--output", "-o", type=str,
                       help="Save results to JSON file")
    parser.add_argument("--performance-endpoint", "-p", action="store_true",
                       help="Also fetch and display performance metrics from /v1/performance")

    args = parser.parse_args()

    async with ProxyBenchmark(args.url) as benchmark:
        # Run benchmarks
        results = await benchmark.run_benchmark_suite(args.models, args.iterations)

        # Analyze results
        benchmark.analyze_results(results)

        # Fetch performance metrics if requested
        if args.performance_endpoint:
            try:
                perf_response = await benchmark.client.get(f"{args.url}/v1/performance")
                if perf_response.status_code == 200:
                    perf_data = perf_response.json()
                    print(f"\n🔧 PROXY PERFORMANCE METRICS")
                    print("=" * 60)
                    for operation, stats in perf_data.items():
                        print(f"{operation}: avg={stats['avg']:.3f}s, count={stats['count']}")
                else:
                    print(f"⚠️  Could not fetch performance metrics: HTTP {perf_response.status_code}")
            except Exception as e:
                print(f"⚠️  Error fetching performance metrics: {e}")

        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
