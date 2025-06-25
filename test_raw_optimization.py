#!/usr/bin/env python3
"""
Test the ultimate raw streaming optimization.
"""

import asyncio
import os
import time
import yaml
from raw_streaming_optimizer import convert_openai_streaming_response_to_anthropic_raw
from dotenv import load_dotenv

load_dotenv()

def load_model_config(model_id: str):
    """Load model configuration from models.yaml"""
    with open("models.yaml") as f:
        models = yaml.safe_load(f)
    
    for model in models:
        if model.get("model_id") == model_id:
            return model
    
    raise ValueError(f"Model '{model_id}' not found")

async def test_raw_streaming():
    """Test the raw streaming optimization"""
    
    model_id = "gemini-2.5-flash-lite-preview-06-17"
    model_config = load_model_config(model_id)
    
    # Prepare OpenAI request parameters
    openai_params = {
        "model": model_config["model_name"],
        "messages": [
            {
                "role": "user",
                "content": "Write a simple Python function to calculate the factorial of a number."
            }
        ],
        "max_tokens": 1000,
        "stream": True,
        "store": False,
        "stream_options": {"include_usage": True},
        # Connection parameters
        "api_base": model_config["api_base"],
        "api_key": os.environ[model_config["api_key_name"]],
        "extra_headers": model_config.get("extra_headers", {})
    }
    
    print("🚀 Testing raw streaming optimization...")
    
    start_time = time.perf_counter()
    chunk_count = 0
    content_chunks = 0
    
    try:
        async for event in convert_openai_streaming_response_to_anthropic_raw(
            openai_params, model_id
        ):
            chunk_count += 1
            if "content_block_delta" in event:
                content_chunks += 1
                if content_chunks <= 5:
                    # Show first few content chunks
                    print(f"Content chunk {content_chunks}: {event[:100]}...")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    print(f"✅ Raw streaming test completed")
    print(f"⏱️  Total time: {total_time:.3f}s")
    print(f"📦 Total chunks: {chunk_count}")
    print(f"📝 Content chunks: {content_chunks}")
    
    if total_time > 0 and content_chunks > 0:
        throughput = content_chunks / total_time
        print(f"🚀 Content throughput: {throughput:.1f} chunks/s")
    
    return {
        "total_time": total_time,
        "chunk_count": chunk_count,
        "content_chunks": content_chunks
    }

async def main():
    """Run the test"""
    print("🧪 RAW STREAMING OPTIMIZATION TEST")
    print("=" * 60)
    
    result = await test_raw_streaming()
    
    if result:
        baseline_time = 25.0  # Previous optimized time
        if result["total_time"] < baseline_time:
            improvement = baseline_time / result["total_time"]
            print(f"\n🎉 MAJOR IMPROVEMENT: {improvement:.1f}x faster!")
            print(f"⚡ Time saved: {baseline_time - result['total_time']:.1f}s")
        else:
            print(f"\n⚠️  No significant improvement")

if __name__ == "__main__":
    asyncio.run(main())