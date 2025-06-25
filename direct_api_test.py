#!/usr/bin/env python3
"""
Direct API performance test - bypasses proxy to measure raw API performance.
"""

import asyncio
import json
import os
import time
import yaml
from typing import Dict, Any
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (reusing server logic)
load_dotenv()


def load_model_config(model_id: str) -> Dict[str, Any]:
    """Load model configuration from models.yaml - reusing server logic"""
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


async def test_direct_openai_streaming():
    """Test streaming request directly via OpenAI SDK using model config"""
    
    model_id = "gemini-2.5-flash-lite-preview-06-17"
    
    try:
        # Load model configuration from models.yaml
        model_config = load_model_config(model_id)
        print(f"📋 Loaded config for {model_id}")
        
        # Get API key
        api_key_name = model_config.get("api_key_name")
        if not api_key_name:
            print(f"❌ No api_key_name specified for {model_id}")
            return
        
        api_key = os.environ.get(api_key_name)
        if not api_key:
            print(f"❌ {api_key_name} not found in environment")
            return
        
        # Create OpenAI client with exact config from models.yaml
        api_base = model_config.get("api_base")
        model_name = model_config.get("model_name", model_id)
        extra_headers = model_config.get("extra_headers", {})
        
        client = AsyncOpenAI(
            base_url=api_base,
            api_key=api_key,
            default_headers=extra_headers
        )
        
        print(f"🔗 API Base: {api_base}")
        print(f"🎯 Model Name: {model_name}")
        print(f"🔑 API Key: {api_key_name}")
        
    except Exception as e:
        print(f"❌ Error loading model config: {e}")
        return
    
    messages = [
        {
            "role": "user", 
            "content": "Write a simple Python function to calculate the factorial of a number. Explain your approach briefly."
        }
    ]
    
    # Track timing and tokens
    request_start = time.perf_counter()
    first_chunk_time = None
    chunk_count = 0
    content_chunks = 0
    accumulated_content = ""
    
    print("🚀 Testing direct OpenAI SDK")
    print("-" * 60)
    
    try:
        # Get max_tokens from config
        max_tokens = model_config.get("max_tokens", 1000)
        
        # Create streaming request using exact model_name from config
        stream = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            store=False,
            stream_options={"include_usage": True}
        )
        
        usage_info = None
        
        async for chunk in stream:
            chunk_count += 1
            
            # Record first chunk time
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            
            # Check for content
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                if choice.delta and choice.delta.content:
                    content_chunks += 1
                    accumulated_content += choice.delta.content
                    
                    # Print first few content chunks
                    if content_chunks <= 5:
                        print(f"Content chunk {content_chunks}: {repr(choice.delta.content)}")
            
            # Check for usage (final chunk before [DONE])
            if chunk.usage:
                usage_info = chunk.usage
                print(f"📊 Usage received: {usage_info}")
        
        request_end = time.perf_counter()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    finally:
        await client.close()
    
    # Calculate metrics
    total_duration = request_end - request_start
    ttfc = first_chunk_time - request_start if first_chunk_time else 0
    
    output_tokens = usage_info.completion_tokens if usage_info else len(accumulated_content) // 4
    token_throughput = output_tokens / total_duration if total_duration > 0 else 0
    
    print(f"\n📈 DIRECT API RESULTS")
    print("=" * 50)
    print(f"⏱️  Total Duration: {total_duration:.3f}s")
    print(f"⚡ Time to First Chunk: {ttfc:.3f}s") 
    print(f"📦 Total Chunks: {chunk_count}")
    print(f"📝 Content Chunks: {content_chunks}")
    print(f"📊 Output Tokens: {output_tokens}")
    print(f"🚀 Token Throughput: {token_throughput:.1f} tokens/s")
    print(f"📄 Content Length: {len(accumulated_content)} chars")
    
    return {
        "total_duration": total_duration,
        "ttfc": ttfc,
        "chunk_count": chunk_count,
        "content_chunks": content_chunks,
        "output_tokens": output_tokens,
        "token_throughput": token_throughput,
        "content_length": len(accumulated_content)
    }


async def test_proxy_streaming():
    """Test streaming request via our proxy for comparison"""
    
    model_id = "gemini-2.5-flash-lite-preview-06-17"  # Use model_id for proxy
    
    try:
        # Load model configuration to get max_tokens
        model_config = load_model_config(model_id)
        max_tokens = model_config.get("max_tokens", 1000)
        print(f"📋 Using model_id: {model_id} (proxy alias)")
        print(f"📋 Max tokens from config: {max_tokens}")
        
    except Exception as e:
        print(f"❌ Error loading model config: {e}")
        max_tokens = 1000
    
    url = "http://127.0.0.1:8082/v1/messages"
    request_data = {
        "model": model_id,  # Proxy uses model_id (alias)
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
    
    # Track timing and tokens
    request_start = time.perf_counter()
    first_chunk_time = None
    chunk_count = 0
    content_chunks = 0
    usage_info = None
    
    print(f"\n🔄 Testing via Proxy → OpenRouter → Gemini 2.5 Flash Lite")
    print(f"🔗 Proxy URL: {url}")
    print("-" * 60)
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status_code != 200:
                    print(f"❌ HTTP {response.status_code}")
                    return None
                
                async for chunk in response.aiter_text():
                    chunk_count += 1
                    
                    # Record first chunk time
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                    
                    if chunk.strip():
                        # Parse SSE format
                        lines = chunk.strip().split('\\n')
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
                                    event_type = data.get("type", "unknown")
                                    
                                    # Count content chunks
                                    if (event_type == "content_block_delta" and 
                                        data.get("delta", {}).get("type") == "text_delta"):
                                        content_chunks += 1
                                        if content_chunks <= 5:
                                            print(f"Content chunk {content_chunks}: {repr(data.get('delta', {}).get('text', ''))}")
                                    
                                    # Extract usage info
                                    elif event_type == "message_delta" and "usage" in data:
                                        usage_info = data["usage"]
                                        print(f"📊 Usage received: {usage_info}")
                                
                                except json.JSONDecodeError:
                                    pass
                        
                        if current_data == "[DONE]":
                            break
        
        request_end = time.perf_counter()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    # Calculate metrics
    total_duration = request_end - request_start
    ttfc = first_chunk_time - request_start if first_chunk_time else 0
    
    output_tokens = usage_info.get("output_tokens", 0) if usage_info else 0
    token_throughput = output_tokens / total_duration if total_duration > 0 and output_tokens > 0 else 0
    
    print(f"\n📈 PROXY RESULTS")
    print("=" * 50)
    print(f"⏱️  Total Duration: {total_duration:.3f}s")
    print(f"⚡ Time to First Chunk: {ttfc:.3f}s")
    print(f"📦 Total Chunks: {chunk_count}")
    print(f"📝 Content Chunks: {content_chunks}")
    print(f"📊 Output Tokens: {output_tokens}")
    print(f"🚀 Token Throughput: {token_throughput:.1f} tokens/s")
    
    return {
        "total_duration": total_duration,
        "ttfc": ttfc, 
        "chunk_count": chunk_count,
        "content_chunks": content_chunks,
        "output_tokens": output_tokens,
        "token_throughput": token_throughput
    }


async def main():
    """Run performance comparison test"""
    print("🧪 PERFORMANCE COMPARISON TEST")
    print("=" * 70)
    
    # Test direct API
    direct_result = await test_direct_openai_streaming()
    
    # Small delay between tests
    await asyncio.sleep(2)
    
    # Test via proxy  
    proxy_result = await test_proxy_streaming()
    
    if direct_result and proxy_result:
        print(f"\n🏁 COMPARISON SUMMARY")
        print("=" * 70)
        
        # Calculate performance ratios
        ttfc_ratio = proxy_result["ttfc"] / direct_result["ttfc"] if direct_result["ttfc"] > 0 else 0
        duration_ratio = proxy_result["total_duration"] / direct_result["total_duration"]
        
        if direct_result["token_throughput"] > 0 and proxy_result["token_throughput"] > 0:
            throughput_ratio = direct_result["token_throughput"] / proxy_result["token_throughput"]
            print(f"🚀 Token Throughput:")
            print(f"   Direct: {direct_result['token_throughput']:.1f} tokens/s")
            print(f"   Proxy:  {proxy_result['token_throughput']:.1f} tokens/s")  
            print(f"   Ratio:  {throughput_ratio:.1f}x slower via proxy")
        
        print(f"⚡ Time to First Chunk:")
        print(f"   Direct: {direct_result['ttfc']:.3f}s")
        print(f"   Proxy:  {proxy_result['ttfc']:.3f}s")
        print(f"   Ratio:  {ttfc_ratio:.1f}x slower via proxy")
        
        print(f"⏱️  Total Duration:")
        print(f"   Direct: {direct_result['total_duration']:.3f}s")
        print(f"   Proxy:  {proxy_result['total_duration']:.3f}s")
        print(f"   Ratio:  {duration_ratio:.1f}x slower via proxy")


if __name__ == "__main__":
    asyncio.run(main())