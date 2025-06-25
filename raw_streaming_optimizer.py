#!/usr/bin/env python3
"""
Ultimate performance optimization: Raw SSE processing without OpenAI SDK overhead.
This bypasses ALL OpenAI SDK Pydantic operations for maximum performance.
"""

import asyncio
import json
import time
import uuid
import logging
from typing import AsyncGenerator, Dict, Any
import httpx

logger = logging.getLogger(__name__)

# Fast JSON operations
try:
    import orjson
    def fast_json_dumps(obj) -> str:
        return orjson.dumps(obj).decode('utf-8')
    def fast_json_loads(s: str):
        return orjson.loads(s)
except ImportError:
    def fast_json_dumps(obj) -> str:
        return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)
    def fast_json_loads(s: str):
        return json.loads(s)

# Cached SSE event templates
SSE_TEMPLATES = {
    'message_start': 'event: message_start\ndata: {{"type":"message_start","message":{{"id":"{message_id}","type":"message","role":"assistant","model":"{model}","content":[],"stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":0,"output_tokens":0}}}}}}\n\n',
    'ping': 'event: ping\ndata: {"type":"ping"}\n\n',
    'content_block_start': 'event: content_block_start\ndata: {{"type":"content_block_start","index":{index},"content_block":{{"type":"text","text":""}}}}\n\n',
    'content_block_delta': 'event: content_block_delta\ndata: {{"type":"content_block_delta","index":{index},"delta":{{"type":"text_delta","text":"{text}"}}}}\n\n',
    'content_block_stop': 'event: content_block_stop\ndata: {{"type":"content_block_stop","index":{index}}}\n\n',
    'message_delta': 'event: message_delta\ndata: {{"type":"message_delta","delta":{{"stop_reason":"{stop_reason}","stop_sequence":null}},"usage":{{"output_tokens":{output_tokens}}}}}\n\n',
    'message_stop': 'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    'done': 'data: [DONE]\n\n'
}

def escape_json_string(text: str) -> str:
    """Fast JSON string escaping"""
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

class RawStreamProcessor:
    """Ultra-fast streaming processor that operates on raw HTTP data"""
    
    def __init__(self, model: str):
        self.model = model
        self.accumulated_text = ""
        self.input_tokens = 0
        self.output_tokens = 0
        self.content_block_started = False
        self.content_block_index = 0
        
    async def process_raw_openai_stream(self, openai_request_params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Process OpenAI streaming response at the raw HTTP level.
        This completely bypasses OpenAI SDK's expensive Pydantic operations.
        """
        
        start_time = time.perf_counter()
        chunk_count = 0
        
        # Extract connection details
        api_base = openai_request_params.pop('api_base')
        api_key = openai_request_params.pop('api_key')
        extra_headers = openai_request_params.pop('extra_headers', {})
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            **extra_headers
        }
        
        # Send initial Claude events
        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield SSE_TEMPLATES['message_start'].format(message_id=message_id, model=self.model)
        yield SSE_TEMPLATES['ping']
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    'POST',
                    f"{api_base}/chat/completions",
                    json=openai_request_params,
                    headers=headers
                ) as response:
                    
                    if response.status_code != 200:
                        logger.error(f"OpenAI API error: {response.status_code}")
                        yield SSE_TEMPLATES['message_stop']
                        yield SSE_TEMPLATES['done']
                        return
                    
                    # Process raw SSE chunks
                    async for raw_chunk in response.aiter_text():
                        if not raw_chunk.strip():
                            continue
                            
                        chunk_count += 1
                        
                        # Parse SSE format efficiently
                        for line in raw_chunk.strip().split('\n'):
                            if line.startswith('data: '):
                                data_part = line[6:].strip()
                                
                                if data_part == '[DONE]':
                                    break
                                
                                if data_part:
                                    try:
                                        # CRITICAL: Minimal JSON parsing without object construction
                                        chunk_data = fast_json_loads(data_part)
                                        
                                        # Extract only essential data
                                        if 'choices' in chunk_data:
                                            choices = chunk_data['choices']
                                            if choices and len(choices) > 0:
                                                choice = choices[0]
                                                
                                                # Check for finish reason
                                                if choice.get('finish_reason'):
                                                    break
                                                
                                                # Extract content delta
                                                delta = choice.get('delta', {})
                                                content = delta.get('content')
                                                
                                                if content:
                                                    self.accumulated_text += content
                                                    
                                                    # Start content block if needed
                                                    if not self.content_block_started:
                                                        yield SSE_TEMPLATES['content_block_start'].format(index=self.content_block_index)
                                                        self.content_block_started = True
                                                    
                                                    # Send content delta with fast escaping
                                                    escaped_content = escape_json_string(content)
                                                    yield SSE_TEMPLATES['content_block_delta'].format(
                                                        index=self.content_block_index,
                                                        text=escaped_content
                                                    )
                                        
                                        # Extract usage data
                                        if 'usage' in chunk_data:
                                            usage = chunk_data['usage']
                                            self.input_tokens = usage.get('prompt_tokens', 0)
                                            self.output_tokens = usage.get('completion_tokens', 0)
                                    
                                    except json.JSONDecodeError:
                                        # Skip malformed JSON
                                        continue
                        
                        if data_part == '[DONE]':
                            break
        
        except Exception as e:
            logger.error(f"Raw streaming error: {e}")
        
        # Finalize stream
        if self.content_block_started:
            yield SSE_TEMPLATES['content_block_stop'].format(index=self.content_block_index)
        
        # Calculate final tokens
        final_output_tokens = self.output_tokens if self.output_tokens > 0 else len(self.accumulated_text) // 4
        
        yield SSE_TEMPLATES['message_delta'].format(
            stop_reason='end_turn',
            output_tokens=final_output_tokens
        )
        yield SSE_TEMPLATES['message_stop']
        yield SSE_TEMPLATES['done']
        
        # Performance logging
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f"Raw stream processing: {chunk_count} chunks, {total_time:.3f}s, {len(self.accumulated_text)} chars")


async def convert_openai_streaming_response_to_anthropic_raw(
    openai_request_params: Dict[str, Any],
    model: str
) -> AsyncGenerator[str, None]:
    """
    Ultra-fast streaming conversion that bypasses ALL OpenAI SDK operations.
    
    Instead of using OpenAI SDK's AsyncStream, this processes raw HTTP responses.
    Expected performance improvement: 5-10x faster than SDK approach.
    """
    
    processor = RawStreamProcessor(model)
    async for event in processor.process_raw_openai_stream(openai_request_params):
        yield event