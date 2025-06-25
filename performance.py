"""
Performance monitoring and timing utilities for the Claude proxy server.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variables for decorators
F = TypeVar('F', bound=Callable[..., Any])
AF = TypeVar('AF', bound=Callable[..., Awaitable[Any]])

class PerformanceMetrics:
    """Thread-safe performance metrics collector"""
    
    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
    
    async def record_timing(self, operation: str, duration: float) -> None:
        """Record a timing metric for an operation"""
        async with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = []
            self._metrics[operation].append(duration)
            
            # Keep only last 1000 measurements to prevent memory bloat
            if len(self._metrics[operation]) > 1000:
                self._metrics[operation] = self._metrics[operation][-1000:]
    
    async def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for an operation"""
        async with self._lock:
            if operation not in self._metrics or not self._metrics[operation]:
                return None
            
            timings = self._metrics[operation]
            return {
                'count': len(timings),
                'avg': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings),
                'recent_avg': sum(timings[-10:]) / min(len(timings), 10),  # Last 10 calls
            }
    
    async def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations"""
        async with self._lock:
            stats = {}
            for operation in self._metrics:
                if self._metrics[operation]:
                    timings = self._metrics[operation]
                    stats[operation] = {
                        'count': len(timings),
                        'avg': sum(timings) / len(timings),
                        'min': min(timings),
                        'max': max(timings),
                        'recent_avg': sum(timings[-10:]) / min(len(timings), 10),
                    }
            return stats

# Global metrics instance
perf_metrics = PerformanceMetrics()

def time_sync_function(operation_name: Optional[str] = None) -> Callable[[F], F]:
    """Decorator to time synchronous functions"""
    def decorator(func: F) -> F:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                # Record timing asynchronously (fire and forget)
                asyncio.create_task(perf_metrics.record_timing(name, duration))
                
                # Log slow operations
                if duration > 0.1:  # Log operations taking more than 100ms
                    logger.warning(f"SLOW: {name} took {duration:.3f}s")
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"PERF: {name} took {duration:.3f}s")
        
        return wrapper
    return decorator

def time_async_function(operation_name: Optional[str] = None) -> Callable[[AF], AF]:
    """Decorator to time async functions"""
    def decorator(func: AF) -> AF:
        name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time
                await perf_metrics.record_timing(name, duration)
                
                # Log slow operations
                if duration > 0.1:  # Log operations taking more than 100ms
                    logger.warning(f"SLOW: {name} took {duration:.3f}s")
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"PERF: {name} took {duration:.3f}s")
        
        return wrapper
    return decorator

class StreamingTimer:
    """Context manager for timing streaming operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.chunk_count = 0
        self.total_bytes = 0
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            await perf_metrics.record_timing(self.operation_name, duration)
            
            # Calculate streaming stats
            if self.chunk_count > 0:
                chunks_per_sec = self.chunk_count / duration
                bytes_per_sec = self.total_bytes / duration
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"STREAM: {self.operation_name} - {duration:.3f}s, "
                        f"{self.chunk_count} chunks ({chunks_per_sec:.1f}/s), "
                        f"{self.total_bytes} bytes ({bytes_per_sec:.1f}/s)"
                    )
    
    def record_chunk(self, byte_size: int = 0):
        """Record a chunk being processed"""
        self.chunk_count += 1
        self.total_bytes += byte_size

# Token rate tracking
class TokenRateTracker:
    """Track token processing rates"""
    
    def __init__(self):
        self._measurements: List[tuple[float, int]] = []  # (timestamp, tokens)
        self._lock = asyncio.Lock()
    
    async def record_tokens(self, token_count: int) -> None:
        """Record tokens processed"""
        async with self._lock:
            now = time.perf_counter()
            self._measurements.append((now, token_count))
            
            # Keep only last 60 seconds of data
            cutoff = now - 60.0
            self._measurements = [(t, c) for t, c in self._measurements if t > cutoff]
    
    async def get_rate(self) -> Optional[float]:
        """Get tokens per second over the last 60 seconds"""
        async with self._lock:
            if len(self._measurements) < 2:
                return None
            
            now = time.perf_counter()
            cutoff = now - 60.0
            
            recent = [(t, c) for t, c in self._measurements if t > cutoff]
            if len(recent) < 2:
                return None
            
            total_tokens = sum(c for _, c in recent)
            time_span = recent[-1][0] - recent[0][0]
            
            if time_span > 0:
                return total_tokens / time_span
            return None

# Global token rate tracker
token_rate_tracker = TokenRateTracker()