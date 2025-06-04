"""
Cache service for storing embedding results
Uses LRU (Least Recently Used) cache with a maximum of 1000 entries
"""

import asyncio
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
import hashlib
import time
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Thread-safe LRU cache for embedding results
    """
    
    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def _generate_key(self, url: str) -> str:
        """Generate a cache key from URL"""
        return hashlib.sha256(url.encode()).hexdigest()
        
    async def get(self, url: str) -> Optional[Any]:
        """
        Get cached result for a URL
        
        Args:
            url: The image URL
            
        Returns:
            Cached embedding result or None if not found
        """
        key = self._generate_key(url)
        
        async with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                result, timestamp = self._cache.pop(key)
                self._cache[key] = (result, timestamp)
                self._hits += 1
                logger.info(f"Cache hit for URL: {url[:100]}... (key: {key[:8]}...)")
                return result
            else:
                self._misses += 1
                logger.info(f"Cache miss for URL: {url[:100]}... (key: {key[:8]}...)")
                return None
                
    async def put(self, url: str, result: Any) -> None:
        """
        Store result in cache
        
        Args:
            url: The image URL
            result: The embedding result to cache
        """
        key = self._generate_key(url)
        
        async with self._lock:
            # Remove key if it already exists (to update position)
            if key in self._cache:
                self._cache.pop(key)
            
            # Add to end (most recently used)
            self._cache[key] = (result, time.time())
            
            # Evict oldest if over capacity
            if len(self._cache) > self._max_size:
                evicted_key = next(iter(self._cache))
                self._cache.pop(evicted_key)
                self._evictions += 1
                logger.info(f"Evicted oldest cache entry (key: {evicted_key[:8]}...)")
                
            logger.info(f"Cached result for URL: {url[:100]}... (key: {key[:8]}...) - Cache size: {len(self._cache)}")
            
    async def clear(self) -> None:
        """Clear all cached entries"""
        async with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            # Get age of oldest and newest entries
            oldest_age = None
            newest_age = None
            
            if self._cache:
                current_time = time.time()
                timestamps = [timestamp for _, timestamp in self._cache.values()]
                oldest_age = current_time - min(timestamps)
                newest_age = current_time - max(timestamps)
            
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": f"{hit_rate:.2f}%",
                "total_requests": total_requests,
                "oldest_entry_age_seconds": oldest_age,
                "newest_entry_age_seconds": newest_age
            }
            
    async def get_size(self) -> int:
        """Get current cache size"""
        async with self._lock:
            return len(self._cache)


# Global cache instance
embedding_cache = EmbeddingCache(max_size=1000)