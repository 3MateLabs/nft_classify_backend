# Model Prediction Queue and Cache Implementation

## Overview

This implementation ensures that model predictions run one at a time (single-threaded) while allowing image downloads to happen asynchronously. It also includes an LRU cache for the latest 1000 URLs to avoid redundant processing. This prevents memory issues and ensures consistent performance when multiple requests are made concurrently.

## Architecture

### Components

1. **Queue Service** (`api/services/queue_service.py`)
   - Implements a thread-safe queue using `asyncio.Queue`
   - Ensures only one model prediction runs at a time
   - Handles both sync and async functions

2. **Model Service Updates** (`api/services/model_service.py`)
   - `generate_embedding()` is now decorated with `@queued_prediction`
   - Actual model inference happens in `_generate_embedding_sync()`
   - Queue automatically manages execution order

3. **Async Image Download** (`api/services/image_service.py`)
   - Added `download_image_async()` using `aiohttp`
   - Added `process_image_from_url_async()` for async image processing
   - Downloads happen concurrently while model predictions are queued

4. **Handler Updates** (`api/handlers/embedding_handlers.py`)
   - `url_to_embedding()` now uses async image download
   - Model predictions are awaited (handled by queue)
   - Cache check happens before processing

5. **Cache Service** (`api/services/cache_service.py`)
   - LRU (Least Recently Used) cache for 1000 URLs
   - Thread-safe implementation using `asyncio.Lock`
   - Tracks cache hits, misses, and evictions
   - SHA256 hash-based key generation

## How It Works

### Queue System
1. **Multiple requests arrive** at `/embed_from_url` endpoint
2. **Cache is checked first** - if URL exists in cache, return immediately
3. **Images download concurrently** using `aiohttp` (for cache misses)
4. **Model predictions are queued** and execute one at a time
5. **Results are cached** before returning to client

### Cache System
1. **Cache lookup** using SHA256 hash of URL
2. **Cache hit** returns result immediately (sub-millisecond)
3. **Cache miss** proceeds with normal processing
4. **LRU eviction** when cache exceeds 1000 entries
5. **Thread-safe** operations using asyncio locks

## Testing

### Queue Testing
```bash
# Test concurrent requests and queue behavior
python test_queue.py
```

### Cache Testing
```bash
# Test cache functionality
python test_cache.py
```

The tests will verify:
- Model predictions run sequentially (queue)
- Cache hits are significantly faster than misses
- LRU eviction works correctly
- Statistics are tracked accurately

## Benefits

1. **Memory Safety**: Only one model runs at a time, preventing OOM errors
2. **Better Performance**: Image downloads happen concurrently
3. **Cache Efficiency**: Repeated requests return instantly from cache
4. **Scalability**: Queue can handle many pending requests
5. **Transparency**: Detailed logging shows queue and cache status
6. **Cost Savings**: Reduced model inference for duplicate requests

## Configuration

The queue starts automatically when the FastAPI app starts and stops gracefully on shutdown. No additional configuration is needed.

## API Endpoints

### Cache Management
- `GET /cache/stats` - Get cache statistics (hits, misses, size, hit rate)
- `POST /cache/clear` - Clear all cached entries

## Monitoring

Check the logs for:
- Queue size when tasks are added
- When model predictions start/complete
- Cache hits/misses with URL and key info
- Any errors during processing

Example cache stats response:
```json
{
  "size": 42,
  "max_size": 1000,
  "hits": 123,
  "misses": 45,
  "evictions": 0,
  "hit_rate": "73.21%",
  "total_requests": 168,
  "oldest_entry_age_seconds": 3600.5,
  "newest_entry_age_seconds": 2.1
}
```