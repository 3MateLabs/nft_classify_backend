# Model Prediction Queue Implementation

## Overview

This implementation ensures that model predictions run one at a time (single-threaded) while allowing image downloads to happen asynchronously. This prevents memory issues and ensures consistent performance when multiple requests are made concurrently.

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

## How It Works

1. **Multiple requests arrive** at `/embed_from_url` endpoint
2. **Images download concurrently** using `aiohttp`
3. **Model predictions are queued** and execute one at a time
4. **Results return** as each prediction completes

## Testing

Run the test script to verify the implementation:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn api.main:app --reload

# In another terminal, run the test
python test_queue.py
```

The test will:
- Send 5 concurrent requests
- Show timing for each request
- Verify that model predictions run sequentially

## Benefits

1. **Memory Safety**: Only one model runs at a time, preventing OOM errors
2. **Better Performance**: Image downloads happen concurrently
3. **Scalability**: Queue can handle many pending requests
4. **Transparency**: Detailed logging shows queue status

## Configuration

The queue starts automatically when the FastAPI app starts and stops gracefully on shutdown. No additional configuration is needed.

## Monitoring

Check the logs for queue information:
- Queue size when tasks are added
- When model predictions start/complete
- Any errors during processing