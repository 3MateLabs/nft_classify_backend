"""
Queue service for managing model predictions
Ensures only one model prediction runs at a time
"""

import asyncio
from typing import Any, Callable, TypeVar
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ModelPredictionQueue:
    """
    A queue system that ensures model predictions run sequentially.
    This prevents multiple threads from running model inference simultaneously.
    """
    
    def __init__(self):
        self._queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._worker_task = None
        self._running = False
        
    async def start(self):
        """Start the queue worker"""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("Model prediction queue started")
            
    async def stop(self):
        """Stop the queue worker"""
        self._running = False
        if self._worker_task:
            await self._worker_task
            logger.info("Model prediction queue stopped")
            
    async def _worker(self):
        """Worker that processes queue items one at a time"""
        while self._running:
            try:
                # Wait for an item with a timeout to check _running periodically
                task_item = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                
                func = task_item['func']
                args = task_item['args']
                kwargs = task_item['kwargs']
                future = task_item['future']
                
                try:
                    # Run the function with the lock to ensure single-threaded execution
                    async with self._lock:
                        logger.info(f"Processing model prediction task: {func.__name__}")
                        if asyncio.iscoroutinefunction(func):
                            result = await func(*args, **kwargs)
                        else:
                            # Run sync function in thread pool to avoid blocking
                            result = await asyncio.get_event_loop().run_in_executor(
                                None, func, *args, **kwargs
                            )
                        future.set_result(result)
                        logger.info(f"Completed model prediction task: {func.__name__}")
                except Exception as e:
                    logger.error(f"Error in model prediction task: {e}")
                    future.set_exception(e)
                    
            except asyncio.TimeoutError:
                # Timeout is normal, just continue to check if still running
                continue
            except Exception as e:
                logger.error(f"Unexpected error in queue worker: {e}")
                
    async def add_task(self, func: Callable, *args, **kwargs) -> Any:
        """
        Add a task to the queue and wait for its completion
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function execution
        """
        future = asyncio.get_event_loop().create_future()
        
        task_item = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'future': future
        }
        
        await self._queue.put(task_item)
        logger.info(f"Added task to queue: {func.__name__}, queue size: {self._queue.qsize()}")
        
        # Wait for the task to complete
        return await future


# Global instance of the queue
model_queue = ModelPredictionQueue()


def queued_prediction(func: Callable) -> Callable:
    """
    Decorator to queue model predictions
    
    Usage:
        @queued_prediction
        def generate_embedding(image):
            # Model prediction code
            pass
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await model_queue.add_task(func, *args, **kwargs)
    
    return wrapper