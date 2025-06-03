"""
Test script to verify the single-threaded model execution queue
"""

import asyncio
import aiohttp
import time
import os
from typing import List
import dotenv

# Load environment variables
dotenv.load_dotenv()


async def test_concurrent_requests(base_url: str, api_key: str, test_urls: List[str]):
    """
    Test concurrent requests to verify model predictions run one at a time
    """
    headers = {"X-API-Key": api_key}

    async def make_request(session: aiohttp.ClientSession, url: str, index: int):
        """Make a single request and track timing"""
        payload = {"img_url": url}
        start_time = time.time()

        print(f"[Request {index}] Starting request at {start_time:.2f}")

        try:
            async with session.post(
                f"{base_url}/embed_from_url", json=payload, headers=headers
            ) as response:
                result = await response.json()
                end_time = time.time()
                duration = end_time - start_time

                if response.status == 200:
                    print(f"[Request {index}] ✓ Completed in {duration:.2f}s")
                    return {"index": index, "duration": duration, "success": True}
                else:
                    print(
                        f"[Request {index}] ✗ Failed with status {response.status}: {result}"
                    )
                    return {
                        "index": index,
                        "duration": duration,
                        "success": False,
                        "error": result,
                    }

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"[Request {index}] ✗ Exception after {duration:.2f}s: {str(e)}")
            return {
                "index": index,
                "duration": duration,
                "success": False,
                "error": str(e),
            }

    # Create session and make concurrent requests
    async with aiohttp.ClientSession() as session:
        print(f"\nStarting {len(test_urls)} concurrent requests...")
        start_time = time.time()

        # Launch all requests concurrently
        tasks = [make_request(session, url, i) for i, url in enumerate(test_urls)]

        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Analyze results
        print(f"\n{'='*50}")
        print(f"Total time for all requests: {total_time:.2f}s")
        print(f"Average time per request: {total_time/len(test_urls):.2f}s")

        successful = [r for r in results if r["success"]]
        print(f"Successful requests: {len(successful)}/{len(results)}")

        if successful:
            avg_duration = sum(r["duration"] for r in successful) / len(successful)
            print(f"Average duration for successful requests: {avg_duration:.2f}s")

        print(f"\nNote: If the queue is working correctly, requests should be")
        print(f"processed sequentially for model inference, while downloads")
        print(f"happen concurrently. The total time should be close to the")
        print(f"sum of individual model inference times.")


async def main():
    """Main test function"""
    # Configuration
    BASE_URL = "http://localhost:3001"  # Adjust if needed
    API_KEY = os.getenv("API_KEY", "test_api_key")  # Load from .env file

    # Test URLs - using small images for faster testing
    test_urls = [
        "https://picsum.photos/200/200?random=1",
        "https://picsum.photos/200/200?random=2",
        "https://picsum.photos/200/200?random=3",
        "https://picsum.photos/200/200?random=4",
        "https://picsum.photos/200/200?random=5",
    ]

    print("Testing Model Prediction Queue")
    print("==============================")
    print(f"Base URL: {BASE_URL}")
    print(f"Number of test requests: {len(test_urls)}")
    
    # Check if API key is loaded
    if API_KEY == "test_api_key":
        print("⚠️  Warning: Using default API key. Make sure .env file contains API_KEY")
    else:
        print(f"✓ API key loaded from .env")

    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/ping") as response:
                if response.status != 200:
                    print("✗ Server is not responding. Please start the server first.")
                    return
                print("✓ Server is running")
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        print("Please start the server with: uvicorn api.main:app --reload")
        return

    # Run the test
    await test_concurrent_requests(BASE_URL, API_KEY, test_urls)


if __name__ == "__main__":
    asyncio.run(main())
