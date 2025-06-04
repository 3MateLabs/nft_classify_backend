"""
Test script to verify the cache functionality
"""

import asyncio
import aiohttp
import time
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()


async def test_cache_functionality(base_url: str, api_key: str):
    """Test cache functionality with repeated requests"""
    headers = {"X-API-Key": api_key}
    test_url = "https://picsum.photos/200/200?random=cache_test"
    
    async with aiohttp.ClientSession() as session:
        print("\n=== Testing Cache Functionality ===\n")
        
        # 1. Clear cache first
        print("1. Clearing cache...")
        async with session.post(f"{base_url}/cache/clear", headers=headers) as resp:
            result = await resp.json()
            print(f"   Response: {result}")
        
        # 2. Check initial cache stats
        print("\n2. Initial cache stats:")
        async with session.get(f"{base_url}/cache/stats", headers=headers) as resp:
            stats = await resp.json()
            print(f"   {stats}")
        
        # 3. First request (should be cache miss)
        print(f"\n3. First request to {test_url}")
        start_time = time.time()
        async with session.post(
            f"{base_url}/embed_from_url",
            json={"img_url": test_url},
            headers=headers
        ) as resp:
            result = await resp.json()
            first_time = time.time() - start_time
            print(f"   Time: {first_time:.2f}s")
            print(f"   Cached: {result.get('cached', 'N/A')}")
            print(f"   Embedding preview: {result['embedding'][0][:5]}...")
        
        # 4. Second request (should be cache hit)
        print(f"\n4. Second request to same URL")
        start_time = time.time()
        async with session.post(
            f"{base_url}/embed_from_url",
            json={"img_url": test_url},
            headers=headers
        ) as resp:
            result = await resp.json()
            second_time = time.time() - start_time
            print(f"   Time: {second_time:.2f}s")
            print(f"   Cached: {result.get('cached', 'N/A')}")
            print(f"   Embedding preview: {result['embedding'][0][:5]}...")
        
        # 5. Check cache stats after requests
        print("\n5. Cache stats after requests:")
        async with session.get(f"{base_url}/cache/stats", headers=headers) as resp:
            stats = await resp.json()
            print(f"   {stats}")
        
        # Analysis
        print("\n=== Analysis ===")
        print(f"First request time: {first_time:.2f}s")
        print(f"Second request time: {second_time:.2f}s")
        print(f"Speed improvement: {first_time/second_time:.1f}x faster")
        
        if second_time < first_time * 0.1:  # Should be at least 10x faster
            print("✓ Cache is working correctly!")
        else:
            print("⚠️  Cache might not be working as expected")


async def test_cache_capacity(base_url: str, api_key: str, num_urls: int = 10):
    """Test cache with multiple URLs"""
    headers = {"X-API-Key": api_key}
    
    async with aiohttp.ClientSession() as session:
        print(f"\n=== Testing Cache with {num_urls} Different URLs ===\n")
        
        # Clear cache first
        await session.post(f"{base_url}/cache/clear", headers=headers)
        
        # Make requests to different URLs
        for i in range(num_urls):
            url = f"https://picsum.photos/200/200?random={i}"
            async with session.post(
                f"{base_url}/embed_from_url",
                json={"img_url": url},
                headers=headers
            ) as resp:
                result = await resp.json()
                print(f"Request {i+1}: cached={result.get('cached', False)}")
        
        # Check final cache stats
        async with session.get(f"{base_url}/cache/stats", headers=headers) as resp:
            stats = await resp.json()
            print(f"\nFinal cache stats: {stats}")


async def main():
    """Main test function"""
    BASE_URL = "http://localhost:3001"
    API_KEY = os.getenv("API_KEY", "test_api_key")
    
    print("Testing Embedding Cache System")
    print("==============================")
    
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
    
    # Run tests
    await test_cache_functionality(BASE_URL, API_KEY)
    await test_cache_capacity(BASE_URL, API_KEY, num_urls=10)


if __name__ == "__main__":
    asyncio.run(main())