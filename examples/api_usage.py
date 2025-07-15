#!/usr/bin/env python3
"""
API Usage Example for BERT Sentiment Analysis

This script demonstrates how to use the FastAPI application for sentiment analysis.
"""

import requests
import json
import time
import asyncio
import httpx

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_api_sync():
    """Test the API using synchronous requests."""
    print("🌐 Testing BERT Sentiment Analysis API (Synchronous)")
    print("=" * 60)
    
    # Test texts
    test_texts = [
        "Eu adorei este produto! É excelente e recomendo para todos.",
        "Este produto é horrível, não funciona e é muito caro.",
        "O produto é ok, nada demais mas cumpre o que promete.",
        "Fantástico! Melhor compra que já fiz na vida.",
        "Terrível, não comprem este produto de jeito nenhum.",
    ]
    
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Health Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    print("\n2. Testing Sentiment Predictions...")
    for i, text in enumerate(test_texts, 1):
        try:
            # Make prediction request
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"\\n📝 Test {i}:")
                print(f"   Text: {text[:50]}...")
                print(f"   Sentiment: {result['sentiment']}")
                print(f"   Confidence: {result['confidence']:.4f}")
                print(f"   Response time: {end_time - start_time:.3f}s")
            else:
                print(f"❌ Request {i} failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error in request {i}: {e}")
    
    print("\n3. Testing Error Handling...")
    
    # Test empty text
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": ""},
            headers={"Content-Type": "application/json"}
        )
        print(f"Empty text response: {response.status_code}")
        if response.status_code != 200:
            print(f"Expected error: {response.json()}")
    except Exception as e:
        print(f"Empty text test error: {e}")
    
    # Test invalid JSON
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"invalid_field": "test"},
            headers={"Content-Type": "application/json"}
        )
        print(f"Invalid JSON response: {response.status_code}")
        if response.status_code != 200:
            print(f"Expected error: {response.json()}")
    except Exception as e:
        print(f"Invalid JSON test error: {e}")

async def test_api_async():
    """Test the API using asynchronous requests."""
    print("\\n🚀 Testing BERT Sentiment Analysis API (Asynchronous)")
    print("=" * 60)
    
    async with httpx.AsyncClient() as client:
        # Test multiple requests concurrently
        test_texts = [
            "Produto excelente, super recomendo!",
            "Não gostei, muito ruim mesmo.",
            "Mediano, nem bom nem ruim.",
            "Adorei a qualidade, vale muito a pena!",
            "Péssimo produto, não comprem.",
        ]
        
        print("\\nTesting concurrent requests...")
        
        # Create tasks for concurrent requests
        tasks = []
        for i, text in enumerate(test_texts):
            task = client.post(
                f"{API_BASE_URL}/predict",
                json={"text": text}
            )
            tasks.append(task)
        
        # Execute all requests concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"\\n📊 Concurrent Processing Results:")
        print(f"Total requests: {len(test_texts)}")
        print(f"Total time: {end_time - start_time:.3f}s")
        print(f"Average time per request: {(end_time - start_time) / len(test_texts):.3f}s")
        
        for i, response in enumerate(responses):
            if response.status_code == 200:
                result = response.json()
                print(f"\\n📝 Request {i+1}:")
                print(f"   Text: {test_texts[i][:40]}...")
                print(f"   Sentiment: {result['sentiment']}")
                print(f"   Confidence: {result['confidence']:.4f}")
            else:
                print(f"❌ Request {i+1} failed: {response.status_code}")

def benchmark_api():
    """Benchmark the API performance."""
    print("\\n📈 Benchmarking API Performance")
    print("=" * 40)
    
    # Test text
    text = "Este produto é muito bom e eu recomendo para todos!"
    
    # Number of requests for benchmark
    num_requests = 50
    
    print(f"Running {num_requests} requests...")
    
    times = []
    successes = 0
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json={"text": text}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                successes += 1
                times.append(end_time - start_time)
                
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"\\n📊 Benchmark Results:")
        print(f"Total requests: {num_requests}")
        print(f"Successful requests: {successes}")
        print(f"Success rate: {successes/num_requests*100:.1f}%")
        print(f"Average response time: {avg_time:.3f}s")
        print(f"Min response time: {min_time:.3f}s")
        print(f"Max response time: {max_time:.3f}s")
        print(f"Requests per second: {1/avg_time:.2f}")

def main():
    print("Starting API usage examples...")
    print("Make sure the API is running on http://localhost:8000")
    print("Run: python api.py")
    print()
    
    # Test synchronous requests
    test_api_sync()
    
    # Test asynchronous requests
    asyncio.run(test_api_async())
    
    # Benchmark performance
    benchmark_api()
    
    print("\\n✅ All API tests completed!")

if __name__ == "__main__":
    main()