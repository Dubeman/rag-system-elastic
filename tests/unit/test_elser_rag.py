"""
Unit tests for ELSER functionality via RAG system API.
Tests ELSER model deployment, inference, and integration with the RAG system.
"""

import unittest
import requests
import json
import time
from typing import Dict, Any


class TestELSERRAG(unittest.TestCase):
    """Test ELSER functionality through RAG system API."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_url = "http://localhost:8000"
        self.elasticsearch_url = "http://localhost:9200"
        self.test_queries = [
            "What is discussed about docker?",
            "Explain machine learning concepts",
            "Tell me about artificial intelligence",
            "How does containerization work?",
            "What are the benefits of microservices?"
        ]
        
        # Wait for services to be ready
        self._wait_for_services()
    
    def _wait_for_services(self):
        """Wait for API and Elasticsearch services to be ready."""
        max_retries = 30
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Check API health
                api_response = requests.get(f"{self.base_url}/healthz", timeout=5)
                if api_response.status_code == 200:
                    print("✅ API service is ready")
                    break
            except requests.exceptions.RequestException:
                pass
            
            try:
                # Check Elasticsearch health
                es_response = requests.get(f"{self.elasticsearch_url}/_cluster/health", timeout=5)
                if es_response.status_code == 200:
                    print("✅ Elasticsearch service is ready")
                    break
            except requests.exceptions.RequestException:
                pass
            
            retry_count += 1
            time.sleep(2)
            print(f"⏳ Waiting for services... ({retry_count}/{max_retries})")
        
        if retry_count >= max_retries:
            self.fail("Services did not become ready within expected time")
    
    def test_01_elser_model_status(self):
        """Test if ELSER model is properly deployed."""
        print("\n🔍 Testing ELSER model deployment status...")
        
        # Check if ELSER model exists
        response = requests.get(f"{self.elasticsearch_url}/_ml/trained_models/.elser_model_2")
        self.assertEqual(response.status_code, 200, "ELSER model should exist")
        
        model_info = response.json()
        print(f"✅ ELSER model exists: {model_info.get('model_id', 'Unknown')}")
        
        # Check deployment status
        response = requests.get(f"{self.elasticsearch_url}/_ml/trained_models/.elser_model_2/_stats")
        self.assertEqual(response.status_code, 200, "Should be able to get deployment stats")
        
        deployment_info = response.json()
        print(f"✅ Deployment stats retrieved: {json.dumps(deployment_info, indent=2)}")
        
        # Verify deployment is running
        if "deployment_stats" in deployment_info:
            for node_stats in deployment_info["deployment_stats"].values():
                if "state" in node_stats:
                    state = node_stats["state"]
                    print(f"🎯 Deployment state: {state}")
                    self.assertIn(state, ["started", "starting"], "ELSER should be started or starting")
    
    def test_02_elser_inference_direct(self):
        """Test ELSER inference directly via Elasticsearch API."""
        print("\n🧠 Testing ELSER inference directly...")
        
        test_text = "This is a test document about machine learning and artificial intelligence."
        
        payload = {
            "docs": [{"text_field": test_text}]
        }
        
        response = requests.post(
            f"{self.elasticsearch_url}/_ml/trained_models/.elser_model_2/_infer",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200, "ELSER inference should succeed")
        
        result = response.json()
        print(f"✅ ELSER inference successful!")
        
        # Verify response structure
        self.assertIn("inference_results", result, "Response should contain inference_results")
        self.assertGreater(len(result["inference_results"]), 0, "Should have at least one result")
        
        # Extract and verify text_expansion
        first_result = result["inference_results"][0]
        self.assertIn("predicted_value", first_result, "Result should contain predicted_value")
        
        predicted_value = first_result["predicted_value"]
        self.assertIn("text_expansion", predicted_value, "Should contain text_expansion")
        
        text_expansion = predicted_value["text_expansion"]
        self.assertIn("tokens", text_expansion, "text_expansion should contain tokens")
        
        tokens = text_expansion["tokens"]
        self.assertIsInstance(tokens, dict, "Tokens should be a dictionary")
        self.assertGreater(len(tokens), 0, "Should generate some tokens")
        
        print(f"🎯 Generated {len(tokens)} ELSER tokens")
        print(f"Sample tokens: {dict(list(tokens.items())[:5])}")
    
    def test_03_elser_rag_query(self):
        """Test ELSER through RAG system query endpoint."""
        print("\n🔍 Testing ELSER via RAG API...")
        
        for query in self.test_queries:
            print(f"\n📝 Testing query: '{query}'")
            
            payload = {
                "question": query,
                "search_mode": "elser_only",  # Force ELSER mode
                "top_k": 5
            }
            
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            self.assertEqual(response.status_code, 200, f"Query should succeed: {query}")
            
            result = response.json()
            print(f"✅ Query successful for: {query}")
            
            # Verify response structure
            self.assertIn("results", result, "Response should contain results")
            self.assertIn("llm_response", result, "Response should contain llm_response")
            
            print(f"📖 Results: {len(result.get('results', []))}")
            print(f"🔗 LLM Response: {result.get('llm_response', {}).get('answer', 'No answer')[:100]}...")
            
            # If results exist, verify they have expected structure
            results = result.get("results", [])
            if results:
                for result_item in results:
                    self.assertIn("content", result_item, "Result should contain content")
                    self.assertIn("filename", result_item, "Result should contain filename")
    
    def test_04_elser_hybrid_search(self):
        """Test ELSER in hybrid search mode."""
        print("\n🔄 Testing ELSER in hybrid search mode...")
        
        query = "What are the main concepts discussed in the documents?"
        
        payload = {
            "question": query,
            "search_mode": "full_hybrid",  # Use full hybrid search (ELSER + Dense + BM25)
            "top_k": 5
        }
        
        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200, "Hybrid search should succeed")
        
        result = response.json()
        print(f"✅ Hybrid search successful!")
        print(f"📖 Results: {len(result.get('results', []))}")
        print(f"🔗 LLM Response: {result.get('llm_response', {}).get('answer', 'No answer')[:100]}...")
    
    def test_05_elser_error_handling(self):
        """Test ELSER error handling with invalid queries."""
        print("\n⚠️ Testing ELSER error handling...")
        
        # Test with empty query
        payload = {
            "question": "",
            "search_mode": "elser_only",
            "top_k": 5
        }
        
        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Should handle empty query gracefully
        self.assertIn(response.status_code, [200, 400, 422], "Should handle empty query")
        
        # Test with very long query
        long_query = "This is a very long query " * 100  # Create a very long query
        
        payload = {
            "question": long_query,
            "search_mode": "elser_only",
            "top_k": 5
        }
        
        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Should handle long query gracefully
        self.assertIn(response.status_code, [200, 400, 413], "Should handle long query")
    
    def test_06_elser_performance(self):
        """Test ELSER response time and performance."""
        print("\n⏱️ Testing ELSER performance...")
        
        query = "What is the main topic discussed?"
        
        payload = {
            "question": query,
            "search_mode": "elser_only",
            "top_k": 5
        }
        
        start_time = time.time()
        
        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60  # Longer timeout for performance test
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200, "Performance test query should succeed")
        
        print(f"✅ Query completed in {response_time:.2f} seconds")
        
        # Performance assertion (adjust threshold as needed)
        self.assertLess(response_time, 30.0, f"Query should complete within 30 seconds, took {response_time:.2f}s")
        
        result = response.json()
        print(f"📖 Results count: {len(result.get('results', []))}")
        print(f"🔗 LLM Response length: {len(result.get('llm_response', {}).get('answer', ''))}")


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestELSERRAG)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n💥 Some tests failed!")
