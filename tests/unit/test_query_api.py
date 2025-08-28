"""Unit tests for the Query API endpoints."""

import unittest
import requests
import time
import json
from typing import Dict, Any


class TestQueryAPI(unittest.TestCase):
    """Test cases for the Query API functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_url = "http://localhost:8000"
        self.query_endpoint = f"{self.base_url}/query"
        self.health_endpoint = f"{self.base_url}/healthz"
        
        # Test questions covering different scenarios - practical and technical
        self.test_questions = [
            # Practical/Technical Questions
            "What is a Dockerfile and how does it work?",
            "What are CSRF tokens in the FortiOS REST API reference?",
            "How do you create a Docker container from an image?",
            "What is the difference between Docker and Docker Compose?",
            "How do you implement authentication in a REST API?",
            "What are the best practices for container security?",
            "How does load balancing work in microservices?",
            "What is the purpose of health checks in Docker?",
            "How do you manage environment variables in Docker?",
            "What are the benefits of using container orchestration?",
            
            # Context Engineering Questions (for comparison)
            "What is context engineering and how does it work?",
            "Explain the three pillars of context engineering",
            "What are the benefits of using ELSER embeddings?",
            "How does BM25 search differ from semantic search?",
            "What is the difference between prompt engineering and context engineering?"
        ]
        
        # Search modes to test - reordered to show BM25 first, then hybrid
        self.search_modes = [
            "bm25_only",      # Start with keyword search
            "dense_only",     # Then dense embeddings
            "elser_only",     # Then sparse embeddings
            "dense_bm25",     # Then hybrid approaches
            "full_hybrid"     # Finally full hybrid
        ]
        
        # Results counts to test
        self.result_counts = [3, 5, 10]
        
        # Wait for API to be ready
        self._wait_for_api()
    
    def _wait_for_api(self, max_attempts: int = 30, delay: float = 2.0):
        """Wait for the API to be ready."""
        print("Waiting for API to be ready...")
        for attempt in range(max_attempts):
            try:
                response = requests.get(self.health_endpoint, timeout=5)
                if response.status_code == 200:
                    print(f"API ready after {attempt + 1} attempts")
                    return
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_attempts - 1:
                time.sleep(delay)
        
        raise Exception("API failed to become ready within expected time")
    
    def test_01_api_health(self):
        """Test API health endpoint."""
        print("\n🔍 Testing API Health...")
        
        response = requests.get(self.health_endpoint, timeout=10)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        
        print("✅ API health check passed")
    
    def test_02_practical_questions_bm25_first(self):
        """Test practical questions with BM25 first, then hybrid for comparison."""
        print("\n🔍 Testing Practical Questions - BM25 First, Then Hybrid...")
        
        practical_questions = [
            "What is a Dockerfile and how does it work?",
            "What are CSRF tokens in the FortiOS REST API reference?",
            "How do you create a Docker container from an image?",
            "What is the difference between Docker and Docker Compose?"
        ]
        
        for question in practical_questions:
            print(f"\n📝 Question: {question}")
            print("-" * 80)
            
            # First test with BM25 (keyword search)
            print("🔍 Testing with BM25 (Keyword Search)...")
            bm25_payload = {
                "question": question,
                "search_mode": "bm25_only",
                "top_k": 5,
                "generate_answer": True
            }
            
            bm25_response = requests.post(self.query_endpoint, json=bm25_payload, timeout=60)
            if bm25_response.status_code == 200:
                bm25_data = bm25_response.json()
                print(f"✅ BM25 Results: {len(bm25_data.get('results', []))} documents found")
                if 'llm_response' in bm25_data and 'answer' in bm25_data['llm_response']:
                    bm25_answer = bm25_data['llm_response']['answer']
                    print(f"📖 BM25 Answer: {bm25_answer[:200]}...")
            else:
                print(f"❌ BM25 failed: {bm25_response.status_code}")
            
            # Then test with full hybrid for comparison
            print("\n🔍 Testing with Full Hybrid Search...")
            hybrid_payload = {
                "question": question,
                "search_mode": "full_hybrid",
                "top_k": 5,
                "generate_answer": True
            }
            
            hybrid_response = requests.post(self.query_endpoint, json=hybrid_payload, timeout=60)
            if hybrid_response.status_code == 200:
                hybrid_data = hybrid_response.json()
                print(f"✅ Hybrid Results: {len(hybrid_data.get('results', []))} documents found")
                if 'llm_response' in hybrid_data and 'answer' in hybrid_data['llm_response']:
                    hybrid_answer = hybrid_data['llm_response']['answer']
                    print(f"📖 Hybrid Answer: {hybrid_answer[:200]}...")
            else:
                print(f"❌ Hybrid failed: {hybrid_response.status_code}")
            
            print("=" * 80)
    
    def test_03_search_mode_comparison_practical(self):
        """Test different search modes with practical questions."""
        print("\n🔍 Testing Search Mode Comparison with Practical Questions...")
        
        test_question = "What is a Dockerfile and how does it work?"
        results_by_mode = {}
        
        for mode in self.search_modes:
            print(f"\n🔍 Testing search mode: {mode}")
            
            payload = {
                "question": test_question,
                "search_mode": mode,
                "top_k": 5,
                "generate_answer": True
            }
            
            response = requests.post(self.query_endpoint, json=payload, timeout=60)
            self.assertEqual(response.status_code, 200, f"Failed for mode: {mode}")
            
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["search_mode"], mode)
            
            results = data.get("results", [])
            answer = data.get("llm_response", {}).get("answer", "")
            
            results_by_mode[mode] = {
                "result_count": len(results),
                "answer_length": len(answer),
                "has_results": len(results) > 0,
                "answer_preview": answer[:150] + "..." if len(answer) > 150 else answer
            }
            
            print(f"✅ {mode}: {len(results)} results")
            print(f"📖 Answer preview: {results_by_mode[mode]['answer_preview']}")
        
        # Verify all modes returned results
        for mode, stats in results_by_mode.items():
            self.assertTrue(stats["has_results"], f"Mode {mode} returned no results")
            self.assertGreater(stats["answer_length"], 10, f"Mode {mode} returned very short answer")
        
        print("\n📊 Search Mode Comparison Summary:")
        for mode, stats in results_by_mode.items():
            print(f"  {mode:15}: {stats['result_count']:2} results, {stats['answer_length']:3} chars")
    
    def test_04_docker_specific_questions(self):
        """Test Docker-specific questions to see how well the system handles technical content."""
        print("\n🔍 Testing Docker-Specific Questions...")
        
        docker_questions = [
            "What is a Dockerfile and how does it work?",
            "How do you create a Docker container from an image?",
            "What is the difference between Docker and Docker Compose?",
            "What are the best practices for container security?",
            "How do you manage environment variables in Docker?"
        ]
        
        for question in docker_questions:
            print(f"\n🐳 Docker Question: {question}")
            print("-" * 60)
            
            # Test with BM25 first
            print("🔍 BM25 Search Results:")
            bm25_payload = {
                "question": question,
                "search_mode": "bm25_only",
                "top_k": 3,
                "generate_answer": True
            }
            
            response = requests.post(self.query_endpoint, json=bm25_payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                answer = data.get("llm_response", {}).get("answer", "")
                
                print(f"✅ Found {len(results)} documents")
                print(f"📖 Answer: {answer[:300]}...")
                
                # Show top result details
                if results:
                    top_result = results[0]
                    print(f"📄 Top Result: {top_result.get('filename', 'Unknown')}")
                    print(f"   Content: {top_result.get('content', '')[:100]}...")
            else:
                print(f"❌ BM25 search failed: {response.status_code}")
            
            print("=" * 60)
    
    def test_05_fortios_and_api_questions(self):
        """Test FortiOS and API-related questions."""
        print("\n🔍 Testing FortiOS and API Questions...")
        
        api_questions = [
            "What are CSRF tokens in the FortiOS REST API reference?",
            "How do you implement authentication in a REST API?",
            "What are the best practices for API security?",
            "How does load balancing work in microservices?",
            "What is the purpose of health checks in APIs?"
        ]
        
        for question in api_questions:
            print(f"\n🔐 API Question: {question}")
            print("-" * 60)
            
            # Test with hybrid search for better results
            payload = {
                "question": question,
                "search_mode": "full_hybrid",
                "top_k": 5,
                "generate_answer": True
            }
            
            response = requests.post(self.query_endpoint, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                answer = data.get("llm_response", {}).get("answer", "")
                
                print(f"✅ Found {len(results)} documents")
                print(f"📖 Answer: {answer[:300]}...")
                
                # Show result distribution
                if results:
                    print("📚 Document Sources:")
                    for i, result in enumerate(results[:3], 1):
                        filename = result.get('filename', 'Unknown')
                        search_type = result.get('search_type', 'Unknown')
                        print(f"   {i}. {filename} ({search_type})")
            else:
                print(f"❌ Search failed: {response.status_code}")
            
            print("=" * 60)
    
    def test_06_result_count_variations(self):
        """Test different result count configurations."""
        print("\n🔍 Testing Result Count Variations...")
        
        test_question = "What is a Dockerfile and how does it work?"
        
        for count in self.result_counts:
            print(f"Testing result count: {count}")
            
            payload = {
                "question": test_question,
                "search_mode": "full_hybrid",
                "top_k": count,
                "generate_answer": True
            }
            
            response = requests.post(self.query_endpoint, json=payload, timeout=60)
            self.assertEqual(response.status_code, 200, f"Failed for count: {count}")
            
            data = response.json()
            self.assertEqual(data["status"], "success")
            
            results = data["results"]
            self.assertLessEqual(len(results), count, f"Returned {len(results)} results, expected max {count}")
            self.assertGreater(len(results), 0, f"No results returned for count: {count}")
            
            print(f"✅ Count {count}: {len(results)} results returned")
    
    def test_07_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n🔍 Testing Edge Cases...")
        
        # Test empty question
        print("Testing empty question...")
        payload = {
            "question": "",
            "search_mode": "full_hybrid",
            "top_k": 5,
            "generate_answer": True
        }
        
        response = requests.post(self.query_endpoint, json=payload, timeout=30)
        self.assertIn(response.status_code, [400, 422])  # Should return validation error
        
        # Test very long question
        print("Testing very long question...")
        long_question = "What is a Dockerfile? " * 100  # Very long question
        payload = {
            "question": long_question,
            "search_mode": "full_hybrid",
            "top_k": 5,
            "generate_answer": True
        }
        
        response = requests.post(self.query_endpoint, json=payload, timeout=60)
        self.assertEqual(response.status_code, 200)  # Should handle long questions
        
        # Test invalid search mode
        print("Testing invalid search mode...")
        payload = {
            "question": "What is a Dockerfile?",
            "search_mode": "invalid_mode",
            "top_k": 5,
            "generate_answer": True
        }
        
        response = requests.post(self.query_endpoint, json=payload, timeout=30)
        self.assertIn(response.status_code, [400, 422])  # Should return validation error
        
        print("✅ Edge case tests completed")
    
    def test_08_response_quality(self):
        """Test response quality and consistency."""
        print("\n🔍 Testing Response Quality...")
        
        test_question = "What is a Dockerfile and how does it work?"
        
        # Test multiple runs for consistency
        responses = []
        for i in range(3):
            print(f"Quality test run {i + 1}/3")
            
            payload = {
                "question": test_question,
                "search_mode": "full_hybrid",
                "top_k": 5,
                "generate_answer": True
            }
            
            response = requests.post(self.query_endpoint, json=payload, timeout=60)
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            responses.append({
                "answer": data["llm_response"]["answer"],
                "result_count": len(data["results"]),
                "search_mode": data["search_mode"]
            })
        
        # Verify consistency
        result_counts = [r["result_count"] for r in responses]
        self.assertTrue(all(count > 0 for count in result_counts), "All runs should return results")
        
        # Verify answer quality
        for response in responses:
            answer = response["answer"]
            self.assertGreater(len(answer), 20, "Answer should be substantial")
            # Check if answer mentions Docker or related concepts
            docker_terms = ["docker", "container", "image", "file", "build"]
            has_docker_content = any(term in answer.lower() for term in docker_terms)
            self.assertTrue(has_docker_content, "Answer should mention Docker-related concepts")
        
        print("✅ Response quality tests passed")
    
    def test_09_performance_benchmark(self):
        """Test API performance with timing measurements."""
        print("\n🔍 Testing API Performance...")
        
        test_questions = [
            "What is a Dockerfile?",
            "What are CSRF tokens?",
            "How do you create a Docker container?",
            "What is the difference between Docker and Docker Compose?",
            "How do you implement authentication in a REST API?"
        ]
        
        performance_data = []
        
        for question in test_questions:
            print(f"Performance testing: {question[:40]}...")
            
            start_time = time.time()
            
            payload = {
                "question": question,
                "search_mode": "full_hybrid",
                "top_k": 5,
                "generate_answer": True
            }
            
            response = requests.post(self.query_endpoint, json=payload, timeout=120)
            end_time = time.time()
            
            self.assertEqual(response.status_code, 200)
            
            response_time = end_time - start_time
            performance_data.append({
                "question": question[:40],
                "response_time": response_time,
                "status": "success"
            })
            
            print(f"✅ Response time: {response_time:.2f}s")
        
        # Performance analysis
        avg_response_time = sum(p["response_time"] for p in performance_data) / len(performance_data)
        print(f"📊 Average response time: {avg_response_time:.2f}s")
        
        # Verify reasonable performance (should be under 30 seconds on average)
        self.assertLess(avg_response_time, 30.0, f"Average response time {avg_response_time:.2f}s is too slow")
        
        print("✅ Performance tests completed")
    
    def test_10_integration_scenarios(self):
        """Test realistic integration scenarios."""
        print("\n🔍 Testing Integration Scenarios...")
        
        # Scenario 1: Technical question (Docker)
        print("Testing technical question scenario (Docker)...")
        tech_payload = {
            "question": "How does Docker containerization work and what are its benefits?",
            "search_mode": "full_hybrid",
            "top_k": 10,
            "generate_answer": True
        }
        
        response = requests.post(self.query_endpoint, json=tech_payload, timeout=60)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(len(data["results"]), 0)
        
        # Scenario 2: Security question (CSRF)
        print("Testing security question scenario (CSRF)...")
        security_payload = {
            "question": "What are CSRF tokens and how do they protect against attacks?",
            "search_mode": "full_hybrid",
            "top_k": 8,
            "generate_answer": True
        }
        
        response = requests.post(self.query_endpoint, json=security_payload, timeout=60)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(len(data["results"]), 0)
        
        # Scenario 3: Comparison question (Docker vs alternatives)
        print("Testing comparison question scenario...")
        compare_payload = {
            "question": "Compare Docker with other containerization technologies like Podman or LXC",
            "search_mode": "full_hybrid",
            "top_k": 6,
            "generate_answer": True
        }
        
        response = requests.post(self.query_endpoint, json=compare_payload, timeout=60)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(len(data["results"]), 0)
        
        print("✅ Integration scenario tests completed")


def run_tests():
    """Run all tests with detailed output."""
    print("🚀 Starting Query API Tests with Practical Questions...")
    print("=" * 80)
    print("🎯 Focus: Docker, FortiOS, API questions with BM25 first, then hybrid")
    print("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQueryAPI)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=open('/dev/stdout', 'w'))
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
