"""Simple Ollama LLM client for RAG system."""

import requests
import json
import os
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with the Ollama LLM service."""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or os.getenv("LLM_SERVICE_URL", "http://ollama:11434")
        self.model = model or os.getenv("LLM_MODEL_NAME", "tinyllama")
        logger.info(f"LLMClient initialized with base_url: {self.base_url}, model: {self.model}")

    def generate(self, prompt: str, model: str = None, stream: bool = False) -> str:
        """Generates text using the Ollama LLM service."""
        target_model = model or self.model
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": target_model,
                    "prompt": prompt,
                    "stream": stream
                },
                timeout=float(os.getenv("LLM_TIMEOUT", 180))
            )
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                full_response = ""
                for chunk in response.iter_lines():
                    if chunk:
                        decoded_chunk = json.loads(chunk.decode('utf-8'))
                        full_response += decoded_chunk.get("response", "")
                return full_response
            else:
                return response.json().get("response", "")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return f"Error: Could not connect to LLM service. {e}"
        except Exception as e:
            logger.error(f"Unexpected error during LLM generation: {e}")
            return f"Error: An unexpected error occurred. {e}"

    def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
