"""Advanced guardrails using Outer SDK for RAG system."""

from typing import Dict, List, Optional
from guardrails import Guard
from guardrails.validators import ValidLength, ValidRange, ValidChoices
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class QueryGuardrails(BaseModel):
    """Guardrails for user queries."""
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(..., ge=1, le=20)
    search_mode: str = Field(..., regex="^(bm25_only|dense_only|elser_only|dense_bm25|full_hybrid)$")

class ContentSafetyGuardrails:
    """Content safety and moderation guardrails."""
    
    def __init__(self):
        self.harmful_patterns = [
            'nuclear weapon', 'bomb', 'explosive', 'weapon', 'hack', 'cyber attack',
            'illegal', 'criminal', 'harmful', 'dangerous', 'toxic', 'poison',
            'kill', 'murder', 'suicide', 'terrorism', 'drugs', 'weapons'
        ]
        
        # Initialize Outer SDK guard
        self.query_guard = Guard.from_pydantic(QueryGuardrails)
    
    def validate_query(self, query_data: Dict) -> Dict:
        """Validate query using Outer SDK."""
        try:
            validated = self.query_guard(query_data)
            return {"valid": True, "data": validated}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def check_content_safety(self, text: str) -> Dict:
        """Check content for harmful content."""
        text_lower = text.lower()
        detected_harmful = []
        
        for pattern in self.harmful_patterns:
            if pattern in text_lower:
                detected_harmful.append(pattern)
        
        return {
            "safe": len(detected_harmful) == 0,
            "harmful_patterns": detected_harmful,
            "risk_level": "high" if detected_harmful else "low"
        }
    
    def apply_response_guardrails(self, response: str) -> Dict:
        """Apply guardrails to LLM responses."""
        safety_check = self.check_content_safety(response)
        
        if not safety_check["safe"]:
            return {
                "safe_response": "I cannot provide that information due to safety concerns.",
                "original_response": response,
                "safety_issues": safety_check["harmful_patterns"]
            }
        
        return {"safe_response": response, "safety_issues": []}
