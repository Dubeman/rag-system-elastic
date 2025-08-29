"""Simplified guardrails for RAG system without external dependencies."""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class ContentSafetyGuardrails:
    """Content safety and moderation guardrails."""
    
    def __init__(self):
        self.harmful_patterns = [
            'nuclear weapon', 'bomb', 'explosive', 'weapon', 'hack', 'cyber attack',
            'illegal', 'criminal', 'harmful', 'dangerous', 'toxic', 'poison',
            'kill', 'murder', 'suicide', 'terrorism', 'drugs', 'weapons'
        ]
    
    def validate_query(self, query_data: Dict) -> Dict:
        """Validate query data."""
        try:
            question = query_data.get('question', '')
            top_k = query_data.get('top_k', 5)
            search_mode = query_data.get('search_mode', 'dense_bm25')
            
            # Basic validation
            if not question or len(question) < 3 or len(question) > 1000:
                return {"valid": False, "error": "Question must be between 3 and 1000 characters"}
            
            if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
                return {"valid": False, "error": "top_k must be between 1 and 20"}
            
            valid_modes = ['bm25_only', 'dense_only', 'elser_only', 'dense_bm25', 'full_hybrid']
            if search_mode not in valid_modes:
                return {"valid": False, "error": f"search_mode must be one of {valid_modes}"}
            
            return {"valid": True, "data": query_data}
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
