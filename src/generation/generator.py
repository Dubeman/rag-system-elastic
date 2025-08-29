"""Answer generator for RAG system with basic guardrails."""

import logging
from typing import Dict, List, Optional

from .llm_client import LLMClient
from ..guardrails.guardrails import ContentSafetyGuardrails

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates answers using an LLM based on retrieved contexts."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.guardrails = ContentSafetyGuardrails()  # Add this line

    def check_context_relevance(self, query: str, contexts: List[Dict]) -> bool:
        """Check if retrieved contexts are actually relevant to the query."""
        if not contexts:
            return False
        
        # Extract key terms from query (remove common words)
        stop_words = {'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'how', 'to', 'make', 'create', 'build', 'develop', 'do', 'you', 'can', 'tell', 'me', 'about', 'explain', 'describe'}
        query_terms = set(query.lower().split()) - stop_words
        
        if not query_terms:
            return True  # If query has no meaningful terms, allow it
        
        # Check relevance of each context
        relevant_contexts = 0
        for context in contexts:
            content = context.get('content', '')
            if not content:
                continue
                
            context_terms = set(content.lower().split()) - stop_words
            
            # Calculate overlap
            overlap = len(query_terms.intersection(context_terms))
            relevance_score = overlap / len(query_terms)
            
            # If any context is relevant, consider it valid
            if relevance_score >= 0.2:  # 20% threshold
                relevant_contexts += 1
        
        # Require at least 50% of contexts to be relevant
        return relevant_contexts >= len(contexts) * 0.5

    def check_content_safety(self, query: str) -> bool:
        """Check if the query is safe to process."""
        dangerous_topics = [
            'nuclear weapon', 'bomb', 'explosive', 'weapon', 'hack', 'cyber attack',
            'illegal', 'criminal', 'harmful', 'dangerous', 'toxic', 'poison',
            'kill', 'murder', 'suicide', 'terrorism', 'drugs', 'weapons'
        ]
        
        query_lower = query.lower()
        for topic in dangerous_topics:
            if topic in query_lower:
                return False
        
        return True

    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """Builds a safe, grounded prompt for the LLM using the query and retrieved contexts."""
        if not contexts:
            return f"Question: {query}\n\nAnswer: I don't have any relevant documents to answer this question."
        
        # Limit to top 5 contexts to avoid very long prompts
        limited_contexts = contexts[:5]
        
        context_text = "\n\n".join([
            f"Document {i+1} ({ctx.get('filename', 'N/A')}):\n{ctx.get('content', '')}"
            for i, ctx in enumerate(limited_contexts)
        ])

        prompt = f"""IMPORTANT SAFETY INSTRUCTIONS:
- You MUST ONLY use information from the provided documents
- You MUST NOT make up facts, speculate, or provide information not in the documents
- You MUST NOT provide instructions for dangerous, illegal, or harmful activities
- If the documents don't contain relevant information, say "I cannot answer this question based on the available documents"
- Always cite which document(s) you used for your answer

DOCUMENTS:
{context_text}

QUESTION: {query}

SAFE ANSWER (using only the documents above):"""
        return prompt

    def format_citations(self, contexts: List[Dict]) -> List[Dict]:
        """Formats citations from the retrieved contexts."""
        citations = []
        for i, ctx in enumerate(contexts[:5]):  # Limit to 5 citations
            citations.append({
                "source_id": i + 1,
                "filename": ctx.get("filename", "N/A"),
                "chunk_id": ctx.get("chunk_id", "N/A"),
                "content_excerpt": ctx.get("content", "")[:200] + "..." if len(ctx.get("content", "")) > 200 else ctx.get("content", ""),
                "score": round(ctx.get("_score", ctx.get('score', 0)), 4),
                "file_url": ctx.get("file_url", ctx.get('source_url', 'N/A'))
            })
        return citations

    def generate_with_citations(self, query: str, contexts: List[Dict]) -> Dict:
        """Generates an answer and citations from the LLM."""
        
        # Check if we have contexts
        if not contexts:
            return {
                "answer": "I don't have enough information to answer that question as no relevant documents were retrieved.",
                "citations": [],
                "status": "no_documents",
                "model_used": self.llm_client.model
            }

        # GUARDRAIL 1: Content safety check
        if not self.check_content_safety(query):
            return {
                "answer": "I cannot and will not provide information about dangerous or harmful topics.",
                "status": "content_blocked",
                "model_used": self.llm_client.model,
                "reason": "Query blocked for safety reasons"
            }

        # GUARDRAIL 2: Context relevance check
        if not self.check_context_relevance(query, contexts):
            return {
                "answer": "I cannot answer this question based on the provided documents. The retrieved documents are not relevant to your query.",
                "status": "irrelevant_documents",
                "model_used": self.llm_client.model,
                "query": query,
                "contexts_checked": len(contexts)
            }

        try:
            # Build prompt and generate answer
            prompt = self.build_prompt(query, contexts)
            logger.info(f"Generating answer for query: {query[:50]}...")
            
            llm_answer = self.llm_client.generate(prompt)
            
            # Enhanced post-processing and validation
            if not llm_answer or llm_answer.strip() == "":
                llm_answer = "I don't have enough information to answer that question."
            
            # Check for truncated responses (common indicators)
            truncated_indicators = [
                "...", "etc", "and so on", "continues", "more", "further", 
                "truncated", "cut off", "incomplete"
            ]
            
            is_truncated = any(indicator in llm_answer.lower() for indicator in truncated_indicators)
            if is_truncated:
                logger.warning(f"Detected potentially truncated response for query: {query[:50]}...")
                # Try to regenerate with a more focused prompt
                limited_contexts = contexts[:5] # Re-limit contexts for the focused prompt
                context_text = "\n\n".join([
                    f"Document {i+1} ({ctx.get('filename', 'N/A')}):\n{ctx.get('content', '')}"
                    for i, ctx in enumerate(limited_contexts)
                ])
                focused_prompt = f"""Based on the documents provided, give a complete answer to: {query}

Focus on providing a complete, coherent response. If you cannot complete the answer, please say so explicitly.

Documents:
{context_text}

Complete answer:"""
                
                llm_answer = self.llm_client.generate(focused_prompt)
            
            # Clean up the response if it starts with common prefixes
            prefixes_to_remove = ["Answer:", "ANSWER:", "A:", "Response:", "Please provide a complete answer:"]
            for prefix in prefixes_to_remove:
                if llm_answer.strip().startswith(prefix):
                    llm_answer = llm_answer.strip()[len(prefix):].strip()

            citations = self.format_citations(contexts)

            return {
                "answer": llm_answer,
                "citations": citations,
                "status": "success",
                "model_used": self.llm_client.model,
                "num_contexts": len(contexts),
                "query": query,
                "response_quality": "complete" if not is_truncated else "potentially_truncated"
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return {
                "answer": "I'm sorry, there was an error generating an answer to your question.",
                "citations": self.format_citations(contexts) if contexts else [],
                "status": "error",
                "error": str(e),
                "model_used": self.llm_client.model
            }
