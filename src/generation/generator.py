"""Answer generator for RAG system with basic guardrails."""

import logging
from typing import Dict, List, Optional

from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates answers using an LLM based on retrieved contexts."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """Builds a prompt for the LLM using the query and retrieved contexts."""
        if not contexts:
            return f"Question: {query}\n\nAnswer: I don't have any relevant documents to answer this question."
        
        # Limit to top 5 contexts to avoid very long prompts
        limited_contexts = contexts[:5]
        
        context_text = "\n\n".join([
            f"Document {i+1} ({ctx.get('filename', 'N/A')}):\n{ctx.get('content', '')}"
            for i, ctx in enumerate(limited_contexts)
        ])

        prompt = f"""Based on the following documents, answer the question clearly and concisely. If the information is not available in the documents, say "I don't have enough information to answer that question."

DOCUMENTS:
{context_text}

QUESTION: {query}

ANSWER:"""
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

        try:
            # Build prompt and generate answer
            prompt = self.build_prompt(query, contexts)
            logger.info(f"Generating answer for query: {query[:50]}...")
            
            llm_answer = self.llm_client.generate(prompt)
            
            # Basic post-processing
            if not llm_answer or llm_answer.strip() == "":
                llm_answer = "I don't have enough information to answer that question."
            
            # Clean up the response if it starts with common prefixes
            prefixes_to_remove = ["Answer:", "ANSWER:", "A:", "Response:"]
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
                "query": query
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
