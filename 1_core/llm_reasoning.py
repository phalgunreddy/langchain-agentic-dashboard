import json
from typing import Dict, Any, List

from config import LLM_REASON_MODEL, LLM_EXPLAIN_MODEL, OLLAMA_BASE_URL
from utils import logger
import requests

class LLMReasoning:
    """
    Handles complex reasoning and explanation tasks using larger LLMs.
    This component will take retrieved context and a query, and generate
    a comprehensive response.
    """
    def __init__(self):
        self.reasoning_model = LLM_REASON_MODEL  # Use actual model name
        self.explanation_model = LLM_EXPLAIN_MODEL  # Use actual model name
        logger.info(f"LLMReasoning initialized with reasoning model: {self.reasoning_model} and explanation model: {self.explanation_model}")

    def _call_ollama_llm(self, prompt: str, model: str, temperature: float = 0.2) -> str:
        """Helper to call Ollama LLM with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature, "num_gpu": 1}
                    },
                    timeout=180 # Longer timeout for complex LLM tasks
                )
                response.raise_for_status()
                response_data = response.json()
                return response_data.get("response", "").strip()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for model {model}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Error calling Ollama model {model} after {max_retries} attempts: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"Ollama error response status: {e.response.status_code}")
                        logger.error(f"Ollama error response body: {e.response.text}")
                    return f"LLM Error: Could not get response from LLM model {model} after multiple attempts. Please ensure Ollama is running."
                import time
                time.sleep(2) # Wait before retrying

    def perform_reasoning(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generates a reasoned answer based on the query, provided context, and chat history.
        """
        context_str = "\n---\n".join([f"Document ID: {d.get('doc_id', 'N/A')}\nContent: {d.get('content', 'N/A')}\nMetadata: {json.dumps(d.get('metadata', {}), indent=2)}" for d in context])
        
        history_str = ""
        if chat_history:
            for entry in chat_history[-3:]: # Include last 3 turns of conversation
                history_str += f"\nUser: {entry['query']}\nMendy: {entry['response']}"

        prompt = f"""You are a helpful energy data analyst. Answer questions clearly, informatively, and comprehensively.

Your answers should be:
1. **Direct** - Start with the answer.
2. **Detailed** - Provide specific numbers, comparisons, and insights. Don't be too brief.
3. **Conversational** - Like explaining to a colleague.
4. **Accurate** - Use the exact numbers from the data.
5. **Clean** - **NEVER** repeat the raw "Context data" or "Document ID" blocks in your output. Only use the information to form your answer.

Example good answer:
Q: "What is the top feeder in July 2024?"
A: "The top consumer in July 2024 was **I/C Panel (Location: I/C-1)** with 20,585,392 KWH total consumption. This feeder consistently shows the highest readings across the monitoring period, indicating it is the primary power input for the facility."

Context data:
{context_str}

Question: "{query}"

Answer (be clear, helpful, and DO NOT repeat context):
"""
        logger.info(f"Performing reasoning for query (first 50 chars): {query[:50]}...")
        reasoned_answer = self._call_ollama_llm(prompt, self.reasoning_model, temperature=0.0)

        # Extract provenance from the LLM's response if possible, or infer from context
        provenance_docs = []
        for doc in context:
            doc_id = doc.get('doc_id')
            if doc_id and doc_id in reasoned_answer: # Check if doc_id exists and is in answer
                provenance_docs.append(doc_id)
        
        if not provenance_docs:
            # Fallback: if LLM didn't explicitly mention, include all context doc_ids as potential provenance
            provenance_docs = [d.get('doc_id') for d in context if d.get('doc_id')]

        return {
            "answer": reasoned_answer,
            "type": "Mendy-Reasoning", # Branded type
            "provenance": provenance_docs
        }

    def generate_explanation(self, query: str, context: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generates an explanation or summary based on the query, provided context, and chat history.
        """
        context_str = "\n---\n".join([f"Document ID: {d.get('doc_id', 'N/A')}\nContent: {d.get('content', 'N/A')}\nMetadata: {json.dumps(d.get('metadata', {}), indent=2)}" for d in context])

        history_str = ""
        if chat_history:
            for entry in chat_history[-3:]: # Include last 3 turns of conversation
                history_str += f"\nUser: {entry['query']}\nMendy: {entry['response']}"

        prompt = f"""
        You are Mendy, an AI assistant designed to provide clear, concise, and human-like explanations and summaries.
        Based on the following context documents and previous conversation, provide a comprehensive explanation or summary to answer the query.
        Ensure your explanation is easy to understand and directly addresses the user's request. Always maintain a friendly and professional tone.
        Cite the Document IDs for the information you use.

        Previous Conversation (if any):
        {history_str}

        Query: "{query}"

        Context Documents:
        {context_str}

        Explanation/Summary from Mendy:
        """
        logger.info(f"Generating explanation for query (first 50 chars): {query[:50]}...")
        explanation = self._call_ollama_llm(prompt, self.explanation_model)

        # Extract provenance
        provenance_docs = []
        for doc in context:
            doc_id = doc.get('doc_id')
            if doc_id and doc_id in explanation: # Check if doc_id exists and is in explanation
                provenance_docs.append(doc_id)

        if not provenance_docs:
            provenance_docs = [d.get('doc_id') for d in context if d.get('doc_id')]

        return {
            "answer": explanation,
            "type": "Mendy-Explanation", # Branded type
            "provenance": provenance_docs
        }

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    llm_reasoning = LLMReasoning()

    # Dummy context documents for demonstration
    dummy_context = [
        {
            "doc_id": "doc1",
            "content": "The sales of product A increased by 15% in Q1 due to a successful marketing campaign.",
            "metadata": {"source": "sales_report.pdf", "page": 5}
        },
        {
            "doc_id": "doc2",
            "content": "Product B saw a 5% decrease in sales in the same quarter, possibly due to increased competition.",
            "metadata": {"source": "competitor_analysis.docx", "page": 2}
        }
    ]

    query_reason = "Why did sales of product A increase in Q1?"
    reason_result = llm_reasoning.perform_reasoning(query_reason, dummy_context)
    print(f"\nReasoning Result for \'{query_reason}\':")
    print(f"Answer: {reason_result['answer']}")
    print(f"Provenance: {reason_result['provenance']}")

    query_explain = "Summarize the sales performance of Product B in Q1."
    explain_result = llm_reasoning.generate_explanation(query_explain, dummy_context)
    print(f"\nExplanation Result for \'{query_explain}\':")
    print(f"Answer: {explain_result['answer']}")
    print(f"Provenance: {explain_result['provenance']}")
