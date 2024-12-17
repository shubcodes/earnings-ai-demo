# earnings_ai_demo/earnings_ai_demo/query.py
from fireworks.client import Fireworks
from typing import List, Dict, Optional
import logging

class QueryInterface:
    def __init__(self, 
                 api_key: str,
                 database_operations,
                 model: str = "accounts/fireworks/models/llama-v3p3-70b-instruct",
                 embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.client = Fireworks(api_key=api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.db = database_operations

    def query(self, query: str, company_ticker: Optional[str] = None, doc_type: Optional[str] = None, num_results: int = 5) -> Dict:
        try:
            query_embedding = self.client.embeddings.create(
                input=[query],
                model=self.embedding_model
            ).data[0].embedding

            filters = {}
            if company_ticker:
                filters["metadata.company_ticker"] = company_ticker
            if doc_type:
                filters["metadata.document_type"] = doc_type

            # Wait for index readiness
            retries = 3
            for attempt in range(retries):
                relevant_docs = self.db.query_similar(query_embedding, limit=num_results, filters=filters)
                if relevant_docs:
                    break
                if attempt < retries - 1:
                    time.sleep(2)

            if not relevant_docs:
                return {
                    "response": "No relevant documents found in the database or index not ready. Please try again.",
                    "sources": []
                }

            context = self._build_context(relevant_docs)
            system_prompt = "You are analyzing MongoDB earnings data. Use only information from the provided context. Include specific numbers and details when available."
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide an answer using only the context."}
                ]
            )

            return {
                "response": response.choices[0].message.content,
                "sources": relevant_docs
            }

        except Exception as e:
            logging.error(f"Query failed: {e}")
            raise

    def _build_context(self, documents: List[Dict]) -> str:
        context_parts = []
        for doc in documents:
            source = f"Source: {doc['metadata'].get('document_type', 'unknown')} - {doc['metadata'].get('filename', 'unknown')}"
            text = doc.get('text', '')[:2000]
            score = doc.get('score', 'N/A')
            context_parts.append(f"{source} (Relevance: {score})\n{text}\n")
        return "\n---\n".join(context_parts)