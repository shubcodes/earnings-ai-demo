# earnings_ai_demo/earnings_ai_demo/embedding.py 
from typing import List
import fireworks.client
from fireworks.client import Fireworks
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbeddingGenerator:
    def __init__(self, api_key: str, model: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.client = Fireworks(api_key=api_key)
        self.model = model
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_embedding(self, text: str, prefix: str = "") -> List[float]:
        if prefix:
            text = prefix + text
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding

    def generate_embeddings_batch(self, texts: List[str], prefix: str = "", chunk_size: int = 1000) -> List[List[float]]:
        if prefix:
            texts = [prefix + text for text in texts]
            
        embeddings = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            response = self.client.embeddings.create(
                input=chunk,
                model=self.model
            )
            embeddings.extend([d.embedding for d in response.data])
            
        return embeddings

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
        
    def generate_document_embedding(self, text: str, prefix: str = "", method: str = "mean") -> List[float]:
        chunks = self._chunk_text(text)
        chunk_embeddings = self.generate_embeddings_batch(chunks, prefix)
        
        if method == "mean":
            return np.mean(chunk_embeddings, axis=0).tolist()
        elif method == "max":
            return np.max(chunk_embeddings, axis=0).tolist()
        else:
            raise ValueError(f"Unsupported combining method: {method}")