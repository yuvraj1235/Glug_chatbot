import logging
import httpx
from typing import Optional, Sequence, Any
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.callbacks import Callbacks
from pydantic import Field 

logger = logging.getLogger("chatbot")

class HFServerlessReranker(BaseDocumentCompressor):
    api_token: str
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_n: int = Field(default=4)

    model_config = {
        "protected_namespaces": (),
    }

    # ==========================================
    # 1. SYNCHRONOUS METHOD (Fallback/LangChain Sync)
    # ==========================================
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": [{"text": query, "text_pair": doc.page_content} for doc in documents]
        }

        try:
            # FORCE IPv4 to fix Docker DNS [Errno -5]
            transport = httpx.HTTPTransport(local_address="0.0.0.0")
            
            # Use standard Client for the synchronous method
            with httpx.Client(transport=transport) as client:
                response = client.post(url, headers=headers, json=payload, timeout=15.0)

            if response.status_code != 200:
                logger.warning(f"HF Reranker API returned status {response.status_code}. Using fallback.")
                return list(documents)[:self.top_n]

            scores = response.json()
            parsed_scores = self._parse_scores(scores)

            scored_docs = sorted(zip(documents, parsed_scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:self.top_n]]

        except Exception as e:
            logger.error(f"HF Reranker sync execution failed: {e}. Falling back to baseline.")
            return list(documents)[:self.top_n]

    # ==========================================
    # 2. ASYNCHRONOUS METHOD (Used by your FastAPI app)
    # ==========================================
    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": [{"text": query, "text_pair": doc.page_content} for doc in documents]
        }

        try:
            # FORCE IPv4 for Async requests to fix Docker DNS [Errno -5]
            transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0")
            
            # Use AsyncClient with 'async with' and 'await'
            async with httpx.AsyncClient(transport=transport) as client:
                response = await client.post(url, headers=headers, json=payload, timeout=15.0)

            if response.status_code != 200:
                logger.warning(f"HF Reranker API returned status {response.status_code}. Using fallback.")
                return list(documents)[:self.top_n]

            scores = response.json()
            parsed_scores = self._parse_scores(scores)

            scored_docs = sorted(zip(documents, parsed_scores), key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs[:self.top_n]]

        except Exception as e:
            logger.error(f"HF Reranker async execution failed: {e}. Falling back to baseline.")
            return list(documents)[:self.top_n]

    # ==========================================
    # 3. HELPER METHOD (To avoid repeating code)
    # ==========================================
    def _parse_scores(self, scores: Any) -> list[float]:
        """Safely extracts scores from HF's unpredictable JSON formats."""
        parsed_scores = []
        if isinstance(scores, list):
            for s in scores:
                if isinstance(s, dict):
                    parsed_scores.append(s.get('score', 0.0))
                elif isinstance(s, list) and len(s) > 0 and isinstance(s[0], dict):
                    parsed_scores.append(s[0].get('score', 0.0))
                else:
                    parsed_scores.append(float(s))
        else:
            raise ValueError(f"Unexpected response format from HF API: {scores}")
        return parsed_scores