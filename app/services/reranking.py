import logging
import httpx
from typing import Optional, Sequence
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.callbacks import Callbacks

logger = logging.getLogger("chatbot")


# --- CUSTOM FREE HUGGING FACE RERANKER ---
class HFServerlessReranker(BaseDocumentCompressor):
    api_token: str
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 4


    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        # Structure the payload exactly how Hugging Face's cross-encoder task expects it
        payload = {
            "inputs": [{"text": query, "text_pair": doc.page_content} for doc in documents]
        }

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=payload, timeout=10.0)

            if response.status_code != 200:
                logger.warning(f"HF Reranker API returned status {response.status_code}. Using fallback ranking.")
                return documents[:self.top_n]

            scores = response.json()

            # If the response returns a list of scores, pair them up and sort
            if isinstance(scores, list):
                # Handle cases where API returns a list of dicts like [{'score': 0.9}, ...]
                parsed_scores = [s['score'] if isinstance(s, dict) else float(s) for s in scores]

                scored_docs = sorted(zip(documents, parsed_scores), key=lambda x: x[1], reverse=True)
                return [doc for doc, score in scored_docs[:self.top_n]]

            return documents[:self.top_n]
        except Exception as e:
            logger.error(f"HF Reranker execution failed: {e}. Falling back to baseline retrieval.")
            return documents[:self.top_n]
