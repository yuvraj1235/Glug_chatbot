# app/reranker.py
import logging
import httpx
from typing import Optional, Sequence
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.callbacks import Callbacks
from pydantic import Field

logger = logging.getLogger("chatbot")

class HFServerlessReranker(BaseDocumentCompressor):
    api_token: str
    model_name: str = Field(default="BAAI/bge-reranker-base")
    top_n: int = Field(default=4)

    model_config = {
        "protected_namespaces": (),
    }

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": [
                [query, doc.page_content]
                for doc in documents
            ],
            "options": {
                "wait_for_model": True
            }
        }

        try:
            with httpx.Client() as client:
                response = client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )

            if response.status_code != 200:
                logger.warning(f"[Reranker] HF returned {response.status_code}. Falling back.")
                return list(documents)[:self.top_n]

            scores = response.json()

            if isinstance(scores, dict) and scores.get("error"):
                logger.warning(f"[Reranker] HF error: {scores['error']}. Falling back.")
                return list(documents)[:self.top_n]

            parsed_scores = []
            for s in scores:
                if isinstance(s, (int, float)):
                    parsed_scores.append(float(s))
                elif isinstance(s, dict):
                    parsed_scores.append(float(s.get("score", 0.0)))
                elif isinstance(s, list) and s and isinstance(s[0], dict):
                    best = max(s, key=lambda x: x.get("score", 0.0))
                    parsed_scores.append(float(best.get("score", 0.0)))
                else:
                    parsed_scores.append(0.0)

            if len(parsed_scores) != len(documents):
                logger.warning("[Reranker] Score/doc count mismatch. Falling back.")
                return list(documents)[:self.top_n]

            scored_docs = sorted(
                zip(documents, parsed_scores),
                key=lambda x: x[1],
                reverse=True
            )

            top_docs = [doc for doc, score in scored_docs[:self.top_n]]
            logger.info(f"[Reranker] Reranked {len(documents)} → top {len(top_docs)}")
            return top_docs

        except Exception as e:
            logger.error(f"[Reranker] Failed: {e}. Falling back.")
            return list(documents)[:self.top_n]


class CohereReranker(BaseDocumentCompressor):
    api_token: str
    model_name: str = Field(default="rerank-v3.5")
    top_n: int = Field(default=4)

    model_config = {
        "protected_namespaces": (),
    }

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        url = "https://api.cohere.com/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": self.top_n
        }

        try:
            with httpx.Client() as client:
                response = client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )

            if response.status_code != 200:
                logger.warning(f"[Reranker] Cohere returned {response.status_code}. Falling back to default order.")
                return list(documents)[:self.top_n]

            res_data = response.json()
            results = res_data.get("results", [])

            reranked_docs = []
            for item in results:
                idx = item.get("index")
                if idx is not None and 0 <= idx < len(documents):
                    reranked_docs.append(documents[idx])

            logger.info(f"[Reranker] Cohere Reranked {len(documents)} → top {len(reranked_docs)}")
            return reranked_docs

        except Exception as e:
            logger.error(f"[Reranker] Cohere Failed: {e}. Falling back.")
            return list(documents)[:self.top_n]