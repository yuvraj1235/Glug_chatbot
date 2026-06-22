import logging
import asyncio
from typing import Optional, Sequence, Any
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.callbacks import Callbacks
from pydantic import Field, PrivateAttr

logger = logging.getLogger("chatbot")

class FlashRankReranker(BaseDocumentCompressor):
    model_name: str = Field(default="ms-marco-MiniLM-L-6-v2")
    top_n: int = Field(default=4)
    
    # PrivateAttr tells Pydantic to ignore this field during schema validation
    _ranker: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from flashrank import Ranker
        logger.info(f"Initializing FlashRank with local model: {self.model_name}")
        # This downloads the ~85MB model directly into your container the first time it boots
        self._ranker = Ranker(model_name=self.model_name)

    # ==========================================
    # 1. SYNCHRONOUS METHOD (Core Logic)
    # ==========================================
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        from flashrank import RerankRequest
        
        # FlashRank expects passages formatted as a list of dictionaries
        passages = [
            {"id": i, "text": doc.page_content, "metadata": doc.metadata}
            for i, doc in enumerate(documents)
        ]

        try:
            req = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(req)
            
            # Re-map the sorted results back to LangChain Document objects
            reranked_docs = []
            for res in results[:self.top_n]:
                doc_id = res["id"]
                original_doc = documents[doc_id]
                # Inject the local FlashRank score into the document metadata for debugging
                original_doc.metadata["relevance_score"] = res["score"]
                reranked_docs.append(original_doc)
                
            return reranked_docs
            
        except Exception as e:
            logger.error(f"FlashRank execution failed: {e}. Falling back to baseline.")
            return list(documents)[:self.top_n]

    # ==========================================
    # 2. ASYNCHRONOUS METHOD (FastAPI non-blocking wrapper)
    # ==========================================
    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Runs the synchronous FlashRank CPU calculations in a background thread to prevent pausing your server."""
        if not documents:
            return []
            
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            self.compress_documents, 
            documents, 
            query, 
            callbacks
        )