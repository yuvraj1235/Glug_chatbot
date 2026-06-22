import logging
import re
import math
import asyncio
from supabase.client import Client, create_client
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings
from app.services.reranker import HFServerlessReranker, CohereReranker

logger = logging.getLogger("chatbot")

# In-memory document cache to optimize search performance
_cached_documents = []
_cache_lock = asyncio.Lock()

class LocalSupabaseVectorStore:
    def __init__(self, client: Client, embeddings, table_name="documents"):
        self.client = client
        self.embeddings = embeddings
        self.table_name = table_name

    async def _get_all_documents(self) -> list:
        global _cached_documents
        async with _cache_lock:
            if _cached_documents:
                return _cached_documents
            
            try:
                logger.info("Loading documents into memory cache from Supabase...")
                res = self.client.table(self.table_name).select("id, content, metadata, embedding").execute()
                docs = []
                for item in res.data:
                    emb_str = item.get("embedding")
                    if isinstance(emb_str, str):
                        emb_str = emb_str.strip().lstrip("[").rstrip("]")
                        emb = [float(x) for x in emb_str.split(",") if x.strip()]
                    elif isinstance(emb_str, list):
                        emb = [float(x) for x in emb_str]
                    else:
                        emb = []
                    
                    docs.append({
                        "id": item.get("id"),
                        "content": item.get("content"),
                        "metadata": item.get("metadata") or {},
                        "embedding": emb
                    })
                _cached_documents = docs
                logger.info(f"Loaded {len(_cached_documents)} documents into memory cache successfully.")
            except Exception as e:
                logger.error(f"Failed to load documents from database: {e}")
            
            return _cached_documents

    def clear_cache(self):
        global _cached_documents
        _cached_documents = []
        logger.info("In-memory document cache cleared.")

    async def asimilarity_search(self, query: str, k: int = 5, filter: dict = None) -> list[Document]:
        docs = await self._get_all_documents()
        if not docs:
            return []

        if hasattr(self.embeddings, "aembed_query"):
            query_vector = await self.embeddings.aembed_query(query)
        else:
            query_vector = self.embeddings.embed_query(query)

        def dot_product(v1, v2):
            return sum(x * y for x, y in zip(v1, v2))

        def magnitude(v):
            return math.sqrt(sum(x * x for x in v))

        def cosine_similarity(v1, v2):
            m1 = magnitude(v1)
            m2 = magnitude(v2)
            if m1 == 0 or m2 == 0:
                return 0.0
            return dot_product(v1, v2) / (m1 * m2)

        scored_docs = []
        for doc in docs:
            if filter:
                match = True
                for fk, fv in filter.items():
                    if doc["metadata"].get(fk) != fv:
                        match = False
                        break
                if not match:
                    continue

            emb = doc["embedding"]
            if not emb:
                continue

            sim = cosine_similarity(query_vector, emb)
            scored_docs.append((doc, sim))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, sim in scored_docs[:k]:
            results.append(Document(
                page_content=doc["content"],
                metadata={**doc["metadata"], "similarity": sim}
            ))
        return results

    def as_retriever(self, search_kwargs: dict = None):
        return LocalVectorStoreRetriever(self, search_kwargs or {})


class LocalVectorStoreRetriever:
    def __init__(self, vector_store: LocalSupabaseVectorStore, search_kwargs: dict):
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs

    async def ainvoke(self, query: str) -> list[Document]:
        k = self.search_kwargs.get("k", 5)
        filter = self.search_kwargs.get("filter")
        return await self.vector_store.asimilarity_search(query, k=k, filter=filter)


class LLMService:
    def __init__(self):

        # --- DB + Retriever ---
        try:
            self.embeddings = HuggingFaceEndpointEmbeddings(
                model="BAAI/bge-base-en-v1.5",
                huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
            )
            self.supabase_client: Client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_KEY
            )
            self.db = LocalSupabaseVectorStore(
                client=self.supabase_client,
                embeddings=self.embeddings,
                table_name="documents"
            )

            # ✅ Plain base retriever — no wrapper needed
            self.base_retriever = self.db.as_retriever(
                search_kwargs={"k": 20}
            )

            # ✅ Reranker called directly in generate_response
            if settings.COHERE_API_KEY:
                logger.info("Initializing CohereReranker...")
                self.compressor = CohereReranker(
                    api_token=settings.COHERE_API_KEY,
                    model_name="rerank-v3.5",
                    top_n=4
                )
            else:
                logger.info("Initializing HFServerlessReranker...")
                self.compressor = HFServerlessReranker(
                    api_token=settings.HUGGINGFACEHUB_API_TOKEN,
                    model_name="BAAI/bge-reranker-base",
                    top_n=4
                )

            logger.info("DB, Retriever and Reranker initialized successfully")

        except Exception as e:
            logger.error(f"Error connecting to DB/Retriever: {e}")
            self.db = None
            self.base_retriever = None
            self.compressor = None

        # --- LLM ---
        try:
            model_name = settings.DEFAULT_MODEL
            if "gemini" in model_name.lower():
                model_name = "llama-3.1-8b-instant"
                logger.warning(f"Gemini model detected. Swapping to: {model_name}")

            self.llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model=model_name,
                temperature=0.3,
                max_tokens=512
            )
            logger.info(f"LLM initialized: {model_name}")

        except Exception as e:
            logger.error(f"Error configuring LLM: {e}")
            self.llm = None

    def _get_search_filter(self, text: str) -> dict:
        if any(k in text for k in [
            "alumni", "graduated", "passed out",
            "passout", "senior", "batch", "graduate"
        ]):
            return {"source": "alumni"}
        if any(k in text for k in [
            "event", "workshop", "hackathon",
            "competition", "seminar", "fest"
        ]):
            return {"source": "events"}
        if any(k in text for k in [
            "member", "team", "profile",
            "core team", "coordinator"
        ]):
            return {"source": "profiles"}
        if any(k in text for k in [
            "ctf", "capture the flag",
            "cybersecurity", "hacking"
        ]):
            return {"source": "ctf"}
        if any(k in text for k in [
            "project", "open source",
            "repository", "github"
        ]):
            return {"source": "project"}
        if any(k in text for k in [
            "blog", "article", "techbytes",
            "post", "write-up"
        ]):
            return {"source": "techbytes"}
        return {}

    async def _get_year_exact_matches(
        self,
        message: str,
        years: list[str],
        search_filter: dict,
        limit: int = 3
    ) -> list[str]:
        try:
            broad_docs = await self.db.asimilarity_search(
                message,
                k=30,
                filter=search_filter if search_filter else None
            )
            matched = []
            for doc in broad_docs:
                if any(year in doc.page_content for year in years):
                    tagged = f"[VERIFIED PROFILE]\n{doc.page_content}"
                    matched.append(tagged)
                    if len(matched) >= limit:
                        break
            return matched
        except Exception as e:
            logger.error(f"[YearMatch] Failed: {e}")
            return []

    async def generate_response(self, message: str) -> str:
        if not self.llm or not self.db:
            return "System is currently unavailable or disconnected."

        text = message.lower().strip()
        context_string = ""

        try:
            # Step 1 — Route to correct source
            search_filter = self._get_search_filter(text)
            logger.debug(f"[Routing] filter={search_filter}")

            # Step 2 — Year exact match boost
            exact_match_docs = []
            years_in_prompt = re.findall(r'\b20\d{2}\b', text)
            if years_in_prompt:
                exact_match_docs = await self._get_year_exact_matches(
                    message, years_in_prompt, search_filter
                )
                logger.debug(f"[YearMatch] Found {len(exact_match_docs)} matches")

            # Step 3 — Vector search + rerank directly
            reranked_docs = []
            try:
                # Update filter per query
                self.base_retriever.search_kwargs = {
                    "k": 20,
                    "filter": search_filter if search_filter else None
                }

                # ✅ Vector search
                retrieved = await self.base_retriever.ainvoke(message)
                logger.debug(f"[Retriever] Got {len(retrieved)} docs")

                # ✅ Rerank directly — no ContextualCompressionRetriever needed
                if self.compressor and retrieved:
                    reranked = self.compressor.compress_documents(retrieved, message)
                    reranked_docs = [doc.page_content for doc in reranked]
                    logger.debug(f"[Reranker] Returned {len(reranked_docs)} docs")
                else:
                    reranked_docs = [doc.page_content for doc in retrieved[:4]]

            except Exception as e:
                # Fallback to plain vector search
                logger.warning(f"[Retrieval] Failed, using fallback: {e}")
                fallback = await self.db.asimilarity_search(
                    message,
                    k=5,
                    filter=search_filter if search_filter else None
                )
                reranked_docs = [doc.page_content for doc in fallback]

            # Step 4 — Merge deduplicated
            seen = set()
            final_chunks = []
            for chunk in exact_match_docs + reranked_docs:
                if chunk not in seen:
                    seen.add(chunk)
                    final_chunks.append(chunk)

            # Step 5 — Build context, cap at 8000 chars
            context_string = "\n\n---\n\n".join(final_chunks[:5])
            if len(context_string) > 8000:
                context_string = context_string[:8000]

            logger.debug(f"[Context] {len(final_chunks[:5])} chunks, {len(context_string)} chars")

        except Exception as e:
            logger.error(f"[Retrieval] Error: {e}")

        # Step 6 — LLM
        system_instruction = (
            "You are the official GLUG Chatbot of NIT Durgapur. "
            "Answer the user's question directly using ONLY the provided context below. "
            "If the context contains profiles with names and years, safely assume they "
            "correspond to the members or alumni being asked about. "
            "List their names clearly. "
            "If you truly cannot find the answer in the context, say: "
            "'I don't have that information right now.'"
        )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(
                content=f"Context:\n{context_string}\n\nUser Question: {message}"
            )
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"[LLM] Invocation error: {e}")
            return f"Error connecting to LLM: {e}"

    def refresh_cache(self):
        if hasattr(self.db, "clear_cache"):
            self.db.clear_cache()



llm_service = LLMService()

def get_llm_service() -> LLMService:
    return llm_service