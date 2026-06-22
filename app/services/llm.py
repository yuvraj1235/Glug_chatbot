import logging
import re
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings

logger = logging.getLogger("chatbot")

class LLMService:
    def __init__(self):
        # --- DB + Retriever ---
        try:
            self.embeddings = HuggingFaceEndpointEmbeddings(
                model="BAAI/bge-large-en-v1.5",
                huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
            )
            self.supabase_client: Client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_KEY
            )
            self.db = SupabaseVectorStore(
                client=self.supabase_client,
                embedding=self.embeddings,
                table_name="documents",
                query_name="match_documents"
            )
            base_retriever = self.db.as_retriever(search_kwargs={"k": 20})
            
            # Reranker disabled - using only vector DB relevance scoring
            self.compressor = None
            self.retriever = base_retriever
            logger.info("DB and Retriever initialized successfully (reranker disabled)")
        except Exception as e:
            logger.error(f"Error connecting to DB/Retriever: {e}")
            self.db, self.retriever = None, None

        # --- LLM ---
        try:
            model_name = settings.DEFAULT_MODEL
            if "gemini" in model_name.lower():
                model_name = "llama-3.1-8b-instant"
                logger.warning(f"Gemini model config detected on Groq endpoint. Swapping to fallback: {model_name}")

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

    async def generate_response(self, message: str) -> str:
        text = message.lower().strip()

        if not self.llm or not self.db:
            return "System is currently unavailable or disconnected."

        context_string = ""
        try:
            # 1. Routing filter by topic
            search_filter = {}
            if any(k in text for k in ["alumni", "graduated", "passed out", "senior", "passout"]):
                search_filter = {"source": "alumni"}
            elif any(k in text for k in ["event", "workshop", "hackathon"]):
                search_filter = {"source": "events"}

            # 2. Year-based exact match boost (CAP AT 1 MATCH)
            exact_match_docs = []
            years_in_prompt = re.findall(r'\b20\d{2}\b', text)
            if years_in_prompt:
                broad_docs = await self.db.asimilarity_search(
                    message, k=30,
                    filter=search_filter if search_filter else None
                )
                for doc in broad_docs:
                    if any(year in doc.page_content for year in years_in_prompt):
                        tagged = f"[VERIFIED PROFILE]\n{doc.page_content}"
                        exact_match_docs.append(tagged)
                        if len(exact_match_docs) >= 1:
                            break

            # 3. Local Reranked retrieval 
            compressed_docs = []
            if self.retriever:
                self.retriever.search_kwargs = {
                    "k": 10,  # Feed 10 documents to FlashRank to score locally
                    "filter": search_filter if search_filter else None
                }
                retrieved = await self.retriever.ainvoke(message)
                
                if hasattr(self, 'compressor') and self.compressor:
                    # Executes safely in an isolated threadpool to prevent blocking FastAPI
                    reranked = await self.compressor.acompress_documents(retrieved, message)
                    compressed_docs = [doc.page_content for doc in reranked]
                else:
                    logger.debug("Reranker unavailable. Using baseline retrieval fallback.")
                    compressed_docs = [doc.page_content for doc in retrieved[:1]]

            # 4. Merge: year matches first, then reranked chunks, deduplicated
            final_chunks = exact_match_docs + [
                c for c in compressed_docs if c not in exact_match_docs
            ]
            
            # Allowed up to 2 high-quality reranked chunks since FlashRank minimizes noise
            context_string = "\n\n---\n\n".join(final_chunks[:2])

            # AGGRESSIVE SAFEGUARD: Force context string below 8000 chars (~2000 tokens)
            max_context_chars = 8000
            if len(context_string) > max_context_chars:
                logger.warning(f"Context string too large ({len(context_string)} chars)! Truncating to {max_context_chars}.")
                context_string = context_string[:max_context_chars]

            logger.debug(f"Retrieved {len(final_chunks[:2])} chunks. Final String length: {len(context_string)}")

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")

        system_instruction = (
            "You are the official GLUG Chatbot of NIT Durgapur. "
            "Answer the user's question directly using ONLY the provided context below. "
            "If the context contains profiles with names and years, safely assume they "
            "correspond to the members or alumni being asked about. "
            "List their names clearly. If you truly cannot find the answer, apologize."
        )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Context:\n{context_string}\n\nUser Question: {message}")
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM invocation error: {e}")
            return f"Error connecting to LLM: {e}"


llm_service = LLMService()

def get_llm_service() -> LLMService:
    return llm_service