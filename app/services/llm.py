import logging
import re
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings
from app.services.reranking import HFServerlessReranker

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
            self.compressor = HFServerlessReranker(
                api_token=settings.HUGGINGFACEHUB_API_TOKEN,
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=4
            )
            self.retriever = base_retriever
            logger.info("DB and retriever initialized successfully")
        except Exception as e:
            logger.error(f"Error connecting to DB/Reranker: {e}")
            self.db, self.retriever = None, None

        # --- LLM ---
        try:
            # Fallback handling in case DEFAULT_MODEL remains pointed to a Gemini string
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

            # 2. Year-based exact match boost (CAP AT 2 MATCHES)
            exact_match_docs = []
            years_in_prompt = re.findall(r'\b20\d{2}\b', text)
            if years_in_prompt:
                broad_docs = await self.db.asimilarity_search(
                    message, k=50,
                    filter=search_filter if search_filter else None
                )
                for doc in broad_docs:
                    if any(year in doc.page_content for year in years_in_prompt):
                        tagged = f"[VERIFIED PROFILE]\n{doc.page_content}"
                        exact_match_docs.append(tagged)
                        # SAFEGUARD: Stop once we hit 2 exact matches
                        if len(exact_match_docs) >= 2:
                            break

            # 3. Reranked retrieval with explicit fallback safety
            compressed_docs = []
            if self.retriever:
                self.retriever.search_kwargs = {
                    "k": 15,
                    "filter": search_filter if search_filter else None
                }
                retrieved = await self.retriever.ainvoke(message)
                
                if hasattr(self, 'compressor') and self.compressor:
                    # CRITICAL: Now using the asynchronous `acompress_documents`
                    reranked = await self.compressor.acompress_documents(retrieved, message)
                    compressed_docs = [doc.page_content for doc in reranked]
                else:
                    logger.warning("Reranker unavailable. Slicing baseline retrieval to top 2 chunks.")
                    compressed_docs = [doc.page_content for doc in retrieved[:2]]

            # 4. Merge: year matches first, then reranked, deduplicated
            final_chunks = exact_match_docs + [
                c for c in compressed_docs if c not in exact_match_docs
            ]
            
            # STRICT CEILING: Maximum 2 chunks total to stay under Groq limits
            context_string = "\n\n---\n\n".join(final_chunks[:2])

            # BRUTE FORCE SAFEGUARD: Force string below ~4000 tokens (approx 16000 chars)
            if len(context_string) > 16000:
                logger.warning(f"Context string too large ({len(context_string)} chars)! Truncating.")
                context_string = context_string[:16000]

            logger.debug(f"Retrieved {len(final_chunks[:2])} chunks. String length: {len(context_string)}")

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