import logging
import os
import re
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_classic.retrievers import ContextualCompressionRetriever
from app.config import settings
from app.services.reranking import HFServerlessReranker

logger = logging.getLogger("chatbot")

class LLMService:
    def __init__(self):
        try:
            self.embeddings = HuggingFaceEndpointEmbeddings(
                model="BAAI/bge-large-en-v1.5",
                huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
            )
            self.supabase_client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
            self.db = SupabaseVectorStore(
                client=self.supabase_client,
                embedding=self.embeddings,
                table_name="documents",
                query_name="match_documents"
            )

            base_retriever = self.db.as_retriever(search_kwargs={"k": 20})
            compressor = HFServerlessReranker(
                api_token=settings.HUGGINGFACEHUB_API_TOKEN,
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=4
            )
            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
        except Exception as e:
            logger.error(f"Error connecting to DB/Reranker: {e}")
            self.db, self.retriever = None, None

        try:
            raw_llm = HuggingFaceEndpoint(
                repo_id=settings.HF_MODEL_REPO,
                temperature=0.3, # Lowered temperature for more factual responses
                max_new_tokens=512,
                huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
            )
            self.llm = ChatHuggingFace(llm=raw_llm)
        except Exception as e:
            logger.error(f"Error configuring LLM: {e}")
            self.llm = None

    async def generate_response(self, message: str) -> str:
        text = message.lower().strip()

        if not self.llm or not self.db:
            return "System is currently unavailable or disconnected."

        context_string = ""
        try:
            # 1. Routing Filter
            search_filter = {}
            if any(k in text for k in ["alumni", "graduated", "passed out", "senior", "passout"]):
                search_filter = {"source": "alumni"}
            elif any(k in text for k in ["event", "workshop", "hackathon"]):
                search_filter = {"source": "events"}
            
            # 2. Extract Exact Years manually FIRST to guarantee they aren't lost
            exact_match_docs = []
            years_in_prompt = re.findall(r'\b20\d{2}\b', text)
            if years_in_prompt:
                broad_docs = await self.db.asimilarity_search(
                    message, k=100, filter=search_filter if search_filter else None
                )
                for doc in broad_docs:
                    if any(year in doc.page_content for year in years_in_prompt):
                        # Force the LLM to understand what this data is
                        tagged_content = f"[VERIFIED ALUMNI PROFILE]\n{doc.page_content}"
                        exact_match_docs.append(tagged_content)

            # 3. Standard Reranked Retrieval
            compressed_docs = []
            if self.retriever:
                self.retriever.base_retriever.search_kwargs = {"k": 20, "filter": search_filter if search_filter else None}
                retrieved_objects = await self.retriever.ainvoke(message)
                compressed_docs = [doc.page_content for doc in retrieved_objects]

            # 4. Combine: Put exact year matches at the very top
            final_chunks = exact_match_docs + [c for c in compressed_docs if c not in exact_match_docs]
            context_string = "\n\n---\n\n".join(final_chunks[:8])

            # 🔥 THE ULTIMATE DEBUGGER: Watch your Uvicorn terminal when you ask the question
            print("\n" + "="*50)
            print(f"🧠 WHAT THE AI IS READING ({len(final_chunks)} chunks found):")
            print(context_string)
            print("="*50 + "\n")

        except Exception as e:
            logger.error(f"Error during retrieval: {e}")

        # Upgraded system prompt to be more lenient with formatting
        system_instruction = (
            "You are the official GLUG Chatbot of NIT Durgapur. Answer the user's question directly using ONLY the provided context below. "
            "If the context contains profiles with names and years, safely assume they correspond to the members or alumni being asked about. "
            "List their names clearly. If you truly cannot find the answer in the text, apologize."
        )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Context:\n{context_string}\n\nUser Question: {message}")
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            return f"Error connecting to LLM server: {e}"

llm_service = LLMService()
def get_llm_service() -> LLMService:
    return llm_service