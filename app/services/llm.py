import logging
import os
import re
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
# 1. FIXED: Swapped HuggingFaceEndpointEmbeddings for local CPU execution class
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from app.config import settings

logger = logging.getLogger("chatbot")

class LLMService:
    def __init__(self):
        # Initialize Vector Database connection
        try:
            # 2. FIXED: Computes embeddings inside your container runtime. No API token required.
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            supabase_url = settings.SUPABASE_URL
            supabase_key = settings.SUPABASE_SERVICE_KEY

            if not supabase_url or not supabase_key:
                logger.error("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in settings configuration.")
                self.db = None
            else:
                self.supabase_client: Client = create_client(supabase_url, supabase_key)
                self.db = SupabaseVectorStore(
                    client=self.supabase_client,
                    embedding=self.embeddings,
                    table_name="documents",
                    query_name="match_documents"
                )
                logger.info("Supabase Vector DB successfully connected with local embeddings.")
        except Exception as e:
            logger.error(f"Error connecting to Supabase Vector DB: {e}")
            self.db = None

        # Configure Groq Engine
        try:
            self.llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.4,
                groq_api_key=os.environ.get("GROQ_API_KEY")
            )
            logger.info("Groq Chat client successfully configured.")
        except Exception as e:
            logger.error(f"Error configuring Groq Endpoint: {e}")
            self.llm = None

    async def generate_response(self, message: str) -> str:
        text = message.lower().strip()

        # Exact keyword intercepts for academic notes
        academic_patterns = [
            r"\bpyqs?\b",
            r"\bprevious\s+year\s+questions?\b",
            r"\bprevious\s+year\s+papers?\b",
            r"\bquestion\s+papers?\b",
            r"\bexam\s+papers?\b",
            r"\bpast\s+papers?\b",
            r"\bquestion\s+banks?\b",
            r"\bstudy\s+materials?\b",
            r"\bnotes?\b",
            r"\bpdfs?\b",
            r"\bacademic\s+resources?\b",
            r"\blecture\s+notes?\b",
            r"\bclass\s+notes?\b",
            r"\bsyllabus\b",
            r"\bsem(?:ester)?\s*[1-8]\b",
            r"\b[1-8](?:st|nd|rd|th)?\s*sem(?:ester)?\b",
        ]

        if any(re.search(pattern, text) for pattern in academic_patterns):
            return "Please visit this website: acad-assist.vercel.app"

        if not self.llm:
            return (
                "I am the GLUG Chatbot. If you are looking for PYQs or academic resources, "
                "please visit this website: acad-assist.vercel.app\n\n"
                "*Note: Groq Engine is not configured properly.*"
            )

        context_string = ""
        if self.db:
            try:
                matched_docs = self.db.similarity_search(message, k=4)
                context_chunks = [doc.page_content for doc in matched_docs]
                context_string = "\n\n---\n\n".join(context_chunks)
                print(f"\n🚀 [DEBUG] Retrieved Vector Context Chunks for LLM: {len(context_chunks)}")
            except Exception as e:
                logger.error(f"Error querying Supabase rows: {e}")

        system_instruction = (
            "You are the official GLUG Chatbot of NIT Durgapur. Be polite, friendly, and helpful. "
            "You answer questions about the club's activities, events, team, projects, and history "
            "using the provided context. If the context does not contain the answer, say "
            "'I don't have that information in my current records, but you can check our main site or ask a club executive.' "
            "Do not make up facts or details outside the provided context."
        )

        user_content = f"User Question: {message}"
        if context_string:
            user_content = (
                f"Use the following verified club platform context to answer the question accurately.\n"
                f"Context:\n{context_string}\n\n"
                f"User Question: {message}"
            )

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_content)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error executing prompt against Groq API: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"

# --- EXPORTS ---
llm_service = LLMService()

def get_llm_service() -> LLMService:
    return llm_service