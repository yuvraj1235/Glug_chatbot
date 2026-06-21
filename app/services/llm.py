import logging
import os
import re
from supabase.client import Client, create_client
from app.config import settings

# Updated Imports for ChatHuggingFace wrapper
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("chatbot")

class LLMService:
    def __init__(self):
        # 1. Initialize Vector Database connection
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
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
                logger.info("Supabase Vector DB successfully connected to LLMService.")
        except Exception as e:
            logger.error(f"Error connecting to Supabase Vector DB: {e}")
            self.db = None

        # 2. Fix: Wrap endpoint with ChatHuggingFace to support conversational task routing
        try:
            raw_llm = HuggingFaceEndpoint(
                repo_id=settings.HF_MODEL_REPO,
                temperature=0.4,
                max_new_tokens=512,
                huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
            )
            # This wrapper formats messages correctly as a conversational task payload
            self.llm = ChatHuggingFace(llm=raw_llm)
            logger.info(f"Hugging Face Chat client ({settings.HF_MODEL_REPO}) successfully configured.")
        except Exception as e:
            logger.error(f"Error configuring Hugging Face Endpoint: {e}")
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
                "*Note: Hugging Face Cloud client is not configured properly.*"
            )
        
     # 3. Pull text blocks directly from Supabase using Strict Server-Side Filtering
        context_string = ""
        try:
            is_profile_query = any(w in text for w in ["member", "profile", "team", "year", "who is", "alumni", "coordinator"])
            is_event_query = any(w in text for w in ["event", "audition", "talk", "upcoming", "past"])
            
            context_chunks = []
            
            if is_profile_query:
                # Force Supabase to find the profiles regardless of where they are in the 269 rows
                prof_resp = self.supabase_client.table("documents").select("content").ilike("metadata->>url", "%profile%").limit(50).execute()
                alum_resp = self.supabase_client.table("documents").select("content").ilike("metadata->>url", "%alumni%").limit(20).execute()
                
                for row in (prof_resp.data or []) + (alum_resp.data or []):
                    context_chunks.append(row['content'])
                    
            elif is_event_query:
                # Force Supabase to find the events
                event_resp = self.supabase_client.table("documents").select("content").ilike("metadata->>url", "%event%").limit(40).execute()
                for row in event_resp.data or []:
                    context_chunks.append(row['content'])
            else:
                # Standard fallback for broad questions
                general_resp = self.supabase_client.table("documents").select("content").limit(30).execute()
                for row in general_resp.data or []:
                    context_chunks.append(row['content'])
                    
            context_string = "\n\n---\n\n".join(context_chunks)
            print(f"\n🚀 [DEBUG] Extracted Context Rows for LLM: {len(context_chunks)}")
            
        except Exception as e:
            logger.error(f"Error querying Supabase rows: {e}")
        # 4. Construct Structured Prompt Messages
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
        
        # Structure payload cleanly using native Message formatting
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_content)
        ]
        
        # 5. Execute Chat Call
        try:
            # We utilize ainvoke to keep backend route processing perfectly asynchronous
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error executing prompt against Hugging Face Endpoint: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"

llm_service = LLMService()

def get_llm_service() -> LLMService:
    return llm_service