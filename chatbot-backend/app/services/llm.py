import logging
import os
import re
from google import genai
from google.genai import types
from app.config import settings

# RAG Imports
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("chatbot")

class LLMService:
    def __init__(self):
        self.is_configured = False
        self.client = None
        api_key = settings.GEMINI_API_KEY
        
        # 1. Initialize Gemini Client
        if api_key and api_key != "your_gemini_api_key_here":
            try:
                self.client = genai.Client(api_key=api_key)
                self.is_configured = True
                logger.info("Google GenAI client successfully configured.")
            except Exception as e:
                logger.error(f"Error configuring Google GenAI client: {e}")
        else:
            logger.warning("Gemini API key is not configured. Running in unconfigured mode.")

        # 2. Initialize Vector Database connection
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.chroma_path = os.path.join(os.path.dirname(__file__), "../../chroma_data")
            self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embeddings)
            logger.info("Chroma Vector DB successfully connected to LLMService.")
        except Exception as e:
            logger.error(f"Error connecting to Chroma Vector DB: {e}")
            self.db = None

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

        if not self.is_configured or not self.client:
            return (
                "I am the GLUG Chatbot. If you are looking for PYQs or academic resources, "
                "please visit this website: acad-assist.vercel.app\n\n"
                "*Note: To enable general AI conversation, please configure your GEMINI_API_KEY in the .env file.*"
            )
        
        # 3. Retrieve Context from Vector Database
        context_string = ""
        if self.db:
            try:
                # Search for top 4 most relevant chunks from your scraped data
                docs = self.db.similarity_search(message, k=4)
                if docs:
                    context_chunks = []
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        context_chunks.append(f"[Source: {source}]\n{doc.page_content}")
                    context_string = "\n\n---\n\n".join(context_chunks)
            except Exception as e:
                logger.error(f"Error searching Chroma DB: {e}")

        # 4. Construct the prompt with context injected
        full_prompt = f"User Question: {message}\n\n"
        if context_string:
            full_prompt = (
                f"Use the following verified club platform context to answer the question accurately.\n"
                f"Context:\n{context_string}\n\n"
                f"User Question: {message}"
            )
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=(
                    "You are the official GLUG Chatbot of NIT Durgapur. Be polite, friendly, and helpful. "
                    "You answer questions about the club's activities, events, team, projects, and history "
                    "using the provided context. If the context does not contain the answer, say "
                    "'I don't have that information in my current records, but you can check our main site or ask a club executive.' "
                    "Do not make up facts or details outside the provided context."
                ),
                temperature=0.4, # Slightly lowered to keep responses highly accurate to context
            )
            
            response = await self.client.aio.models.generate_content(
                model=settings.DEFAULT_MODEL,
                contents=full_prompt,
                config=config
            )
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"

llm_service = LLMService()

def get_llm_service() -> LLMService:
    return llm_service