import logging
import os
import re
import httpx
from typing import Sequence
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import BaseDocumentCompressor, Document
from app.config import settings

logger = logging.getLogger("chatbot")

# --- CUSTOM FREE HUGGING FACE RERANKER ---
class HFServerlessReranker(BaseDocumentCompressor):
    api_token: str
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 4

    def compress_documents(self, documents: Sequence[Document], query: str) -> Sequence[Document]:
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


# --- LLM SERVICE WITH TWO-STAGE RAG RETRIEVAL ---
class LLMService:
    def __init__(self):
        # Initialize Vector Database connection
        try:
            self.embeddings = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
            )
            
            supabase_url = settings.SUPABASE_URL
            supabase_key = settings.SUPABASE_SERVICE_KEY
            
            if not supabase_url or not supabase_key:
                logger.error("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in settings configuration.")
                self.db = None
                self.retriever = None
            else:
                self.supabase_client: Client = create_client(supabase_url, supabase_key)
                self.db = SupabaseVectorStore(
                    client=self.supabase_client,
                    embedding=self.embeddings,
                    table_name="documents",
                    query_name="match_documents"
                )
                
                # Stage 1: Broad search to extract top 20 candidate text blocks
                base_retriever = self.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 20}
                )
                
                # Stage 2: Custom Serverless Cross-Encoder pipeline
                compressor = HFServerlessReranker(
                    api_token=settings.HUGGINGFACEHUB_API_TOKEN,
                    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    top_n=4  # Compress the broad net down to the 4 best records
                )
                
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, 
                    base_retriever=base_retriever
                )
                
                logger.info("Supabase Vector DB and HF Reranker pipeline successfully initialized.")
        except Exception as e:
            logger.error(f"Error connecting to Supabase Vector DB / Reranker: {e}")
            self.db = None
            self.retriever = None

        # Wrap endpoint with ChatHuggingFace to support conversational task routing
        try:
            raw_llm = HuggingFaceEndpoint(
                repo_id=settings.HF_MODEL_REPO,
                temperature=0.4,
                max_new_tokens=512,
                huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
            )
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
        
        context_string = ""
        if self.retriever:
            try:
                # Runs similarity search then passes candidates through the custom cloud cross-encoder wrapper
                compressed_docs = await self.retriever.ainvoke(message)
                context_chunks = [doc.page_content for doc in compressed_docs]
                context_string = "\n\n---\n\n".join(context_chunks)
                print(f"\n🚀 [DEBUG] Reranked Context Chunks for LLM: {len(context_chunks)}")
            except Exception as e:
                logger.error(f"Error executing compression pipeline retrieval: {e}")
            
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
            logger.error(f"Error executing prompt against Hugging Face Endpoint: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"

# --- EXPORTS ---
llm_service = LLMService()

def get_llm_service() -> LLMService:
    return llm_service