import logging
import re
import math
import time
import asyncio
import json
import redis.asyncio as redis
from supabase.client import Client, create_client
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import SupabaseVectorStore
from app.config import settings

logger = logging.getLogger("chatbot")

FALLBACK_MESSAGE = (
    "I'm GLUG's official assistant and can only answer questions about "
    "GLUG (GNU/Linux Users' Group) — its members, alumni, events, projects, "
    "CTF challenges, and TechBytes articles. "
    "Your question seems to be outside that scope. "
    "Feel free to ask me anything about GLUG! 🐧"
)

class LLMService:
    MAX_PROMPT_LENGTH = 200

    def __init__(self):
        self._cooldowns: dict[str, float] = {}
        self.redis_client = None

        try:
            self.embeddings = HuggingFaceEndpointEmbeddings(
                model="BAAI/bge-base-en-v1.5",
                huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
            )
            self.supabase_client: Client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_KEY
            )
            self.db = SupabaseVectorStore(
                embedding=self.embeddings,
                client=self.supabase_client,
                table_name="documents",
                query_name="match_documents"
            )
            self.base_retriever = self.db.as_retriever(search_kwargs={"k": 20})

            if settings.USE_RERANKER:
                from app.services.reranker import HFServerlessReranker, CohereReranker
                if settings.COHERE_API_KEY:
                    self.compressor = CohereReranker(
                        api_token=settings.COHERE_API_KEY,
                        model_name="rerank-v3.5",
                        top_n=80
                    )
                else:
                    self.compressor = HFServerlessReranker(
                        api_token=settings.HUGGINGFACEHUB_API_TOKEN,
                        model_name="BAAI/bge-reranker-base",
                        top_n=80
                    )
            else:
                self.compressor = None
        except Exception as e:
            logger.error(f"Error connecting to DB/Retriever: {e}")
            self.db = None
            self.base_retriever = None
            self.compressor = None

        try:
            self.llm = ChatBedrockConverse(
                model=settings.DEFAULT_MODEL,
                temperature=0.3,
                max_tokens=1500,
                region_name=settings.AWS_REGION_NAME,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            logger.info(f"LLM initialized: {settings.DEFAULT_MODEL} (AWS Bedrock)")
        except Exception as e:
            logger.error(f"Error configuring LLM: {e}")
            self.llm = None

    async def _get_redis_client(self):
        if not hasattr(self, 'redis_client') or self.redis_client is None:
            try:
                kwargs = {
                    "encoding": "utf-8", "decode_responses": True, "socket_timeout": 3.0,
                    "socket_connect_timeout": 3.0, "retry_on_timeout": False, "health_check_interval": 30
                }
                if settings.REDIS_URL.startswith("rediss://"):
                    kwargs["ssl_cert_reqs"] = "none"
                self.redis_client = redis.from_url(settings.REDIS_URL, **kwargs)
            except Exception as e:
                logger.error(f"Error initializing Redis client: {e}")
                self.redis_client = None
        return self.redis_client

    def _get_search_filter(self, text: str) -> dict:
        if any(k in text for k in ["alumni", "graduated", "passed out", "passout", "senior", "batch", "graduate"]):
            return {"source": "alumni"}
        if any(k in text for k in ["event", "workshop", "hackathon", "competition", "seminar", "fest"]):
            return {"source": "events"}
        if any(k in text for k in ["member", "team", "profile", "core team", "coordinator"]):
            return {"source": "profiles"}
        if any(k in text for k in ["ctf", "capture the flag", "cybersecurity", "hacking"]):
            return {"source": "ctf"}
        if any(k in text for k in ["project", "open source", "repository", "github"]):
            return {"source": "project"}
        if any(k in text for k in ["blog", "article", "techbytes", "post", "write-up"]):
            return {"source": "techbytes"}
        return {}

    async def _get_year_exact_matches(self, message: str, years: list[str], search_filter: dict, limit: int = 10) -> list[str]:
        try:
            broad_docs = await self.db.asimilarity_search(message, k=20, filter=search_filter if search_filter else None)
            matched = []
            for doc in broad_docs:
                if any(year in doc.page_content for year in years):
                    matched.append(f"[VERIFIED PROFILE]\n{doc.page_content}")
                    if len(matched) >= limit: break
            return matched
        except Exception as e:
            return []

    async def generate_response(self, message: str, session_id: str = "default_session"):
        if not self.llm or not self.db:
            yield f"data: {json.dumps({'response': 'System is currently unavailable.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        now = time.time()
        if session_id in self._cooldowns:
            expiration = self._cooldowns[session_id]
            if now < expiration:
                ttl = int(math.ceil(expiration - now))
                yield f"data: {json.dumps({'error': f'Please wait {ttl} seconds.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

        self._cooldowns[session_id] = now + settings.PROMPT_COOLDOWN_SECONDS
        if len(self._cooldowns) > 1000:
            self._cooldowns = {k: v for k, v in self._cooldowns.items() if v > now}

        if len(message) > self.MAX_PROMPT_LENGTH:
            self._cooldowns.pop(session_id, None)
            yield f"data: {json.dumps({'error': f'Your message is too long.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        text = message.lower().strip()
        import string
        normalized_text = text.translate(str.maketrans('', '', string.punctuation))
        normalized_text = ' '.join(normalized_text.split())
        
        redis_client = await self._get_redis_client()
        cache_key = f"llm_response:{normalized_text}"
        try:
            if redis_client:
                cached = await redis_client.get(cache_key)
                if cached:
                    yield f"data: {json.dumps({'response': cached})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
        except Exception:
            pass

        context_string = ""
        try:
            search_filter = self._get_search_filter(text)
            years_in_prompt = re.findall(r'\b20\d{2}\b', text)
            self.base_retriever.search_kwargs = {"k": 20, "filter": search_filter if search_filter else None}

            if years_in_prompt:
                year_task = self._get_year_exact_matches(message, years_in_prompt, search_filter)
                retrieval_task = self.base_retriever.ainvoke(message)
                exact_match_docs, retrieved = await asyncio.gather(year_task, retrieval_task)
            else:
                exact_match_docs = []
                retrieved = await self.base_retriever.ainvoke(message)

            reranked_docs = []
            try:
                if self.compressor and retrieved:
                    reranked = await self.compressor.acompress_documents(retrieved, message)
                    reranked_docs = [doc.page_content for doc in reranked]
                else:
                    reranked_docs = [doc.page_content for doc in retrieved[:20]]
            except Exception as e:
                fallback = await self.db.asimilarity_search(message, k=20, filter=search_filter if search_filter else None)
                reranked_docs = [doc.page_content for doc in fallback]

            seen = set()
            final_chunks = []
            for chunk in exact_match_docs + reranked_docs:
                if chunk not in seen:
                    seen.add(chunk)
                    final_chunks.append(chunk)

            if final_chunks:
                processed_chunks = []
                current_length = 0
                for chunk in final_chunks:
                    added_length = len(chunk) + 5
                    if current_length + added_length <= 8000:
                        processed_chunks.append(chunk)
                        current_length += added_length
                    else:
                        break
                context_string = "\n\n---\n\n".join(processed_chunks)
        except Exception as e:
            logger.error(f"[Retrieval] Error: {e}")

        if not context_string.strip():
            yield f"data: {json.dumps({'response': FALLBACK_MESSAGE})}\n\n"
            yield "data: [DONE]\n\n"
            return

        TABLE_DESCRIPTIONS = {
            "profiles": "Profiles of current students and core members of the club.",
            "alumni": "Profiles of graduated students (alumni).",
            "events": "Details about events, hackathons, and workshops conducted by the club.",
            "project": "Open-source projects or GitHub repositories created by the club.",
            "ctf": "Cybersecurity challenges, CTFs (Capture The Flag), or hacking events.",
            "techbytes": "Technical blogs, write-ups, or articles published by the club."
        }
        readme_str = "\\n".join([f"- {k}: {v}" for k, v in TABLE_DESCRIPTIONS.items()])
        
        system_instruction = (
            "You are the official chatbot of GLUG (GNU/Linux Users' Group), the official Open Source and Linux community of NIT Durgapur. "
            "Always refer to the organization as 'GLUG' or 'GLUG (GNU/Linux Users' Group)'. "
            "Never refer to the club simply as 'NIT Durgapur'. "
            "Answer the user's question using ONLY the provided context. "
            "Do not make up, infer, or assume information that is not present in the context. "
            f"To help you understand the context, here is a README describing the available data sources:\n\n{readme_str}\n\n"
            "IMPORTANT: You MUST respond using beautifully formatted Markdown. "
            "Never output raw JSON, Python dictionaries, database records, or plain text dumps.\n\n"
            "Formatting Rules:\n"
            "1. Begin with a short introductory summary (1-3 sentences) that directly answers the user's question.\n"
            "2. Organize the response using Markdown headings (##, ###, or ####).\n"
            "3. Use bullet points or numbered lists whenever appropriate.\n"
            "4. Highlight important information such as names, dates, roles, locations, technologies, and keywords using bold text.\n"
            "5. Whenever presenting structured information (events, members, projects, schedules, repositories, achievements, etc.), use Markdown tables.\n"
            "6. If URLs are available in the context, display them as Markdown links: [text](url).\n"
            "7. If image URLs are available, include them using Markdown image syntax: ![description](image_url).\n"
            "8. If the answer naturally contains multiple categories, create a separate section for each.\n"
            "9. Remove duplicate entries before presenting the response.\n"
            "10. If a field is missing, display '—' instead of leaving it blank.\n"
            "11. Keep the response visually clean with proper spacing.\n"
            "12. End with a short concluding sentence only if it adds value.\n\n"
            "Additional Instructions:\n"
            "- Do not include implementation details, IDs, metadata, embeddings, filenames, or internal database information.\n"
            "- Do not mention that the answer was generated from context.\n"
            "- If multiple context sources contain the same information, merge them.\n"
            "- If the context contains conflicting information, prefer the most complete and recent entry.\n"
            "- STRICT OUT-OF-SCOPE RULE: If the user's question is about anything unrelated to GLUG "
            "you MUST reply with EXACTLY the following message and nothing else:\n"
            f"  {FALLBACK_MESSAGE}\n"
            "- Do NOT attempt to answer out-of-scope questions even partially. Do NOT apologize or explain.\n"
            "- If the requested GLUG-related information is not present in the provided context, reply with EXACTLY:\n"
            f"  {FALLBACK_MESSAGE}"
        )
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Context:\n{context_string}\n\nUser Question: {message}")
        ]

        try:
            full_response = ""
            async for chunk in self.llm.astream(messages):
                # Extract text safely whether Bedrock returns a string or a list of blocks
                chunk_data = chunk.content
                if isinstance(chunk_data, list):
                    text_chunk = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in chunk_data
                    )
                else:
                    text_chunk = str(chunk_data)

                if text_chunk:
                    full_response += text_chunk
                    yield f"data: {json.dumps({'response': text_chunk})}\n\n"
            
            try:
                if redis_client:
                    await redis_client.setex(cache_key, 86400, full_response)
            except Exception:
                pass
                
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[LLM] Invocation error: {e}")
            yield f"data: {json.dumps({'response': f'Error connecting to LLM: {e}'})}\n\n"
            yield "data: [DONE]\n\n"

    async def refresh_cache(self):
        redis_client = await self._get_redis_client()
        if redis_client:
            await redis_client.flushdb()

llm_service = LLMService()

def get_llm_service() -> LLMService:
    return llm_service