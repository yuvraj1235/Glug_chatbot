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
        # --- In-Memory Cooldown Tracker (Independent of Redis) ---
        self._cooldowns: dict[str, float] = {}
        self.redis_client = None

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
            
            # Use Native Supabase Vector Store
            self.db = SupabaseVectorStore(
                embedding=self.embeddings,
                client=self.supabase_client,
                table_name="documents",
                query_name="match_documents"
            )

            # Plain base retriever
            self.base_retriever = self.db.as_retriever(
                search_kwargs={"k": 20}
            )

            # Reranker
            if settings.USE_RERANKER:
                from app.services.reranker import HFServerlessReranker, CohereReranker
                if settings.COHERE_API_KEY:
                    logger.info("Initializing CohereReranker...")
                    self.compressor = CohereReranker(
                        api_token=settings.COHERE_API_KEY,
                        model_name="rerank-v3.5",
                        top_n=80
                    )
                else:
                    logger.info("Initializing HFServerlessReranker...")
                    self.compressor = HFServerlessReranker(
                        api_token=settings.HUGGINGFACEHUB_API_TOKEN,
                        model_name="BAAI/bge-reranker-base",
                        top_n=80
                    )
                logger.info("DB, Retriever and Reranker initialized successfully")
            else:
                logger.info("Reranker is disabled by configuration.")
                self.compressor = None
                logger.info("DB and Retriever initialized successfully")

        except Exception as e:
            logger.error(f"Error connecting to DB/Retriever: {e}")
            self.db = None
            self.base_retriever = None
            self.compressor = None

        # --- LLM ---
        try:
            # Initialize configured LLM via AWS Bedrock
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
                    "encoding": "utf-8",
                    "decode_responses": True,
                    "socket_timeout": 3.0,
                    "socket_connect_timeout": 3.0,
                    "retry_on_timeout": False,
                    "health_check_interval": 30
                }
                if settings.REDIS_URL.startswith("rediss://"):
                    kwargs["ssl_cert_reqs"] = "none"
                self.redis_client = redis.from_url(settings.REDIS_URL, **kwargs)
                logger.info("Redis client initialized.")
            except Exception as e:
                logger.error(f"Error initializing Redis client: {e}")
                self.redis_client = None
        return self.redis_client

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
        limit: int = 10
    ) -> list[str]:
        try:
            broad_docs = await self.db.asimilarity_search(
                message,
                k=20,
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

    async def generate_response(self, message: str, session_id: str = "default_session"):
        print("LLM:", self.llm)
        print("DB:", self.db)
        if not self.llm or not self.db:
            yield f"data: {json.dumps({'response': 'System is currently unavailable or disconnected.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        now = time.time()

        # --- In-Memory 50s Cooldown Guard (No Redis dependency) ---
        if session_id in self._cooldowns:
            expiration = self._cooldowns[session_id]
            if now < expiration:
                ttl = int(math.ceil(expiration - now))
                logger.warning(f"[Guard] Cooldown active for {session_id}. {ttl}s remaining.")
                error_msg = f"Please wait {ttl} seconds before sending another message."
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return

        # Set in-memory cooldown timestamp
        self._cooldowns[session_id] = now + settings.PROMPT_COOLDOWN_SECONDS

        # Periodically purge expired cooldowns to prevent memory growth
        if len(self._cooldowns) > 1000:
            self._cooldowns = {k: v for k, v in self._cooldowns.items() if v > now}

        # --- Prompt length guard ---
        if len(message) > self.MAX_PROMPT_LENGTH:
            logger.warning(
                f"[Guard] Prompt rejected — length {len(message)} exceeds "
                f"limit of {self.MAX_PROMPT_LENGTH}."
            )
            # Release lock so user isn't penalized for an invalid message
            self._cooldowns.pop(session_id, None)
            error_msg = (
                f"Your message is too long ({len(message)} characters). "
                f"Please keep it under {self.MAX_PROMPT_LENGTH} characters."
            )
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
            return

        text = message.lower().strip()
        # Normalize text for smarter caching (removes punctuation and extra spaces)
        import string
        normalized_text = text.translate(str.maketrans('', '', string.punctuation))
        normalized_text = ' '.join(normalized_text.split())

        # --- Redis LLM Response Caching ---
        redis_client = await self._get_redis_client()
        cache_key = f"llm_response:{normalized_text}"
        try:
            if redis_client:
                cached_response = await redis_client.get(cache_key)
                if cached_response:
                    logger.info("Returning cached LLM response from Redis.")
                    yield f"data: {json.dumps({'response': cached_response})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
        except Exception as e:
            logger.error(f"Redis response cache read error: {e}")

        # Step 6a — Short-circuit: Greetings & Single Words
        # Bypasses the LLM completely for common greetings to save tokens and guarantee the fallback message.
        if normalized_text in ["hi", "hello", "hey", "hola", "sup", "greetings", "ping", "test", "who are you"]:
            logger.info("[Guard] Greeting detected — returning fallback message without LLM call.")
            yield f"data: {json.dumps({'response': FALLBACK_MESSAGE})}\n\n"
            
            # --- Save to Redis Cache immediately ---
            try:
                if redis_client:
                    await redis_client.setex(cache_key, 86400, FALLBACK_MESSAGE)
            except Exception:
                pass

            yield "data: [DONE]\n\n"
            return

        context_string = ""

        try:
            # Step 1 — Route to correct source
            search_filter = self._get_search_filter(text)
            logger.debug(f"[Routing] filter={search_filter}")

            # Step 2 — Year exact match boost + vector retrieval (parallelized)
            years_in_prompt = re.findall(r'\b20\d{2}\b', text)

            # Update filter per query before parallel dispatch
            self.base_retriever.search_kwargs = {
                "k": 20,
                "filter": search_filter if search_filter else None
            }

            if years_in_prompt:
                year_task = self._get_year_exact_matches(
                    message, years_in_prompt, search_filter
                )
                retrieval_task = self.base_retriever.ainvoke(message)
                exact_match_docs, retrieved = await asyncio.gather(
                    year_task, retrieval_task
                )
                logger.debug(f"[YearMatch] Found {len(exact_match_docs)} matches")
            else:
                exact_match_docs = []
                retrieved = await self.base_retriever.ainvoke(message)

            logger.debug(f"[Retriever] Got {len(retrieved)} docs")

            # Step 3 — Rerank
            reranked_docs = []
            try:
                if self.compressor and retrieved:
                    reranked = await self.compressor.acompress_documents(retrieved, message)
                    reranked_docs = [doc.page_content for doc in reranked]
                    logger.debug(f"[Reranker] Returned {len(reranked_docs)} docs")
                else:
                    reranked_docs = [doc.page_content for doc in retrieved[:20]]

            except Exception as e:
                logger.warning(f"[Retrieval] Failed, using fallback: {e}")
                fallback = await self.db.asimilarity_search(
                    message,
                    k=20,
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

            # Step 5 — Build context by stacking whole chunks up to the character limit
            if final_chunks:
                processed_chunks = []
                current_length = 0
                max_total_chars = 8000

                for chunk in final_chunks:
                    added_length = len(chunk) + 5
                    
                    if current_length + added_length <= max_total_chars:
                        processed_chunks.append(chunk)
                        current_length += added_length
                    else:
                        logger.warning(f"[Context] Context limit reached. Omitted {len(final_chunks) - len(processed_chunks)} chunks.")
                        break
                        
                context_string = "\n\n---\n\n".join(processed_chunks)
            else:
                context_string = ""
            
            logger.debug(f"[Context] {len(final_chunks)} chunks, {len(context_string)} chars")

        except Exception as e:
            logger.error(f"[Retrieval] Error: {e}")

        # Step 6b — Short-circuit: if no context was retrieved, out of scope
        if not context_string.strip():
            logger.info("[Guard] Empty context — returning fallback message without LLM call.")
            yield f"data: {json.dumps({'response': FALLBACK_MESSAGE})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Step 6 — LLM
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
            "Do not make up, infer, or assume information that is not present in the context.\n\n"
            
            "================================================\n"
            "CRITICAL GUARDRAILS AND FALLBACK RULES:\n"
            "If ANY of the following conditions are met, you MUST reply with EXACTLY the FALLBACK MESSAGE below and absolutely NOTHING else (do not introduce members, do not summarize context):\n"
            "1. The user's input is a greeting (e.g., 'hi', 'hello', 'hey', 'good morning').\n"
            "2. The user's input is general small talk, casual chat, or outside the scope of GLUG.\n"
            "3. The requested GLUG-related information is NOT found in the provided context.\n\n"
            f"FALLBACK MESSAGE to output:\n{FALLBACK_MESSAGE}\n"
            "================================================\n\n"
            
            f"To help you understand the context, here is a README describing the available data sources:\n\n{readme_str}\n\n"

            "Formatting Rules (Apply ONLY if answering a valid GLUG question):\n"
            "1. Begin with a short introductory summary (1-3 sentences) that directly answers the user's question.\n"
            "2. Organize the response using Markdown headings (##, ###, or ####).\n"
            "3. Use bullet points or numbered lists whenever appropriate.\n"
            "4. Highlight important information such as names, dates, roles, locations, technologies, and keywords using bold text.\n"
            "5. Whenever presenting structured information (events, members, projects, schedules, repositories, achievements, etc.), use Markdown tables.\n"
            "6. If URLs are available in the context, display them as Markdown links: [text](url).\n"
            "7. If image URLs are available, include them using Markdown image syntax: ![description](image_url).\n"
            "8. If the answer naturally contains multiple categories (e.g., Upcoming Events and Past Events), create a separate section for each.\n"
            "9. Remove duplicate entries before presenting the response.\n"
            "10. If a field such as time, venue, or link is missing, display '—' instead of leaving it blank.\n"
            "11. Keep the response visually clean with proper spacing between sections, lists, and tables.\n"
            "12. End with a short concluding sentence only if it adds value.\n\n"

            "Additional Instructions:\n"
            "- Do not include implementation details, IDs, metadata, embeddings, filenames, or internal database information.\n"
            "- Do not mention that the answer was generated from context or retrieved documents.\n"
            "- If multiple context sources contain the same information, merge them into a single coherent answer.\n"
            "- If the context contains conflicting information, prefer the most complete and recent entry."
        )
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(
                content=f"Context:\n{context_string}\n\nUser Question: {message}"
            )
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
            
            # --- Save to Redis Cache ---
            try:
                if redis_client:
                    await redis_client.setex(cache_key, 86400, full_response)
            except Exception as e:
                logger.error(f"Redis response cache write error: {e}")
                
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