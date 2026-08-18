import logging
import re
import math
import asyncio
from supabase.client import Client, create_client
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import SupabaseVectorStore
from app.config import settings
from app.services.reranker import HFServerlessReranker, CohereReranker

logger = logging.getLogger("chatbot")

import json
import redis.asyncio as redis
GREETINGS = {"hi", "hii", "hiii", "hello", "hey", "yo", "hola", "hlo",
             "good morning", "good afternoon", "good evening", "sup"}
FAREWELLS = {"bye", "goodbye", "see you", "thanks", "thank you", "thx", "ty"}

def _small_talk_reply(normalized_text: str) -> str | None:
    if normalized_text in GREETINGS:
        return ("Hey! I'm the official chatbot of GLUG (GNU/Linux Users' Group), NIT Durgapur. "
                "Ask me about our members, events, projects, CTFs, or anything Linux/open-source related.")
    if normalized_text in FAREWELLS:
        return "You're welcome! Feel free to come back with more questions about GLUG anytime."
    return None
class LLMService:
    
    def __init__(self):
        # --- Redis Cache ---
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

            # ✅ Plain base retriever — no wrapper needed
            self.base_retriever = self.db.as_retriever(
                search_kwargs={"k": 20}
            )  # k capped at 20; overridden per-query but never above 20

            # ✅ Reranker called directly in generate_response if enabled
            if settings.USE_RERANKER:
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
            model_name = settings.DEFAULT_MODEL
            if "gemini" in model_name.lower():
                model_name = "llama-3.1-8b-instant"
                logger.warning(f"Gemini model detected. Swapping to: {model_name}")

            self.llm = ChatGroq(
                api_key=settings.GROQ_API_KEY,
                model=model_name,
                temperature=0.3,
                max_tokens=1500
            )
            logger.info(f"LLM initialized: {model_name}")

        except Exception as e:
            logger.error(f"Error configuring LLM: {e}")
            self.llm = None

    async def _get_redis_client(self):
        if not hasattr(self, 'redis_client') or self.redis_client is None:
            try:
                kwargs = {
                    "encoding": "utf-8",
                    "decode_responses": True,
                    "socket_timeout": 5.0,
                    "socket_connect_timeout": 5.0,
                    "retry_on_timeout": True,
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
        if any(k in text for k in [
            "chai", "food", "snack", "canteen", "cafe", "restaurant",
            "eat", "hungry", "maggi", "parathe", "jhoops", "chandu",
            "gate", "meal", "lunch", "dinner", "breakfast"
            ]):
            return {"source": "csv"}
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

    async def generate_response(self, message: str):
        print("LLM:", self.llm)
        print("DB:",self.db)
        if not self.llm or not self.db:
            yield f"data: {json.dumps({'response': 'System is currently unavailable or disconnected.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        text = message.lower().strip()
        # Normalize text for smarter caching (removes punctuation and extra spaces)
        import string
        normalized_text = text.translate(str.maketrans('', '', string.punctuation))
        normalized_text = ' '.join(normalized_text.split())
    
         # --- Small talk short-circuit: skip retrieval + LLM entirely ---
        canned = _small_talk_reply(normalized_text)
        if canned:
            yield f"data: {json.dumps({'response': canned})}\n\n"
            yield "data: [DONE]\n\n"
            return
        # --- LLM Response Caching ---
        redis_client = await self._get_redis_client()
        cache_key = f"llm_response:{normalized_text}"
        try:
            if redis_client:
                cached_response = await redis_client.get(cache_key)
                if cached_response:
                    logger.info("Returning cached LLM response.")
                    yield f"data: {json.dumps({'response': cached_response})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
        except Exception as e:
            logger.error(f"Redis response cache read error: {e}")

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
                # Run year-match and vector retrieval concurrently
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
                # ✅ Rerank directly — using async to avoid blocking the event loop
                if self.compressor and retrieved:
                    reranked = await self.compressor.acompress_documents(retrieved, message)
                    reranked_docs = [doc.page_content for doc in reranked]
                    logger.debug(f"[Reranker] Returned {len(reranked_docs)} docs")
                else:
                    reranked_docs = [doc.page_content for doc in retrieved[:20]]

            except Exception as e:
                # Fallback to plain vector search
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
                max_total_chars = 8000  # Reduced to lower LLM input tokens and latency

                for chunk in final_chunks:
                    # Estimate the length this chunk will add (including dividers)
                    added_length = len(chunk) + 5
                    
                    if current_length + added_length <= max_total_chars:
                        processed_chunks.append(chunk)
                        current_length += added_length
                    else:
                        # Stop adding once the budget is full to keep remaining chunks intact
                        logger.warning(f"[Context] Context limit reached. Omitted {len(final_chunks) - len(processed_chunks)} chunks.")
                        break
                        
                context_string = "\n\n---\n\n".join(processed_chunks)
                logger.debug(f"[Context] Full content:\n{context_string}")
            else:
                context_string = ""
            
            logger.debug(f"[Context] {len(final_chunks)} chunks, {len(context_string)} chars")

        except Exception as e:
            logger.error(f"[Retrieval] Error: {e}")

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
    "Do not make up, infer, or assume information that is not present in the context. "
    f"To help you understand the context, here is a README describing the available data sources:\n\n{readme_str}\n\n"

    "CRITICAL — Grounding rules (violating these is a serious failure):\n"
    "- Never state a price, time, phone number, address, or item name unless it appears verbatim in the context.\n"
    "- Never invent proper nouns. Do not create street names, road names, or addresses by combining a "
    "person's name, vendor name, or any other word with generic terms like 'Road', 'Street', or 'Lane'. "
    "Only use a place or location name if it appears verbatim in the context.\n"
    "- If the context contains bracketed placeholder text like '[price range]' or '[timings]', treat that "
    "field as NOT AVAILABLE — do not invent a plausible-sounding value to replace it.\n"
    "- When building a table or list with multiple rows/entries, each row's data must come only from the "
    "single document describing that specific place or item. Never copy or blend a field (like Location) "
    "from one entry's document into a different entry's row, even if they seem related.\n\n"

    "Response format — match the question, don't default to maximal formatting:\n"
    "- For simple, casual, or single-fact questions (e.g. 'where's good chai', 'who leads X'), answer in "
    "2-4 plain conversational sentences. No headings, no tables, no bullet lists.\n"
    "- Reserve Markdown headings/tables/bullets for genuinely structured requests: lists of events, team "
    "rosters, multi-item comparisons, or when the user explicitly asks for a breakdown or list.\n"
    "- When a table IS appropriate, only include a row if you have directly attributable source content "
    "for every cell in that row — do not pad a row with guessed or borrowed values.\n\n"

    "IMPORTANT: Never output raw JSON, Python dictionaries, database records, or plain text dumps.\n\n"

    "When using structured formatting, follow these rules:\n"
    "1. Begin with a short introductory summary (1-3 sentences) that directly answers the user's question.\n"
    "2. Organize the response using Markdown headings (##, ###, or ####) only if the answer has multiple distinct sections.\n"
    "3. Use bullet points or numbered lists whenever appropriate.\n"
    "4. Highlight important information such as names, dates, roles, locations, technologies, and keywords using bold text.\n"
    "5. If URLs are available in the context, display them as Markdown links: [text](url).\n"
    "6. If image URLs are available, include them using Markdown image syntax: ![description](image_url).\n"
    "7. If the answer naturally contains multiple categories (e.g., Upcoming Events and Past Events), create a separate section for each.\n"
    "8. Remove duplicate entries before presenting the response.\n"
    "9. If a field such as time, venue, or link is missing in the source AND is not a placeholder, display '—' instead of leaving it blank or guessing.\n"
    "10. Keep the response visually clean with proper spacing between sections, lists, and tables.\n"
    "11. End with a short concluding sentence only if it adds value.\n\n"

    "Additional Instructions:\n"
    "- Do not include implementation details, IDs, metadata, embeddings, filenames, or internal database information.\n"
    "- Do not mention that the answer was generated from context or retrieved documents.\n"
    "- If multiple context sources describe the SAME place/entity, merge them into a single coherent answer.\n"
    "- If the context contains conflicting information, prefer the most complete and recent entry.\n"
    "- If the requested information is not present in the provided context, reply exactly:\n"
    "  I don't have that information right now."
)
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(
                content=f"Context:\n{context_string}\n\nUser Question: {message}"
            )
        ]

        try:
            # Stream the response chunks in the background
            full_response = ""
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'response': chunk.content})}\n\n"
            
            # --- Save to LLM Response Cache ---
            try:
                redis_client = await self._get_redis_client()
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
        # Redis clear cache is now manual if needed since we cache LLM responses instead of the whole DB
        redis_client = await self._get_redis_client()
        if redis_client:
            await redis_client.flushdb()



llm_service = LLMService()

def get_llm_service() -> LLMService:
    return llm_service