# app/services/enrichment.py
import httpx
import hashlib
import json
import os
import asyncio
import time
import logging

logger = logging.getLogger("chatbot")

CACHE_FILE = "hf_enrichment_cache.json"

# ─────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[Enrichment Cache] Failed to load cache: {e}")
    return {}

def save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"[Enrichment Cache] Failed to save cache: {e}")

# ─────────────────────────────────────────
# Rate Limiting & Pacing for Groq llama-3.1-8b-instant
# RPM limit: 30, TPM limit: 6000
# ─────────────────────────────────────────

_request_history = []
_history_lock = asyncio.Lock()

def estimate_tokens(text: str) -> int:
    # Character count based estimation with safety margin
    # System prompt is ~150 tokens, plus 300 max_tokens for prose output
    input_tokens = int(len(text[:2500]) / 3.5) + 150
    output_tokens = 300
    return input_tokens + output_tokens

async def pace_rate_limit(estimated_tokens: int):
    estimated_tokens = min(estimated_tokens, 5000)
    
    while True:
        async with _history_lock:
            now = time.time()
            _request_history[:] = [entry for entry in _request_history if now - entry[0] < 60.0]
            
            current_requests = len(_request_history)
            current_tokens = sum(entry[1] for entry in _request_history)
            
            if current_requests < 30 and (current_tokens + estimated_tokens) <= 5500:
                _request_history.append((now, estimated_tokens))
                return
                
            if not _request_history:
                return
                
            oldest_time = _request_history[0][0]
            sleep_needed = 60.0 - (now - oldest_time) + 0.1
            
        if sleep_needed > 0:
            logger.info(
                f"[Enrichment Rate Limit] Approaching limits "
                f"(Requests: {current_requests}/30, Tokens: {current_tokens}/{5500}, next request needs {estimated_tokens}). "
                f"Pacing sleep for {sleep_needed:.2f}s..."
            )
            await asyncio.sleep(sleep_needed)

# ─────────────────────────────────────────
# Core Prose Transformer
# ─────────────────────────────────────────

async def enrich_with_hf_summary(
    source: str,
    raw_text: str
) -> str:
    """
    Transforms raw API dumps into clean, uniform natural language paragraphs (prose)
    using Groq Llama-3 to eliminate technical syntax boilerplate, optimizing embeddings 
    and conserving runtime context window space.
    """
    try:
        from app.config import settings
        groq_key = settings.GROQ_API_KEY
    except Exception:
        groq_key = os.environ.get("GROQ_API_KEY")

    if not groq_key:
        logger.warning(f"[Enrichment] Groq API key not found. Skipping transformation for source={source}")
        return f"Source Section: {source}\n\n{raw_text}"

    # Construct source-specific descriptive instruction contexts
    if source == "profiles":
        style_instruction = (
            "Convert the raw profile keys and fields into a highly cohesive biographical paragraph. "
            "Write in the third-person active voice. Highlight their full name, role (if visible), graduation or batch year, degree, and specific social profile handles (GitHub, LinkedIn, Facebook). "
            "Example: 'GLUG Member Profile: Debmalya Das is a BTECH student with username Debmalya_007. Their contact email is dasdebmalya03@gmail.com. They are active on GitHub at github.com/DebmalyaDas-007 and LinkedIn.'"
        )
    elif source == "events":
        style_instruction = (
            "Convert the raw event fields into an engaging, descriptive paragraph summarizing the technical event. "
            "Incorporate the title, key description facts, core technical domains (like cybersecurity or webdev), exact timelines/dates, prize tracks, and registration URLs. "
            "Example: 'GLUG Event: Mini-CTF is a beginner-friendly Capture the Flag competition focusing on cybersecurity and ethical hacking. Held from March 1st to March 2nd, 2023, it featured a prize pool of 1700+ INR and goodies. Official resources can be accessed at minictf.nitdgplug.org.'"
        )
    else:
        style_instruction = (
            "Convert the raw data data fields into clear, natural language prose. Eliminate brackets, trailing braces, structural punctuation, and database table metadata keys."
        )

    system_prompt = (
        "You are an expert data-transformation engineer. "
        f"Your task is to rewrite raw technical text into clean, high-density natural language prose.\n"
        f"CRITICAL RULES:\n"
        f"1. Follow this style: {style_instruction}\n"
        f"2. Output ONLY the resulting paragraph. No conversational preambles, introductory lines, or markdown annotations.\n"
        f"3. Retain every unique personal handle, URL link, date, name, and metric asset precisely. Do not drop links or specific names."
    )

    try:
        async with httpx.AsyncClient() as client:
            for attempt in range(3):
                tokens = estimate_tokens(raw_text)
                await pace_rate_limit(tokens)

                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Raw text data payload to transform:\n{raw_text[:2500]}"}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 300
                    },
                    timeout=20.0
                )

                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    try:
                        sleep_time = float(retry_after) if retry_after else (5.0 * (attempt + 1))
                    except ValueError:
                        sleep_time = 5.0 * (attempt + 1)
                    logger.warning(f"[Enrichment] Groq 429. Retrying in {sleep_time}s...")
                    
                    async with _history_lock:
                        _request_history.append((time.time(), 1000))
                        
                    await asyncio.sleep(sleep_time)
                    continue

                if response.status_code != 200:
                    logger.error(f"[Enrichment] Groq failed ({response.status_code}) for source={source}")
                    return f"Source Section: {source}\n\n{raw_text}"

                result = response.json()
                transformed_prose = result["choices"][0]["message"]["content"].strip()
                
                # Prepend precise tag headers for clear identification by downstream query routers
                return f"Source Section: {source}\n\n{transformed_prose}"

            logger.error(f"[Enrichment] Rate limits completely exhausted. Skipping source={source}")
            return f"Source Section: {source}\n\n{raw_text}"

    except Exception as e:
        logger.error(f"[Enrichment] Exception for source={source}: {e}")
        return f"Source Section: {source}\n\n{raw_text}"

# ─────────────────────────────────────────
# Cached wrapper
# ─────────────────────────────────────────

async def enrich_cached(
    source: str,
    raw_text: str,
    cache: dict
) -> str:
    key = hashlib.md5(raw_text.encode()).hexdigest()

    if key in cache:
        return cache[key]

    enriched = await enrich_with_hf_summary(source, raw_text)
    cache[key] = enriched
    return enriched