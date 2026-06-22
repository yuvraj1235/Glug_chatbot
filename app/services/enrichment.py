# app/services/enrichment.py
#cfbr
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
    # System prompt is ~40 tokens, plus 100 max_tokens for output
    input_tokens = int(len(text[:2000]) / 3.5) + 40
    output_tokens = 100
    return input_tokens + output_tokens

async def pace_rate_limit(estimated_tokens: int):
    # Safety cap to prevent infinite loops if estimation goes wrong
    estimated_tokens = min(estimated_tokens, 5000)
    
    while True:
        async with _history_lock:
            now = time.time()
            # Clean up entries older than 60 seconds
            _request_history[:] = [entry for entry in _request_history if now - entry[0] < 60.0]
            
            current_requests = len(_request_history)
            current_tokens = sum(entry[1] for entry in _request_history)
            
            # TPM limit target is 5500 to leave a safety buffer below 6000 TPM
            # RPM limit target is 30 requests per minute
            if current_requests < 30 and (current_tokens + estimated_tokens) <= 5500:
                # Safe to proceed! Add the entry inside the lock and exit.
                _request_history.append((now, estimated_tokens))
                return
                
            # If not safe, find out how long we must sleep.
            # We must sleep until the oldest request falls out of the window.
            if not _request_history:
                return
                
            oldest_time = _request_history[0][0]
            sleep_needed = 60.0 - (now - oldest_time) + 0.1
            
        # We release the lock before sleeping so other tasks are not blocked
        if sleep_needed > 0:
            logger.info(
                f"[Enrichment Rate Limit] Approaching limits "
                f"(Requests: {current_requests}/30, Tokens: {current_tokens}/{5500}, next request needs {estimated_tokens}). "
                f"Pacing sleep for {sleep_needed:.2f}s..."
            )
            await asyncio.sleep(sleep_needed)

# ─────────────────────────────────────────
# Core summarizer
# ─────────────────────────────────────────

async def enrich_with_hf_summary(
    source: str,
    raw_text: str
) -> str:
    """
    Calls Llama-3 via Groq API to generate a natural language summary.
    Prepends summary + source tag to raw chunk.
    Falls back to raw_text silently on any failure.
    Includes rate limit retry logic (429 handling) and pacing.
    """
    # Too short to summarize meaningfully → skip
    if len(raw_text.split("\n")) < 4:
        return f"Source Section: {source}\n\n{raw_text}"

    try:
        from app.config import settings
        groq_key = settings.GROQ_API_KEY
    except Exception:
        groq_key = os.environ.get("GROQ_API_KEY")

    if not groq_key:
        logger.warning(f"[Enrichment] Groq API key not found. Skipping summary for source={source}")
        return f"Source Section: {source}\n\n{raw_text}"

    try:
        async with httpx.AsyncClient() as client:
            for attempt in range(3):
                # Calculate estimated tokens and pace requests to avoid exceeding 6K TPM / 30 RPM
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
                            {
                                "role": "system",
                                "content": (
                                    "You are a precise summarization assistant. "
                                    "Summarize the given text in a single, short sentence. "
                                    "Do not include any conversational filler, introductory phrases (like 'Here is a summary:'), or prefix/suffix."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Text to summarize:\n{raw_text[:2000]}"
                            }
                        ],
                        "temperature": 0.0,
                        "max_tokens": 100
                    },
                    timeout=15.0
                )

                if response.status_code == 429:
                    retry_after = response.headers.get("retry-after")
                    try:
                        sleep_time = float(retry_after) if retry_after else (5.0 * (attempt + 1))
                    except ValueError:
                        sleep_time = 5.0 * (attempt + 1)
                    logger.warning(f"[Enrichment] Groq returned 429. Retrying in {sleep_time}s... (Attempt {attempt+1}/3)")
                    
                    # Add penalty to request history to force safety cooldown
                    async with _history_lock:
                        _request_history.append((time.time(), 1000))
                        
                    await asyncio.sleep(sleep_time)
                    continue

                if response.status_code != 200:
                    logger.error(f"[Enrichment] Groq failed ({response.status_code}) for source={source}")
                    return f"Source Section: {source}\n\n{raw_text}"

                result = response.json()
                summary = result["choices"][0]["message"]["content"].strip()
                # Clean any accidental conversational prefixes
                if summary.lower().startswith("here is a summary:"):
                    summary = summary[18:].strip()

                return (
                    f"Summary: {summary}\n"
                    f"Source Section: {source}\n\n"
                    f"{raw_text}"
                )

            # If all 3 attempts returned 429
            logger.error(f"[Enrichment] Groq rate limits exceeded. Skipping summary for source={source}")
            return f"Source Section: {source}\n\n{raw_text}"

    except Exception as e:
        logger.error(f"[Enrichment] Exception for source={source}: {e}")
        return f"Source Section: {source}\n\n{raw_text}"   # never crash pipeline

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
        return cache[key]                          # skip API call

    enriched = await enrich_with_hf_summary(source, raw_text)
    cache[key] = enriched
    return enriched