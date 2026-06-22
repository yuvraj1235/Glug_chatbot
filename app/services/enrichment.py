# app/enrichment.py
import httpx
import hashlib
import json
import os

HF_SUMMARIZATION_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

CACHE_FILE = "hf_enrichment_cache.json"

# ─────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# ─────────────────────────────────────────
# Core summarizer
# ─────────────────────────────────────────

async def enrich_with_hf_summary(
    source: str,
    raw_text: str,
    hf_token: str
) -> str:
    """
    Calls Llama-3 via Groq API to generate a natural language summary.
    Prepends summary + source tag to raw chunk.
    Falls back to raw_text silently on any failure.
    Includes rate limit retry logic (429 handling) and pacing.
    """
    import asyncio

    # Too short to summarize meaningfully → skip
    if len(raw_text.split("\n")) < 4:
        return f"Source Section: {source}\n\n{raw_text}"

    try:
        from app.config import settings
        groq_key = settings.GROQ_API_KEY
    except Exception:
        groq_key = os.environ.get("GROQ_API_KEY")

    if not groq_key:
        print(f"[Enrichment] Groq API key not found. Skipping summary for source={source}")
        return f"Source Section: {source}\n\n{raw_text}"

    try:
        async with httpx.AsyncClient() as client:
            for attempt in range(3):
                # Strict rate-limiting sleep before every request attempt (guarantees < 30 requests per minute)
                await asyncio.sleep(2.1)

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
                    print(f"[Enrichment] Groq returned 429. Retrying in {sleep_time}s... (Attempt {attempt+1}/3)")
                    await asyncio.sleep(sleep_time)
                    continue

                if response.status_code != 200:
                    print(f"[Enrichment] Groq failed ({response.status_code}) for source={source}")
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
            print(f"[Enrichment] Groq rate limits exceeded. Skipping summary for source={source}")
            return f"Source Section: {source}\n\n{raw_text}"

    except Exception as e:
        print(f"[Enrichment] Exception for source={source}: {e}")
        return f"Source Section: {source}\n\n{raw_text}"   # never crash pipeline



# ─────────────────────────────────────────
# Cached wrapper
# ─────────────────────────────────────────

async def enrich_cached(
    source: str,
    raw_text: str,
    hf_token: str,
    cache: dict
) -> str:
    key = hashlib.md5(raw_text.encode()).hexdigest()

    if key in cache:
        return cache[key]                          # skip API call

    enriched = await enrich_with_hf_summary(source, raw_text, hf_token)
    cache[key] = enriched
    return enriched