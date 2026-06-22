# scraper.py
import re
import httpx
import hashlib
import json
import os
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from supabase.client import Client, create_client
from app.config import settings

SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_SERVICE_KEY = settings.SUPABASE_SERVICE_KEY

supabase_client: Client = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    if SUPABASE_URL and SUPABASE_SERVICE_KEY
    else None
)

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-base-en-v1.5",
    huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
)

API_ENDPOINTS = {
    "events":           "https://api.nitdgplug.org/api/events/",
    "upcoming-events":  "https://api.nitdgplug.org/api/upcoming-events/",
    "profiles":         "https://api.nitdgplug.org/api/profiles/",
    "about":            "https://api.nitdgplug.org/api/about/",
    "project":          "https://api.nitdgplug.org/api/project/",
    "contact":          "https://api.nitdgplug.org/api/contact/",
    "activity":         "https://api.nitdgplug.org/api/activity/",
    "carousel":         "https://api.nitdgplug.org/api/carousel/",
    "linit":            "https://api.nitdgplug.org/api/linit/",
    "timeline":         "https://api.nitdgplug.org/api/timeline/",
    "timeline_monthly": "https://api.nitdgplug.org/api/timeline_monthly/",
    "alumni":           "https://api.nitdgplug.org/api/alumni/",
    "facads":           "https://api.nitdgplug.org/api/facads/",
    "alumni-by-year":   "https://api.nitdgplug.org/api/alumni-by-year/",
    "techbytes":        "https://api.nitdgplug.org/api/techbytes/",
    "devposts":         "https://api.nitdgplug.org/api/devposts/",
    "configs":          "https://api.nitdgplug.org/api/configs/",
    "ctf":              "https://api.nitdgplug.org/api/ctf/"
}

SOURCE_GROUP = {
    "upcoming-events":  "events",
    "activity":         "events",
    "carousel":         "events",
    "timeline":         "events",
    "timeline_monthly": "events",
    "alumni-by-year":   "alumni",
    "facads":           "profiles",
}

BLOCKED_KEY_RE = re.compile(
    r'image|photo|avatar|logo|icon|thumbnail|banner|picture|pic|img|media',
    re.IGNORECASE
)

MEDIA_VALUE_RE = re.compile(
    r'\.(jpg|jpeg|png|gif|webp|svg|bmp|mp4|mp3|wav)(\?.*)?$'
    r'|/(media|static|uploads|assets)/',
    re.IGNORECASE
)

# ─────────────────────────────────────────
# Enrichment config
# ─────────────────────────────────────────

HF_SUMMARIZATION_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
CACHE_FILE = "hf_enrichment_cache.json"

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# ─────────────────────────────────────────
# Existing helpers (unchanged)
# ─────────────────────────────────────────

def should_skip(key: str, val) -> bool:
    if BLOCKED_KEY_RE.search(key):
        return True
    if val is None or val == "" or val == [] or val == {}:
        return True
    if isinstance(val, str) and MEDIA_VALUE_RE.search(val.strip()):
        return True
    return False

def clean_html(html_content: str) -> str:
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ").strip()

def flatten_item(data: dict) -> str:
    parts = []
    seen = set()

    def recurse(node, parent_key=""):
        if isinstance(node, dict):
            for key, val in node.items():
                if should_skip(key, val):
                    continue
                recurse(val, parent_key=key)
        elif isinstance(node, list):
            for item in node:
                recurse(item, parent_key)
        elif isinstance(node, (str, int, float, bool)):
            val_str = str(node).strip()
            if isinstance(node, str) and "<" in val_str and ">" in val_str:
                val_str = clean_html(val_str)
            if not val_str or len(val_str) <= 1 or val_str in seen:
                return
            seen.add(val_str)
            label = parent_key.replace("_", " ").title() if parent_key else ""
            parts.append(f"{label}: {val_str}" if label else val_str)

    recurse(data)
    return "\n".join(parts)

# ─────────────────────────────────────────
# Enrichment functions (NEW)
# ─────────────────────────────────────────

async def enrich_with_hf_summary(source: str, raw_text: str) -> str:
    """
    Calls Llama-3 via Groq API to generate a natural language summary.
    Prepends it to raw chunk to create semantic bridge for RAG.
    Falls back to raw_text silently on any failure.
    Includes rate limit retry logic (429 handling) and pacing.
    """
    import asyncio

    # Too short to summarize — just tag the source
    if len(raw_text.split("\n")) < 4:
        return f"Source Section: {source}\n\n{raw_text}"

    if not settings.GROQ_API_KEY:
        print("[Enrichment] Groq API Key missing. Skipping summary enrichment.")
        return f"Source Section: {source}\n\n{raw_text}"

    try:
        async with httpx.AsyncClient() as client:
            for attempt in range(3):
                # Strict rate-limiting sleep before every request attempt (guarantees < 30 requests per minute)
                await asyncio.sleep(2.1)

                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
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
                    print(f"[Enrichment] Groq returned {response.status_code} for source={source}")
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
        return f"Source Section: {source}\n\n{raw_text}"  # never crash pipeline




async def enrich_cached(source: str, raw_text: str, cache: dict) -> str:
    """
    Wraps enrich_with_hf_summary with MD5-based caching.
    Same chunk won't hit the HF API twice across scrape runs.
    """
    key = hashlib.md5(raw_text.encode()).hexdigest()

    if key in cache:
        return cache[key]                             # cache hit → skip API

    enriched = await enrich_with_hf_summary(source, raw_text)
    cache[key] = enriched
    return enriched

# ─────────────────────────────────────────
# Main scraper (enrichment wired in)
# ─────────────────────────────────────────

async def scrape_all_endpoints() -> dict:
    all_documents = []
    summary_results = {}
    cache = load_cache()                              # ✅ load cache at start

    async with httpx.AsyncClient() as client:
        for source_name, url in API_ENDPOINTS.items():
            try:
                response = await client.get(url, timeout=10.0)
                if response.status_code != 200:
                    summary_results[source_name] = f"Skipped (Status {response.status_code})"
                    continue

                items = response.json()
                if isinstance(items, dict):
                    items = [items]

                clean_source = SOURCE_GROUP.get(source_name, source_name)

                source_docs_count = 0
                for item in items:
                    if not isinstance(item, dict):
                        continue

                    text = flatten_item(item)
                    if not text.strip():
                        continue

                    # ✅ Only addition — enrich before storing
                    text = await enrich_cached(clean_source, text, cache)

                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": clean_source,
                            "endpoint": source_name,
                            "url": url
                        }
                    )
                    all_documents.append(doc)
                    source_docs_count += 1

                summary_results[source_name] = f"Successfully processed {source_docs_count} records"

            except Exception as e:
                summary_results[source_name] = f"Failed: {str(e)}"

    save_cache(cache)                                 # ✅ persist cache after all done

    if all_documents:
        await SupabaseVectorStore.afrom_documents(
            documents=all_documents,
            embedding=embeddings,
            client=supabase_client,
            table_name="documents",
            query_name="match_documents",
            chunk_size=5
        )
        return {
            "status": "Success",
            "details": summary_results,
            "total_documents_added": len(all_documents)
        }

    return {
        "status": "No documents found",
        "details": summary_results,
        "total_documents_added": 0
    }