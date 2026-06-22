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
from app.services.enrichment import load_cache, save_cache, enrich_cached

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