# app/services/api_scraper.py
import re
import httpx
import hashlib
import json
import os
import uuid
from datetime import datetime
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
    "profiles":         "https://api.nitdgplug.org/api/profiles/",
    "about":            "https://api.nitdgplug.org/api/about/",
    "project":          "https://api.nitdgplug.org/api/project/",
    "contact":          "https://api.nitdgplug.org/api/contact/",
    "linit":            "https://api.nitdgplug.org/api/linit/",
    "timeline_monthly": "https://api.nitdgplug.org/api/timeline_monthly/",
    "alumni":           "https://api.nitdgplug.org/api/alumni/",
    "facads":           "https://api.nitdgplug.org/api/facads/",
    "events":           "https://api.nitdgplug.org/api/events/"
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
# Main scraper (with aggregate roster building)
# ─────────────────────────────────────────

async def scrape_all_endpoints() -> dict:
    all_documents = []
    summary_results = {}
    cache = load_cache()

    # Collectors to keep names/roles for aggregate roster summaries
    roster_collectors = {
        "profiles": [],
        "alumni": [],
        "facads": []
    }

    async with httpx.AsyncClient() as client:
        for source_name, url in API_ENDPOINTS.items():
            try:
                response = await client.get(url, timeout=10.0)
                if response.status_code != 200:
                    summary_results[source_name] = f"Skipped (Status {response.status_code})"
                    continue

                items = response.json()
                if source_name == "timeline_monthly" and isinstance(items, dict):
                    def _parse_month(k):
                        try:
                            return datetime.strptime(k.strip(), "%B %Y")
                        except ValueError:
                            return datetime.min
                    sorted_months = sorted(items.keys(), key=_parse_month, reverse=True)[:3]
                    extracted_events = []
                    for m in sorted_months:
                        month_events = items.get(m, [])
                        if isinstance(month_events, list):
                            for ev in month_events:
                                if isinstance(ev, dict):
                                    extracted_events.append({"month": m, **ev})
                    items = extracted_events
                elif isinstance(items, dict):
                    items = [items]

                clean_source = source_name

                source_docs_count = 0
                for item in items:
                    if not isinstance(item, dict):
                        continue

                    # Capture name profiles for structural counting before flattening strings
                    if clean_source in roster_collectors:
                        first_name = item.get("first_name", "").strip()
                        last_name = item.get("last_name", "").strip()
                        role = item.get("role", "").strip()
                        
                        if first_name or last_name:
                            full_name = f"{first_name} {last_name}".strip()
                            display_str = f"{full_name} ({role})" if role else full_name
                            roster_collectors[clean_source].append(display_str)

                    text = flatten_item(item)
                    if not text.strip():
                        continue

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

    save_cache(cache)

    # ─────────────────────────────────────────
    # Generate Master Summary Roster Chunks
    # ─────────────────────────────────────────
    for target_source, names_list in roster_collectors.items():
        if names_list:
            unique_names = sorted(list(set(names_list)))
            total_count = len(unique_names)
            
            summary_text = (
                f"Source Section: {target_source}\n\n"
                f"GLUG Official Complete {target_source.capitalize()} Summary. "
                f"Total registered count database-wide: {total_count}. "
                f"The exact list of all entry names includes: {', '.join(unique_names)}."
            )
            
            # Generate deterministic namespace UUID string based on source name
            # This ensures we overwrite previous master chunks instead of appending endlessly
            master_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"glug.chatbot.master.{target_source}"))
            
            master_doc = Document(
                page_content=summary_text,
                metadata={
                    "id": master_uuid,
                    "source": target_source,
                    "type": "summary",
                    "is_master": True,
                    "endpoint": "aggregated_summary",
                    "url": "internal://summary"
                }
            )
            all_documents.append(master_doc)
            summary_results[f"master_{target_source}_summary"] = f"Generated complete master roster summary with {total_count} records"

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