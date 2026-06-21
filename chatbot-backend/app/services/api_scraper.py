import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from supabase.client import Client, create_client
from app.config import settings

# 1. Initialize Supabase Cloud configurations instead of CHROMA_PATH
SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_SERVICE_KEY = settings.SUPABASE_SERVICE_KEY # Use service key to bypass RLS policies during writes

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_URL and SUPABASE_SERVICE_KEY else None

# Note: 'all-MiniLM-L6-v2' creates 384-dimensional vectors.
# Ensure your column in Supabase is defined as: embedding vector(384)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

API_ENDPOINTS = {
    "events": "https://api.nitdgplug.org/api/events/",
    "upcoming-events": "https://api.nitdgplug.org/api/upcoming-events/",
    "profiles": "https://api.nitdgplug.org/api/profiles/",
    "about": "https://api.nitdgplug.org/api/about/",
    "project": "https://api.nitdgplug.org/api/project/",
    "contact": "https://api.nitdgplug.org/api/contact/",
    "activity": "https://api.nitdgplug.org/api/activity/",
    "carousel": "https://api.nitdgplug.org/api/carousel/",
    "limit": "https://api.nitdgplug.org/api/limit/",
    "timeline": "https://api.nitdgplug.org/api/timeline/",
    "timeline_monthly": "https://api.nitdgplug.org/api/timeline_monthly/",
    "alumni": "https://api.nitdgplug.org/api/alumni/",
    "facads": "https://api.nitdgplug.org/api/facads/",
    "alumni-by-year": "https://api.nitdgplug.org/api/alumni-by-year/",
    "techbytes": "https://api.nitdgplug.org/api/techbytes/",
    "devposts": "https://api.nitdgplug.org/api/devposts/",
    "configs": "https://api.nitdgplug.org/api/configs/",
    "ctf": "https://api.nitdgplug.org/api/ctf/"
}

def clean_html(html_content: str) -> str:
    """Removes HTML tags and normalizes whitespace."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    return " ".join(soup.get_text().split())

def extract_text_deduplicated(data, current_key="") -> list:
    """
    Recursively extracts all meaningful text values from nested JSON data strings,
    ignoring common non-informative metadata keys.
    """
    ignored_keys = {"id", "image", "avatar", "created_at", "updated_at", "link", "url", "email"}
    extracted = []

    if isinstance(data, dict):
        for key, value in data.items():
            if key in ignored_keys:
                continue
            extracted.extend(extract_text_deduplicated(value, key))
    elif isinstance(data, list):
        for item in data:
            extracted.extend(extract_text_deduplicated(item, current_key))
    elif isinstance(data, str) and data.strip():
        cleaned = clean_html(data)
        if cleaned:
            prefix = f"{current_key.replace('_', ' ').title()}: " if current_key else ""
            extracted.append(f"{prefix}{cleaned}")
    elif isinstance(data, (int, float, bool)):
        if current_key not in ignored_keys:
            extracted.append(f"{current_key.replace('_', ' ').title()}: {data}")
            
    return extracted

async def scrape_all_endpoints() -> dict:
    """Scrapes all endpoints, processes data into vector documents, and stores them in Supabase."""
    all_documents = []
    summary_results = {}

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

                source_docs_count = 0
                for item in items:
                    text_parts = extract_text_deduplicated(item)
                    if not text_parts:
                        continue
                    
                    combined_text = "\n".join(text_parts)
                    
                    doc = Document(
                        page_content=combined_text,
                        metadata={"source": source_name, "url": url}
                    )
                    all_documents.append(doc)
                    source_docs_count += 1

                summary_results[source_name] = f"Successfully processed {source_docs_count} records"

            except Exception as e:
                summary_results[source_name] = f"Failed: {str(e)}"

    # 2. Replaced Chroma initialization with SupabaseVectorStore
    if all_documents:
        SupabaseVectorStore.from_documents(
            documents=all_documents,
            embedding=embeddings,
            client=supabase_client,
            table_name="documents",           # Your target pgvector table name
            query_name="match_documents",      # The custom similarity RPC function
            chunk_size=500                    # Batch uploads to protect cloud memory limits
        )
        return {
            "status": "Success",
            "details": summary_results,
            "total_documents_added": len(all_documents)
        }
    
    return {"status": "No documents found", "details": summary_results, "total_documents_added": 0}