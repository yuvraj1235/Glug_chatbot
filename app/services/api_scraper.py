import re
import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from supabase.client import Client, create_client
from app.config import settings

SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_SERVICE_KEY = settings.SUPABASE_SERVICE_KEY

supabase_client: Client = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    if SUPABASE_URL and SUPABASE_SERVICE_KEY
    else None
)

# BGE-large-en-v1.5 produces 1024-dim vectors
# Make sure your Supabase table is configured for vector(1024)
embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5",
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
    "limit":            "https://api.nitdgplug.org/api/limit/",
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

# Group related endpoints under a cleaner metadata source tag
SOURCE_GROUP = {
    "upcoming-events":  "events",
    "activity":         "events",
    "carousel":         "events",
    "timeline":         "events",
    "timeline_monthly": "events",
    "alumni-by-year":   "alumni",
    "facads":           "profiles",
}

# Block any key whose name suggests it holds media/image content
BLOCKED_KEY_RE = re.compile(
    r'image|photo|avatar|logo|icon|thumbnail|banner|picture|pic|img|media',
    re.IGNORECASE
)

# Block values that are media URLs even if the key name slipped through
MEDIA_VALUE_RE = re.compile(
    r'\.(jpg|jpeg|png|gif|webp|svg|bmp|mp4|mp3|wav)(\?.*)?$'
    r'|/(media|static|uploads|assets)/',
    re.IGNORECASE
)

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
    """
    Recursively flattens a JSON object into readable 'Key: Value' lines,
    skipping all media fields and null/empty values.
    """
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

            # Clean HTML tags if present
            if isinstance(node, str) and "<" in val_str and ">" in val_str:
                val_str = clean_html(val_str)

            if not val_str or len(val_str) <= 1 or val_str in seen:
                return

            seen.add(val_str)
            label = parent_key.replace("_", " ").title() if parent_key else ""
            parts.append(f"{label}: {val_str}" if label else val_str)

    recurse(data)
    return "\n".join(parts)

async def scrape_all_endpoints() -> dict:
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

                clean_source = SOURCE_GROUP.get(source_name, source_name)

                source_docs_count = 0
                for item in items:
                    if not isinstance(item, dict):
                        continue

                    text = flatten_item(item)
                    if not text.strip():
                        continue

                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": clean_source,
                            "endpoint": source_name,   # keep original for debugging
                            "url": url
                        }
                    )
                    all_documents.append(doc)
                    source_docs_count += 1

                summary_results[source_name] = f"Successfully processed {source_docs_count} records"

            except Exception as e:
                summary_results[source_name] = f"Failed: {str(e)}"

    if all_documents:
        await SupabaseVectorStore.afrom_documents(
            documents=all_documents,
            embedding=embeddings,
            client=supabase_client,
            table_name="documents",
            query_name="match_documents",
            chunk_size=100      # safe batch size for HF Inference API rate limits
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