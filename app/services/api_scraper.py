import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
# 1. FIXED: Swapped to the correct non-deprecated module from langchain_huggingface
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from supabase.client import Client, create_client
from app.config import settings

SUPABASE_URL = settings.SUPABASE_URL
SUPABASE_SERVICE_KEY = settings.SUPABASE_SERVICE_KEY 

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_URL and SUPABASE_SERVICE_KEY else None

# 2. FIXED: Use HuggingFaceEndpointEmbeddings for direct Serverless Cloud Calls
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN
)

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
    """Removes HTML tags and returns clean text content."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(separator=" ").strip()

def extract_text_deduplicated(data) -> list:
    """Recursively extracts values from JSON payloads while keeping text unique."""
    extracted = []
    seen = set()

    def recurse(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key in ["id", "image", "logo", "avatar", "icon", "url"] and isinstance(val, (int, str)):
                    continue
                recurse(val)
        elif isinstance(node, list):
            for item in node:
                recurse(item)
        elif isinstance(node, (str, int, float)):
            val_str = str(node).strip()
            if "<" in val_str and ">" in val_str:
                val_str = clean_html(val_str)
            
            if val_str and val_str not in seen and len(val_str) > 1:
                seen.add(val_str)
                extracted.append(val_str)

    recurse(data)
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

    if all_documents:
        SupabaseVectorStore.from_documents(
            documents=all_documents,
            embedding=embeddings,
            client=supabase_client,
            table_name="documents",           
            query_name="match_documents",      
            chunk_size=200  
        )
        return {
            "status": "Success",
            "details": summary_results,
            "total_documents_added": len(all_documents)
        }
    
    return {"status": "No documents found", "details": summary_results, "total_documents_added": 0}