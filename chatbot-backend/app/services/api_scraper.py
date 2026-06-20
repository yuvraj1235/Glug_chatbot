import httpx
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../../chroma_data")

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
            # Format nicely as "Key: Value" if key exists
            prefix = f"{current_key.replace('_', ' ').title()}: " if current_key else ""
            extracted.append(f"{prefix}{cleaned}")
    elif isinstance(data, (int, float, bool)):
        if current_key not in ignored_keys:
            extracted.append(f"{current_key.replace('_', ' ').title()}: {data}")
            
    return extracted

async def scrape_all_endpoints() -> dict:
    """Scrapes all endpoints, processes data into vector documents, and stores them."""
    all_documents = []
    summary_results = {}

    async with httpx.AsyncClient() as client:
        for source_name, url in API_ENDPOINTS.items():
            try:
                # Handle potential 404s or incomplete backend routes gracefully
                response = await client.get(url, timeout=10.0)
                if response.status_code != 200:
                    summary_results[source_name] = f"Skipped (Status {response.status_code})"
                    continue

                items = response.json()
                # Ensure we are wrapping single object responses into a list loop safely
                if isinstance(items, dict):
                    items = [items]

                source_docs_count = 0
                for item in items:
                    # Flatten the JSON item into structured sentences
                    text_parts = extract_text_deduplicated(item)
                    if not text_parts:
                        continue
                    
                    combined_text = "\n".join(text_parts)
                    
                    # Store with metadata so the LLM can reference the right section
                    doc = Document(
                        page_content=combined_text,
                        metadata={"source": source_name, "url": url}
                    )
                    all_documents.append(doc)
                    source_docs_count += 1

                summary_results[source_name] = f"Successfully processed {source_docs_count} records"

            except Exception as e:
                summary_results[source_name] = f"Failed: {str(e)}"

    # Batch save all documents to ChromaDB if any were recovered
    if all_documents:
        db = Chroma.from_documents(all_documents, embeddings, persist_directory=CHROMA_PATH)
        return {
            "status": "Success",
            "details": summary_results,
            "total_documents_added": len(all_documents)
        }
    
    return {"status": "No documents found", "details": summary_results, "total_documents_added": 0}