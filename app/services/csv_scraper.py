# app/services/csv_scraper.py
import csv
import io
import logging
import asyncio
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from app.services.api_scraper import supabase_client, embeddings
from app.services.enrichment import load_cache, save_cache, enrich_cached

logger = logging.getLogger("chatbot")

async def parse_csv_to_documents(content: bytes, source_label: str, filename: str) -> list[Document]:
    """
    Parses a CSV, maps rows to headers, and uses Groq LLM to rewrite the 
    raw tabular data into 'common sense' natural language prose.
    """
    docs: list[Document] = []
    text = content.decode("utf-8-sig")           
    raw_reader = csv.reader(io.StringIO(text))
    all_rows = list(raw_reader)

    if len(all_rows) < 2:
        logger.warning(f"[CSV] '{filename}' has fewer than 2 rows — nothing to parse.")
        return docs

    # Row 0 → title
    csv_title = " ".join(cell.strip() for cell in all_rows[0] if cell.strip())
    if not csv_title:
        csv_title = filename.replace(".csv", "").replace("_", " ")
        
    logger.info(f"[CSV] Title detected: '{csv_title}'")

    # Row 1 → column headers
    headers = [h.strip() for h in all_rows[1]]
    
    # Load the Groq translation cache
    cache = load_cache()

    # Rows 2+ → data
    for row_idx, raw_row in enumerate(all_rows[2:], start=1):
        parts = []
        for header, val in zip(headers, raw_row):
            val = val.strip()
            # Skip empty cells or those weird visual CSV dashes
            if header and val and val != "—":
                parts.append(f"{header}: {val}")

        if not parts:
            continue

        # Create the raw payload
        raw_text = f"Context: {csv_title}\n" + "\n".join(parts)

        # 🚀 MAGIC STEP: Translate the raw CSV row into common sense prose!
        prose_text = await enrich_cached(source_label, raw_text, cache)

        docs.append(
            Document(
                page_content=prose_text,
                metadata={
                    "source": source_label,
                    "csv_title": csv_title,
                    "endpoint": "csv_upload",
                    "url": f"file://{filename}",
                    "row": row_idx,
                },
            )
        )

    # Save the cache so we don't repeat Groq calls on re-runs
    save_cache(cache)
    logger.info(f"[CSV] Parsed and enriched {len(docs)} data rows from '{filename}'.")
    return docs


async def upload_documents_to_vectordb(docs: list[Document], chunk_size: int = 5) -> dict:
    """
    Embed and upsert a list of Documents into the Supabase vector store.
    """
    if not supabase_client:
        raise RuntimeError("Supabase client is not initialised.")

    if not docs:
        return {"status": "No documents", "total_documents_added": 0}

    await SupabaseVectorStore.afrom_documents(
        documents=docs,
        embedding=embeddings,
        client=supabase_client,
        table_name="documents",
        query_name="match_documents",
        chunk_size=chunk_size,
    )
    logger.info(f"Uploaded {len(docs)} CSV-derived documents to vector DB.")
    return {
        "status": "Success",
        "total_documents_added": len(docs),
    }