# app/services/csv_scraper.py
import csv
import io
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from app.services.api_scraper import supabase_client, embeddings

logger = logging.getLogger("chatbot")


def parse_csv_to_documents(content: bytes, source_label: str, filename: str) -> list[Document]:
    """
    Parse a CSV with this layout:
      Row 1 : Title / context heading  (e.g. "RESTAURANTS NEAR NIT DURGAPUR")
      Row 2 : Column headers           (e.g. SNO., RESTAURANT, MAPS URL)
      Row 3+: Data rows

    Each data row becomes one LangChain Document.
    The title is prepended to every document so the LLM always knows
    which collection the entry belongs to.
    Empty / whitespace-only cells are skipped.
    """
    docs: list[Document] = []
    text = content.decode("utf-8-sig")           # strip UTF-8 BOM if present
    raw_reader = csv.reader(io.StringIO(text))
    all_rows = list(raw_reader)

    if len(all_rows) < 2:
        logger.warning(f"[CSV] '{filename}' has fewer than 2 rows — nothing to parse.")
        return docs

    # Row 0 → title (merge all non-empty cells into one string)
    csv_title = " ".join(cell.strip() for cell in all_rows[0] if cell.strip())
    logger.info(f"[CSV] Title detected: '{csv_title}'")

    # Row 1 → column headers
    headers = [h.strip() for h in all_rows[1]]

    # Rows 2+ → data
    for row_idx, raw_row in enumerate(all_rows[2:], start=1):
        # Zip with headers; skip empty cells
        parts = []
        for header, val in zip(headers, raw_row):
            val = val.strip()
            if header and val:
                parts.append(f"{header}: {val}")

        if not parts:
            continue

        # Prepend the title so every chunk carries its context
        page_content = f"{csv_title}\n\n" + "\n".join(parts)

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": source_label,
                    "csv_title": csv_title,
                    "endpoint": "csv_upload",
                    "url": f"file://{filename}",
                    "row": row_idx,
                },
            )
        )

    logger.info(f"[CSV] Parsed {len(docs)} data rows from '{filename}'.")
    return docs


async def upload_documents_to_vectordb(docs: list[Document], chunk_size: int = 5) -> dict:
    """
    Embed and upsert a list of Documents into the Supabase vector store.
    Returns a summary dict.
    """
    if not supabase_client:
        raise RuntimeError("Supabase client is not initialised. Check environment variables.")

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
