# app/routes/scrap.py
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
import logging
from app.services.csv_scraper import parse_csv_to_documents, upload_documents_to_vectordb

router = APIRouter()
logger = logging.getLogger("chatbot")

# Global flag to prevent concurrent runs
CSV_INGEST_IN_PROGRESS = False


async def run_csv_ingest_task(content: bytes, source_label: str, filename: str):
    global CSV_INGEST_IN_PROGRESS
    try:
        logger.info(f"CSV ingest started: file='{filename}', source='{source_label}'")
        docs = parse_csv_to_documents(content, source_label, filename)
        logger.info(f"Parsed {len(docs)} rows from '{filename}'")

        result = await upload_documents_to_vectordb(docs)
        logger.info(f"CSV ingest complete: {result}")

        # Refresh retriever cache if LLM service is running
        try:
            from app.services.llm import get_llm_service
            llm = get_llm_service()
            await llm.refresh_cache()
            logger.info("Vector store cache refreshed after CSV ingest.")
        except Exception as cache_err:
            logger.warning(f"Could not refresh vector store cache: {cache_err}")

    except Exception as e:
        logger.error(f"CSV ingest background task failed: {e}")
    finally:
        CSV_INGEST_IN_PROGRESS = False
        logger.info("CSV ingest flag reset.")


@router.post("/api/scrap")
async def scrap_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file to ingest"),
    source: str = Form(
        default="csv",
        description="Source/category label stored in vector DB metadata (e.g. 'events', 'alumni')",
    ),
):
    """
    Upload a CSV file and ingest every row into the Supabase vector database.

    - **file**: a `.csv` file (UTF-8 or UTF-8-BOM encoded)
    - **source**: metadata tag applied to every document (default: `"csv"`)

    Each CSV row is converted into a natural-language document like:
    ```
    Column A: value
    Column B: value
    ```
    and then embedded and stored in the vector DB.
    Processing runs in the background; the endpoint returns immediately.
    """
    global CSV_INGEST_IN_PROGRESS

    if CSV_INGEST_IN_PROGRESS:
        raise HTTPException(
            status_code=429,
            detail="A CSV ingest is already running. Please wait for it to finish.",
        )

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    CSV_INGEST_IN_PROGRESS = True
    background_tasks.add_task(run_csv_ingest_task, content, source.strip() or "csv", file.filename)

    return {
        "status": "Accepted",
        "message": f"CSV '{file.filename}' queued for ingestion under source '{source}'.",
    }
