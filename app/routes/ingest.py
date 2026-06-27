from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import logging
from app.services.api_scraper import scrape_all_endpoints

router = APIRouter()
logger = logging.getLogger("chatbot")

# Global flag to prevent concurrent background runs
INGESTION_IN_PROGRESS = False

async def run_ingestion_task():
    global INGESTION_IN_PROGRESS
    try:
        logger.info("Starting background ingestion pipeline...")
        result = await scrape_all_endpoints()
        logger.info(f"Scraping complete: {result}")
        
        try:
            from app.services.llm import get_llm_service
            llm = get_llm_service()
            await llm.refresh_cache()
            logger.info("Vector store cache refreshed successfully.")
        except Exception as cache_err:
            logger.warning(f"Could not refresh vector store cache: {cache_err}")
            
    except Exception as e:
        logger.error(f"Ingestion background task failed: {str(e)}")
    finally:
        # Crucial: Reset flag when done or if it crashes
        INGESTION_IN_PROGRESS = False
        logger.info("Ingestion flag reset. Ready for next run.")

@router.post("/api/api-ingest")  # Double check if your route is /api/ingest or /api/api-ingest
async def ingest_data(background_tasks: BackgroundTasks):
    global INGESTION_IN_PROGRESS
    
    if INGESTION_IN_PROGRESS:
        raise HTTPException(
            status_code=429, 
            detail="Ingestion pipeline is already running in the background. Please wait for it to finish."
        )
    
    INGESTION_IN_PROGRESS = True
    background_tasks.add_task(run_ingestion_task)
    
    return {
        "status": "Accepted", 
        "message": "Ingestion pipeline started. Using local cache where available."
    }