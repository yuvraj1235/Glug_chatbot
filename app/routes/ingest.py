from fastapi import APIRouter, HTTPException, Depends
from app.services.api_scraper import scrape_all_endpoints
from app.services.rate_limiter import RateLimiter

router = APIRouter()

@router.post("/api/ingest", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def ingest_data():
    try:
        result = await scrape_all_endpoints()
        try:
            from app.services.llm import get_llm_service
            llm = get_llm_service()
            await llm.refresh_cache()
        except Exception as cache_err:
            # log warning but don't fail ingestion if cache refresh fails
            import logging
            logger = logging.getLogger("chatbot")
            logger.warning(f"Could not refresh vector store cache: {cache_err}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion lifecycle failed: {str(e)}")