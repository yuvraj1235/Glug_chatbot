from fastapi import APIRouter, HTTPException
from app.services.api_scraper import scrape_all_endpoints

router = APIRouter()

@router.post("/api/ingest")
async def ingest_data():
    try:
        result = await scrape_all_endpoints()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion lifecycle failed: {str(e)}")