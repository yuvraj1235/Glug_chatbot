import asyncio
import json
import redis.asyncio as redis
from app.config import settings

async def main():
    r = redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    cache = await r.get("vector_store_documents")
    if not cache:
        print("Cache is empty.")
        return
    docs = json.loads(cache)
    print(f"Total docs in cache: {len(docs)}")
    
    sources = {}
    for doc in docs:
        src = doc.get("metadata", {}).get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        
    print("Sources count:")
    for k, v in sources.items():
        print(f"  {k}: {v}")

asyncio.run(main())
