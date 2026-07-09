from fastapi import Request, HTTPException
from app.services.llm import get_llm_service

class RateLimiter:
    def __init__(self, times: int, seconds: int):
        self.times = times
        self.seconds = seconds

    async def __call__(self, request: Request):
        llm = get_llm_service()
        if not llm.redis_client:
            print("RateLimiter bypassed: No redis_client")
            return
        
        redis_client = llm.redis_client
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        
        key = f"rate_limit:{path}:{client_ip}"
        
        try:
            current = await redis_client.incr(key)
            if current == 1:
                await redis_client.expire(key, self.seconds)
                
            print(f"RateLimiter checking key {key}, current: {current}, limit: {self.times}")
            if current > self.times:
                print(f"RateLimiter THROWING 429 for key {key}")
                raise HTTPException(status_code=429, detail="Too Many Requests")
        except HTTPException:
            raise
        except Exception as e:
            print(f"RateLimiter Exception: {e}")
