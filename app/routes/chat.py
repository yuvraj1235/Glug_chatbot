from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import json
from app.schemas.chat import ChatRequest
from app.services.pyq import get_pyq_response
from app.services.llm import LLMService, get_llm_service
from app.services.rate_limiter import RateLimiter

router = APIRouter()

@router.post("/chat", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def chat(
    req: ChatRequest,
    llm: LLMService = Depends(get_llm_service)
):
    # 1. Try to match and return exact PYQ on demand
    pyq_result = await get_pyq_response(req.message)
    if pyq_result is not None:
        async def pyq_stream():
            yield f"data: {json.dumps({'response': pyq_result})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(pyq_stream(), media_type="text/event-stream")
        
    # 2. Otherwise fall back to General LLM Conversation
    return StreamingResponse(llm.generate_response(req.message), media_type="text/event-stream")