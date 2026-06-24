from fastapi import APIRouter, Depends
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.pyq import get_pyq_response
from app.services.llm import LLMService, get_llm_service
from app.services.rate_limiter import RateLimiter

router = APIRouter()

@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def chat(
    req: ChatRequest,
    llm: LLMService = Depends(get_llm_service)
):
    # 1. Try to match and return exact PYQ on demand
    pyq_result = await get_pyq_response(req.message)
    if pyq_result is not None:
        return ChatResponse(response=pyq_result)
        
    # 2. Otherwise fall back to General LLM Conversation
    ai_reply = await llm.generate_response(req.message)
    return ChatResponse(response=ai_reply)