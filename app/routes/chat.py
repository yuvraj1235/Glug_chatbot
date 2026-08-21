from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import json
from app.schemas.chat import ChatRequest
from app.services.pyq_matcher import get_pyq_response
from app.services.llm import LLMService, get_llm_service
from app.services.rate_limiter import RateLimiter

# Import our security functions (adjust the path if needed)
from app.security import verify_chat_token, generate_token

router = APIRouter()

@router.post("/chat", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def chat(
    req: ChatRequest,
    llm: LLMService = Depends(get_llm_service),
    client_ip: str = Depends(verify_chat_token)  # <-- INJECT SECURITY DEPENDENCY
):
    # Generate the fresh token for the client's next request
    next_token = generate_token(client_ip)
    
    # Attach it to the response headers
    response_headers = {
        "x-chat-token": next_token
    }

    # 1. Try to match and return exact PYQ on demand.
    pyq_result = await get_pyq_response(req.message)
    if pyq_result is not None:
        async def pyq_stream():
            yield f"data: {json.dumps({'response': pyq_result})}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(
            pyq_stream(), 
            media_type="text/event-stream",
            headers=response_headers  # <-- ADD HEADER TO STREAM
        )

    # 2. Otherwise fall back to General LLM Conversation
    return StreamingResponse(
        llm.generate_response(req.message), 
        media_type="text/event-stream",
        headers=response_headers      # <-- ADD HEADER TO STREAM
    )