from fastapi import FastAPI
from app.routes.chat import router as chat_router

app = FastAPI(title="Club Chatbot API")

app.include_router(chat_router)

@app.get("/health")
async def root():
    return {"status": "server running"}