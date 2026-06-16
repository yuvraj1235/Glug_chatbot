from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.chat import router as chat_router

app = FastAPI(title="Club Chatbot API")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(chat_router)

@app.get("/")
async def root():
    return {"status": "running"}