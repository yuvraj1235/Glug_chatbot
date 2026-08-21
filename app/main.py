from fastapi import FastAPI, Request
from app.routes import chat, ingest, scrap
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv

# Import the token generator from your security utility
# (Assuming you save the previous security script as app/security.py)
from app.security import generate_token

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

app = FastAPI(title="GLUG Chatbot API")

default_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "https://club-website-2-0-sable.vercel.app",
    "https://www.nitdgplug.org" 
]

env_origins = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
origins = origins or default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # IMPORTANT: Expose the custom token header so the frontend can read it
    expose_headers=["x-chat-token"] 
)

# --- Security Route ---
@app.get("/chat/init", tags=["Security"])
async def init_chat(request: Request):
    """Provides the initial token to the frontend to begin chatting."""
    client_ip = request.client.host
    if forwarded_for := request.headers.get("x-forwarded-for"):
        client_ip = forwarded_for.split(",")[0].strip()
        
    return {"initialToken": generate_token(client_ip)}

# --- Routers ---
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(scrap.router)