from fastapi import FastAPI
from app.routes import chat, ingest, scrap
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from dotenv import load_dotenv

load_dotenv()  # This loads the variables from your .env file into os.environ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# This is the "app" that Uvicorn is looking for!
app = FastAPI(title="GLUG Chatbot API")

default_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "https://club-website-2-0-sable.vercel.app",
    "https://nitdgplug.org",
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
)

# Attach your routers to the main app
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(scrap.router)