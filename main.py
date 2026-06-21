from fastapi import FastAPI
from app.routes import chat, ingest
import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from your .env file into os.environ

# This is the "app" that Uvicorn is looking for!
app = FastAPI(title="GLUG Chatbot API")

# Attach your routers to the main app
app.include_router(chat.router)
app.include_router(ingest.router)