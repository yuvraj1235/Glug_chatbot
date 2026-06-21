import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamically find the absolute path to the chatbot-backend/.env file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "../.env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 1. Enforce Supabase configurations strictly as required fields
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str

    # 2. Update model settings for Hugging Face Cloud Inference
    HF_MODEL_REPO: str = "Qwen/Qwen2.5-72B-Instruct"
    HUGGINGFACEHUB_API_TOKEN: str

    # 3. Make Gemini optional so the app doesn't crash if it's left out
    GEMINI_API_KEY: str | None = None
    DEFAULT_MODEL: str = "gemini-2.5-flash"

settings = Settings()