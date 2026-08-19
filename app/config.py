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

    # 2. Enforce Groq configuration for fast, free cloud text generation
    GROQ_API_KEY: str | None = None
    DEFAULT_MODEL: str = "openai/gpt-oss-20b"

    # 3. Cloud tokens & optional services
    HUGGINGFACEHUB_API_TOKEN: str | None = None
    HF_MODEL_REPO: str | None = None
    GEMINI_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
    USE_RERANKER: bool = True
    
    # 4. Redis configuration (for response caching)
    REDIS_URL: str = "redis://localhost:6379/0"

    # 5. In-Memory Cooldown Duration
    PROMPT_COOLDOWN_SECONDS: int = 30

settings = Settings()