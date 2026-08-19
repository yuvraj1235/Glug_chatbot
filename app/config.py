import os
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "../.env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 1. Supabase Configurations
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str

    # 2. AWS Bedrock Configuration
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION_NAME: str = "us-east-1"  # Strongly recommended for Amazon Nova models
    DEFAULT_MODEL: str = "amazon.nova-micro-v1:0" # Switched to Nova Micro

    # 3. Cloud tokens & optional services
    HUGGINGFACEHUB_API_TOKEN: str | None = None
    HF_MODEL_REPO: str | None = None
    GEMINI_API_KEY: str | None = None
    COHERE_API_KEY: str | None = None
    USE_RERANKER: bool = True
    
    # 4. Redis configuration
    REDIS_URL: str = "redis://localhost:6379/0"

    # 5. In-Memory Cooldown Duration
    PROMPT_COOLDOWN_SECONDS: int = 30

settings = Settings()