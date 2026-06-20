import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamically find the absolute path to the chatbot-backend/.env file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "../.env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,  # Uses the absolute absolute path now
        env_file_encoding="utf-8",
        extra="ignore"
    )

    GEMINI_API_KEY: str
    DEFAULT_MODEL: str = "gemini-2.5-flash"

settings = Settings()