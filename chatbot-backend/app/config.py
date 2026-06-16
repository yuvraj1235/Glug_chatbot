from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    GEMINI_API_KEY: str = ""
    DEFAULT_MODEL: str = "gemini-1.5-flash"

    # Google Drive configuration
    GD_FOLDER_ID: str = ""
    GD_CREDENTIALS_PATH: str = "credentials.json"

settings = Settings()

