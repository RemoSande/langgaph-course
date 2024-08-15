from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str
    TEST_DATABASE_URL: str
    OPENAI_API_KEY: str
    USE_ASYNC: bool = True
    API_PORT: int = 8000

    S3_BUCKET_NAME: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    ENVIRONMENT: str = "development"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

settings = Settings()
