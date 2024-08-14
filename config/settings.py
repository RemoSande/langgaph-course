from pydantic_settings import BaseSettings

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    DATABASE_URL: str
    TEST_DATABASE_URL: str
    OPENAI_API_KEY: str
    USE_ASYNC: bool = False
    API_PORT: int = 8000
    
    s3_bucket_name: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    environment: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()
