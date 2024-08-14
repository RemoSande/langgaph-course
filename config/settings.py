from pydantic_settings import BaseSettings

from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DB_HOST: str = "db"
    DB_PORT: int = 5432
    OPENAI_API_KEY: str
    USE_ASYNC: bool = False
    API_PORT: int = 8000
    TEST_POSTGRES_DB: str = "test_db"
    TEST_POSTGRES_USER: str = "test_user"
    TEST_POSTGRES_PASSWORD: str = "test_password"
    TEST_DB_PORT: int = 5432
    
    # Construct database URLs
    @property
    def database_url(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.POSTGRES_DB}"
    
    @property
    def test_database_url(self) -> str:
        return f"postgresql+psycopg://{self.TEST_POSTGRES_USER}:{self.TEST_POSTGRES_PASSWORD}@{self.DB_HOST}:{self.TEST_DB_PORT}/{self.TEST_POSTGRES_DB}"

    s3_bucket_name: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    environment: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()
