from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    POSTGRES_DB: Optional[str] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[str] = None
    TEST_DB_PORT: Optional[str] = None
    TEST_POSTGRES_DB: Optional[str] = None
    TEST_POSTGRES_USER: Optional[str] = None
    TEST_POSTGRES_PASSWORD: Optional[str] = None
    USE_ASYNC: bool = True
    API_PORT: int = 8000
    TEST_DATABASE_URL: str
    
    DATABASE_URL: str
    OPENAI_API_KEY: str


    S3_BUCKET_NAME: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    ENVIRONMENT: str = "development"

    # New fields to address "extra inputs are not permitted" errors
    PYTHONPATH: Optional[str] = None
    LANGCHAIN_TRACING_V2: Optional[str] = None
    LANGCHAIN_ENDPOINT: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: Optional[str] = None
    LLAMA_CLOUD_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    TAVILY_API_KEY: Optional[str] = None
  

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra='ignore')

settings = Settings()
