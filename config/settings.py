from pydantic_settings import BaseSettings

from pydantic import BaseSettings

class Settings(BaseSettings):
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    OPENAI_API_KEY: str
    USE_ASYNC: bool = False

    class Config:
        env_file = ".env"
    openai_api_key: str
    database_url: str
    s3_bucket_name: str
    s3_access_key: str
    s3_secret_key: str
    environment: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()
