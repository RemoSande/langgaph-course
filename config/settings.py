from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    database_url: str
    s3_bucket_name: str
    s3_access_key: str
    s3_secret_key: str
    environment: str = "development"

    class Config:
        env_file = ".env"

settings = Settings()