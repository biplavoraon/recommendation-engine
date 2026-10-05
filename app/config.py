from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = (
        "postgresql://recommender:recommender"
        "@localhost:5432/recommendations"
    )
    redis_url: str = "redis://localhost:6379/0"


settings = Settings()
