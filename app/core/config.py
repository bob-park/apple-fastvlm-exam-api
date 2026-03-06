from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "apple-fastvlm-api"
    log_level: str = "INFO"
    inference_device: str = "auto"

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/fastvlm"

    storage_root: Path = Path("./storage")
    model_root: Path = Path("./models")

    fastvlm_model_id: str = "apple/FastVLM-0.5B"
    fastvlm_local_dir: Path = Path("./models/fastvlm")

    text_embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embed_local_dir: Path = Path("./models/text-embedding")

    insightface_model_name: str = "buffalo_l"
    insightface_model_dir: Path = Path("./models/insightface")

    text_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    face_similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    pipeline_concurrency: int = Field(default=1, ge=1)

    page_size_default: int = Field(default=20, ge=1)
    page_size_max: int = Field(default=100, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    settings.model_root.mkdir(parents=True, exist_ok=True)
    return settings
