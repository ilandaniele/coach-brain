"""Configuración central. Lee de .env y expone paths absolutos."""

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]


# Algunos entornos (CI, shells con profile pre-cargado) inyectan env vars como
# strings vacíos. Pydantic-settings les da prioridad sobre el .env, así que las
# limpiamos para que el .env pueda popular el field. Solo limpia vacíos.
def _purge_empty_envs(*names: str) -> None:
    for n in names:
        if n in os.environ and os.environ[n].strip() == "":
            del os.environ[n]


_purge_empty_envs(
    "ANTHROPIC_API_KEY",
    "COACH_MODEL_MAIN",
    "COACH_MODEL_FAST",
    "COACH_MODEL_EXTRACTOR",
    "QDRANT_URL",
    "QDRANT_PATH",
    "QDRANT_API_KEY",
    "EMBED_MODEL",
    "EMBED_DEVICE",
    "RERANK_MODEL",
    "WHISPER_MODEL",
    "WHISPER_DEVICE",
    "WHISPER_COMPUTE_TYPE",
    "WHISPER_LANGUAGE",
    "DATA_DIR",
    "PROMPTS_DIR",
    "LOG_LEVEL",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # Claude
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    model_main: str = Field(default="claude-sonnet-4-6", validation_alias="COACH_MODEL_MAIN")
    model_fast: str = Field(default="claude-haiku-4-5-20251001", validation_alias="COACH_MODEL_FAST")
    model_extractor: str = Field(default="claude-sonnet-4-6", validation_alias="COACH_MODEL_EXTRACTOR")

    # Backend de extracción: "anthropic" (API paga) | "ollama" (local gratis)
    extractor_backend: str = Field(default="anthropic", validation_alias="COACH_EXTRACTOR_BACKEND")
    ollama_host: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_HOST")
    ollama_model: str = Field(default="qwen2.5:14b-instruct", validation_alias="COACH_OLLAMA_MODEL")

    # Backend de la RESPUESTA del MVP: "ollama" (local gratis) | "anthropic" (Sonnet, pago)
    response_backend: str = Field(default="ollama", validation_alias="COACH_RESPONSE_BACKEND")

    # Qdrant
    qdrant_url: str = Field(default="", validation_alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", validation_alias="QDRANT_API_KEY")
    qdrant_path: str = Field(default="./qdrant_storage", validation_alias="QDRANT_PATH")

    # Ollama: False en cloud (sin GPU local)
    ollama_enabled: bool = Field(default=True, validation_alias="OLLAMA_ENABLED")

    # Embeddings
    embed_model: str = Field(default="BAAI/bge-m3", validation_alias="EMBED_MODEL")
    embed_device: str = Field(default="cuda", validation_alias="EMBED_DEVICE")
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3", validation_alias="RERANK_MODEL")

    # Whisper
    whisper_model: str = Field(default="large-v3", validation_alias="WHISPER_MODEL")
    whisper_device: str = Field(default="cuda", validation_alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="float16", validation_alias="WHISPER_COMPUTE_TYPE")
    whisper_language: str = Field(default="es", validation_alias="WHISPER_LANGUAGE")

    # Paths
    data_dir: str = Field(default="./data", validation_alias="DATA_DIR")
    prompts_dir: str = Field(default="./prompts", validation_alias="PROMPTS_DIR")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def prompts_path(self) -> Path:
        p = Path(self.prompts_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def raw_path(self) -> Path:
        return self.data_path / "raw"

    @property
    def processed_path(self) -> Path:
        return self.data_path / "processed"

    @property
    def curated_path(self) -> Path:
        return self.data_path / "curated"

    @property
    def transcripts_path(self) -> Path:
        return self.processed_path / "transcripts"

    @property
    def extracted_path(self) -> Path:
        return self.processed_path / "extracted"

    @property
    def qdrant_storage(self) -> Path:
        if self.qdrant_url:
            return Path("")
        p = Path(self.qdrant_path)
        return p if p.is_absolute() else REPO_ROOT / p


settings = Settings()
