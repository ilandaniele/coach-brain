"""Embeddings con BGE-M3 local en GPU."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from coach_brain.settings import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Carga lazy del modelo. Cached para no recargar entre llamadas."""
    return SentenceTransformer(settings.embed_model, device=settings.embed_device)


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed batch de textos. Devuelve lista de vectores normalizados."""
    if not texts:
        return []
    model = get_model()
    arr = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return arr.tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def embed_dim() -> int:
    """Dimensión del vector. bge-m3 = 1024."""
    model = get_model()
    # API renombrada en sentence-transformers 5.x; fallback al nombre viejo.
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()
