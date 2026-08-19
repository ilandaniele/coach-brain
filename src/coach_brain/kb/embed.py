"""Embeddings con BGE-M3 local en GPU."""

from __future__ import annotations

import threading
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from coach_brain.settings import settings

_MODEL_LOCK = threading.Lock()


def get_model() -> SentenceTransformer:
    """Carga lazy del modelo, serializada.

    bge-m3 son ~2.2GB en fp32: sin el mutex, dos threads de Streamlit podían
    cargarlo en paralelo y duplicar el pico de memoria (OOM en un VM de 4GB).
    """
    with _MODEL_LOCK:
        return _get_model_cached()


@lru_cache(maxsize=1)
def _get_model_cached() -> SentenceTransformer:
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
