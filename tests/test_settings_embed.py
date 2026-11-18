"""Configuración y embeddings.

`settings` resuelve todos los paths del pipeline, así que un error acá manda
cada etapa a escribir al lugar equivocado. Las instancias de prueba se crean
con `_env_file=None` para que el `.env` del repo no se meta.

De `kb/embed.py` se prueba todo menos cargar el modelo: bge-m3 son 2,2 GB y
la GPU está ocupada. El modelo se reemplaza por un doble.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from coach_brain import settings as S
from coach_brain.kb import embed as E
from coach_brain.settings import REPO_ROOT, Settings

# El `lru_cache` real, guardado antes de que cualquier prueba lo reemplace:
# el teardown de un fixture corre antes de que monkeypatch deshaga, asi que
# `E._get_model_cached` todavia es el doble cuando hay que limpiar el cache.
_CACHED = E._get_model_cached


def limpiar_cache() -> None:
    _CACHED.cache_clear()


def hacer(**kw) -> Settings:
    """Settings aislado del `.env` del repo."""
    return Settings(_env_file=None, **kw)


# --- limpieza de env vars vacías ------------------------------------------

def test_una_env_var_vacia_se_borra(monkeypatch):
    """Pydantic le da prioridad al entorno sobre el `.env`: una var vacía
    inyectada por CI dejaba el field en blanco y tapaba al `.env`."""
    monkeypatch.setenv("COACH_TEST_VACIA", "")
    S._purge_empty_envs("COACH_TEST_VACIA")
    assert "COACH_TEST_VACIA" not in os.environ


def test_solo_espacios_tambien_cuenta_como_vacia(monkeypatch):
    monkeypatch.setenv("COACH_TEST_VACIA", "   ")
    S._purge_empty_envs("COACH_TEST_VACIA")
    assert "COACH_TEST_VACIA" not in os.environ


def test_una_env_var_con_valor_no_se_toca(monkeypatch):
    monkeypatch.setenv("COACH_TEST_LLENA", "valor")
    S._purge_empty_envs("COACH_TEST_LLENA")
    assert os.environ["COACH_TEST_LLENA"] == "valor"


def test_una_env_var_que_no_existe_no_rompe(monkeypatch):
    monkeypatch.delenv("COACH_TEST_AUSENTE", raising=False)
    S._purge_empty_envs("COACH_TEST_AUSENTE")  # no levanta KeyError


# --- paths -----------------------------------------------------------------

def test_un_data_dir_relativo_cuelga_del_repo():
    s = hacer(data_dir="./data")
    assert s.data_path == REPO_ROOT / "data"


def test_un_data_dir_absoluto_se_respeta(tmp_path):
    assert hacer(data_dir=str(tmp_path)).data_path == tmp_path


def test_los_paths_del_pipeline_cuelgan_de_data_path(tmp_path):
    s = hacer(data_dir=str(tmp_path))
    assert s.raw_path == tmp_path / "raw"
    assert s.processed_path == tmp_path / "processed"
    assert s.curated_path == tmp_path / "curated"
    assert s.transcripts_path == tmp_path / "processed" / "transcripts"
    assert s.extracted_path == tmp_path / "processed" / "extracted"


def test_los_prompts_tambien_resuelven_relativo_al_repo():
    assert hacer(prompts_dir="./prompts").prompts_path == REPO_ROOT / "prompts"


def test_un_prompts_dir_absoluto_se_respeta(tmp_path):
    assert hacer(prompts_dir=str(tmp_path)).prompts_path == tmp_path


def test_leer_un_path_no_crea_nada(tmp_path):
    """Las properties resuelven, no crean: cada etapa hace su propio mkdir."""
    s = hacer(data_dir=str(tmp_path / "noexiste"))
    _ = s.transcripts_path
    assert not (tmp_path / "noexiste").exists()


# --- Qdrant: local contra remoto ------------------------------------------

def test_sin_url_el_qdrant_es_un_path_local():
    s = hacer(qdrant_url="", qdrant_path="./qdrant_storage")
    assert s.qdrant_storage == REPO_ROOT / "qdrant_storage"


def test_un_qdrant_path_absoluto_se_respeta(tmp_path):
    assert hacer(qdrant_url="", qdrant_path=str(tmp_path)).qdrant_storage == tmp_path


def test_con_url_no_hay_storage_local():
    """Con Qdrant remoto no se abre ningún archivo: el path tiene que quedar
    vacío para que nadie tome el lock de un embedded que no se usa."""
    s = hacer(qdrant_url="https://qdrant.example", qdrant_path="./qdrant_storage")
    assert s.qdrant_storage == Path("")


# --- defaults y env vars ---------------------------------------------------

def test_los_defaults_son_los_del_setup_local():
    s = hacer()
    assert s.embed_model == "BAAI/bge-m3"
    assert s.embed_device == "cuda"
    assert s.whisper_model == "large-v3"
    assert s.whisper_compute_type == "float16"
    assert s.whisper_language == "es"
    assert s.extractor_backend == "anthropic"
    assert s.response_backend == "ollama"
    assert s.ollama_enabled is True


def test_sin_claves_no_hay_claves():
    s = hacer()
    assert s.anthropic_api_key == ""
    assert s.gemini_api_key == ""
    assert s.app_password == ""  # vacío = sin login, solo local


def test_el_alias_de_la_env_var_pobla_el_field(monkeypatch):
    monkeypatch.setenv("COACH_EXTRACTOR_BACKEND", "ollama")
    assert hacer().extractor_backend == "ollama"


def test_el_nombre_del_field_en_minuscula_tambien_sirve(monkeypatch):
    """`case_sensitive=False`: `data_dir` y `DATA_DIR` son la misma var."""
    monkeypatch.setenv("data_dir", "/tmp/x")
    assert hacer().data_dir == "/tmp/x"


def test_ollama_se_apaga_desde_el_entorno(monkeypatch):
    """En Fly no hay GPU: `OLLAMA_ENABLED=false` tiene que leerse como bool."""
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    assert hacer().ollama_enabled is False
    monkeypatch.setenv("OLLAMA_ENABLED", "0")
    assert hacer().ollama_enabled is False


def test_una_env_var_que_no_es_un_field_se_ignora(monkeypatch):
    """`extra="ignore"`: el entorno de Fly tiene decenas de vars propias."""
    monkeypatch.setenv("FLY_APP_NAME", "coach-brain")
    assert hacer().log_level == "INFO"  # no levanta ValidationError


def test_el_override_de_modelo_por_proveedor_arranca_vacio():
    s = hacer()
    assert s.coach_model_gemini == ""
    assert s.coach_model_groq == ""


def test_el_singleton_es_mutable(monkeypatch):
    """Las pruebas del resto del repo dependen de poder pisar un field."""
    from coach_brain.settings import settings
    original = settings.data_dir
    monkeypatch.setattr(settings, "data_dir", "/otro")
    assert settings.data_dir == "/otro"
    monkeypatch.undo()
    assert settings.data_dir == original


# --- embeddings ------------------------------------------------------------

class FakeModel:
    """Doble de SentenceTransformer: registra con qué se lo llamó."""

    def __init__(self, nombre=None, device=None, dim=1024):
        self.nombre = nombre
        self.device = device
        self.dim = dim
        self.llamadas: list[dict] = []

    def encode(self, texts, **kw):
        self.llamadas.append({"texts": texts, **kw})
        return np.zeros((len(texts), self.dim), dtype="float32")

    def get_embedding_dimension(self):
        return self.dim

    def get_sentence_embedding_dimension(self):
        return self.dim


@pytest.fixture
def modelo(monkeypatch):
    """Reemplaza el modelo y limpia el cache antes y después."""
    limpiar_cache()
    fake = FakeModel()
    monkeypatch.setattr(E, "_get_model_cached", lambda: fake)
    yield fake
    limpiar_cache()


def test_una_lista_vacia_no_carga_el_modelo(monkeypatch):
    """Importa: cargar bge-m3 son 2,2 GB y una llamada vacía no los necesita."""
    limpiar_cache()

    def explota():
        raise AssertionError("no tendría que cargar el modelo")

    monkeypatch.setattr(E, "_get_model_cached", explota)
    assert E.embed([]) == []


def test_embed_devuelve_un_vector_por_texto(modelo):
    vecs = E.embed(["uno", "dos"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 1024


def test_embed_pide_vectores_normalizados(modelo):
    """El score de Qdrant es coseno: sin normalizar, `MIN_SCORE` no significa
    lo mismo."""
    E.embed(["x"])
    assert modelo.llamadas[0]["normalize_embeddings"] is True
    assert modelo.llamadas[0]["convert_to_numpy"] is True
    assert modelo.llamadas[0]["show_progress_bar"] is False


def test_el_batch_size_por_defecto_es_32(modelo):
    E.embed(["x"])
    assert modelo.llamadas[0]["batch_size"] == 32


def test_el_batch_size_se_puede_pisar(modelo):
    E.embed(["x"], batch_size=4)
    assert modelo.llamadas[0]["batch_size"] == 4


def test_embed_devuelve_listas_no_arrays(modelo):
    """Qdrant recibe JSON: un ndarray no serializa."""
    v = E.embed(["x"])
    assert isinstance(v, list) and isinstance(v[0], list)
    assert isinstance(v[0][0], float)


def test_embed_one_devuelve_un_solo_vector(modelo):
    v = E.embed_one("una consulta")
    assert len(v) == 1024
    assert modelo.llamadas[0]["texts"] == ["una consulta"]


def test_la_dimension_sale_de_la_api_nueva(modelo):
    assert E.embed_dim() == 1024


def test_con_la_api_vieja_usa_el_nombre_viejo(monkeypatch):
    """sentence-transformers renombró el método en la 5.x."""
    limpiar_cache()

    class SoloViejo:
        def get_sentence_embedding_dimension(self):
            return 768

    monkeypatch.setattr(E, "_get_model_cached", lambda: SoloViejo())
    assert E.embed_dim() == 768
    limpiar_cache()


def test_el_modelo_se_carga_una_sola_vez(monkeypatch):
    """El `lru_cache` más el mutex: dos threads de Streamlit duplicaban el
    pico de memoria cargándolo en paralelo."""
    limpiar_cache()
    cargas = []

    class Contador:
        def __init__(self, nombre, device=None):
            cargas.append((nombre, device))

    monkeypatch.setattr(E, "SentenceTransformer", Contador)
    a = E.get_model()
    b = E.get_model()
    assert a is b
    assert len(cargas) == 1
    limpiar_cache()


def test_el_modelo_se_carga_con_lo_que_dice_settings(monkeypatch):
    from coach_brain.settings import settings
    limpiar_cache()
    monkeypatch.setattr(settings, "embed_model", "modelo-de-prueba")
    monkeypatch.setattr(settings, "embed_device", "cpu")
    visto = {}

    class Contador:
        def __init__(self, nombre, device=None):
            visto.update(nombre=nombre, device=device)

    monkeypatch.setattr(E, "SentenceTransformer", Contador)
    E.get_model()
    assert visto == {"nombre": "modelo-de-prueba", "device": "cpu"}
    limpiar_cache()
