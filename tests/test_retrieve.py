"""Tests del retrieval: el umbral, el formato del prompt y qué pasa si falla.

Esta es la etapa que el usuario toca en cada pregunta, y la que decide con qué
material contesta el modelo. Tres propiedades, las tres con consecuencias
visibles en la respuesta:

1. **El umbral de score existe y se pasa a Qdrant.** `query_points` devuelve
   top-N siempre, aunque nada se parezca a la pregunta; y el system prompt le
   dice al modelo que el material recuperado le gana a su criterio. Sin
   `score_threshold`, una pregunta sobre cualquier cosa se contesta con los 10
   principios menos lejanos y el modelo los trata como doctrina.
2. **Si no se recuperó nada, el prompt tiene que decirlo.** Un bloque vacío se
   lee como "no hay restricciones"; `(ninguno recuperado)` se lee como lo que
   es. Es la diferencia entre el modelo diciendo que no sabe y el modelo
   inventando con tono de coach.
3. **Una colección que falla no se lleva las otras dos.** El `except` por
   colección es a propósito: con el índice a medio reconstruir, media respuesta
   es mejor que un stacktrace en la cara del usuario.

No se abre Qdrant ni se carga bge-m3: el cliente y el embedder son dobles.
"""

from __future__ import annotations

import pytest

from coach_brain.kb import retrieve as R


class FakePoint:
    def __init__(self, payload):
        self.payload = payload


class FakeResp:
    def __init__(self, points):
        self.points = points


class FakeQdrant:
    """Devuelve un punto por colección y anota con qué parámetros lo pidieron."""

    def __init__(self, falla=(), payloads=None):
        self.falla = set(falla)
        self.payloads = payloads or {}
        self.llamadas: list[dict] = []

    def query_points(self, collection_name, query, limit, score_threshold):
        self.llamadas.append({"col": collection_name, "limit": limit,
                              "score_threshold": score_threshold,
                              "dim": len(query)})
        if collection_name in self.falla:
            raise RuntimeError("coleccion no existe")
        payload = self.payloads.get(collection_name, {"texto": "algo"})
        return FakeResp([FakePoint(payload)])


@pytest.fixture
def cliente(monkeypatch):
    c = FakeQdrant()
    monkeypatch.setattr(R, "get_qdrant", lambda: c)
    monkeypatch.setattr(R, "embed_one", lambda q: [0.1] * 8)
    return c


# --- 1) el umbral -----------------------------------------------------------


def test_el_umbral_viaja_a_qdrant(cliente):
    """Si el umbral no llega a la query, no existe: el filtrado no se hace en
    ningún otro lado."""
    R.retrieve("como arranco una charla")
    assert {c["score_threshold"] for c in cliente.llamadas} == {R.MIN_SCORE}


def test_el_umbral_se_puede_bajar_por_llamada(cliente):
    R.retrieve("algo", min_score=0.0)
    assert {c["score_threshold"] for c in cliente.llamadas} == {0.0}


def test_cada_coleccion_tiene_su_propio_tope(cliente):
    R.retrieve("algo", top_principles=3, top_situations=2, top_style=1)
    limites = {c["col"]: c["limit"] for c in cliente.llamadas}
    assert limites == {R.COL_PRINCIPLES: 3, R.COL_SITUATIONS: 2, R.COL_STYLE: 1}


def test_se_busca_en_las_tres_colecciones(cliente):
    R.retrieve("algo")
    assert [c["col"] for c in cliente.llamadas] == [
        R.COL_PRINCIPLES, R.COL_SITUATIONS, R.COL_STYLE]


def test_el_vector_de_la_query_es_el_que_se_busca(cliente):
    """Una query sin embeddear devolvería lo mismo para cualquier pregunta."""
    R.retrieve("algo")
    assert {c["dim"] for c in cliente.llamadas} == {8}


# --- 2) el prompt nunca queda con un bloque vacío ---------------------------


def test_sin_resultados_el_prompt_lo_dice():
    """Un bloque vacío se lee como 'no hay restricciones'."""
    d = R.RetrievalResult([], [], []).format_for_prompt()
    assert d["principles_retrieved"] == "(ninguno recuperado)"
    assert d["situations_retrieved"] == "(ninguna recuperada)"
    assert d["style_retrieved"] == "(ninguna recuperada)"


def test_la_situacion_entra_entera_al_prompt():
    """Las cinco partes de una situación son el material que el modelo tiene
    que respetar. Si una se cae del formato, el modelo la improvisa."""
    s = {"descripcion": "te deja en visto", "diagnostico": "bajo interes",
         "respuesta_recomendada": "no insistas", "respuesta_evitar": "doble mensaje",
         "tono": "seco", "riesgo": "alto"}
    texto = R.RetrievalResult([], [s], []).format_for_prompt()["situations_retrieved"]
    for v in s.values():
        assert v in texto


def test_un_payload_incompleto_no_rompe_el_formato():
    """Los payloads viejos del índice no tienen todos los campos. Formatear
    tiene que seguir devolviendo string, no explotar a mitad de una respuesta."""
    d = R.RetrievalResult([{}], [{}], [{}]).format_for_prompt()
    assert all(isinstance(v, str) and v for v in d.values())


def test_los_principios_salen_como_lista():
    d = R.RetrievalResult([{"texto": "no persigas"}, {"texto": "mantene el marco"}],
                          [], []).format_for_prompt()
    assert d["principles_retrieved"] == "- no persigas\n- mantene el marco"


# --- 3) una colección que falla no se lleva las otras -----------------------


def test_una_coleccion_caida_devuelve_vacio_y_no_corta(monkeypatch):
    c = FakeQdrant(falla=[R.COL_SITUATIONS])
    monkeypatch.setattr(R, "get_qdrant", lambda: c)
    monkeypatch.setattr(R, "embed_one", lambda q: [0.1] * 8)

    r = R.retrieve("algo")
    assert r.situations == []
    assert len(r.principles) == 1 and len(r.style) == 1


def test_con_las_tres_caidas_el_prompt_sigue_siendo_valido(monkeypatch):
    """El caso del índice recién borrado por `--reset`: la app tiene que
    contestar 'no tengo material', no tirar un error."""
    c = FakeQdrant(falla=[R.COL_PRINCIPLES, R.COL_SITUATIONS, R.COL_STYLE])
    monkeypatch.setattr(R, "get_qdrant", lambda: c)
    monkeypatch.setattr(R, "embed_one", lambda q: [0.1] * 8)

    d = R.retrieve("algo").format_for_prompt()
    assert d["principles_retrieved"] == "(ninguno recuperado)"
