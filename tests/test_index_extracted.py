"""Tests del indexador, con un Qdrant y un embedder de juguete.

El indexado no tenía pruebas, y es el único lugar donde una decisión silenciosa
cambia lo que la app contesta. Las tres propiedades que se fijan acá:

1. **Una extracción con 0 items no aporta un solo vector.** Es la frase que
   `scripts/escanear_corpus.py` y STATUS.md usan para distinguir "hay basura en
   el corpus" de "hay basura en el índice". Si el indexador llegara a escribir
   un punto por una lista vacía, esa distinción dejaría de ser cierta y el
   informe del paso 6 pasaría a mentir.
2. **El `source_id` del payload sale del nombre del archivo**, sacándole el
   `.extracted`. Es la clave con la que la app cita de dónde viene cada
   respuesta: si queda `algo.extracted`, las citas apuntan a un archivo que no
   existe.
3. **Un extracto roto no corta el indexado.** Son 8.800 archivos; abortar por
   uno dejaría las 3 colecciones a medio construir justo después del `--reset`,
   que es el momento en que la app no tiene nada que servir.

No se toca Qdrant ni la GPU: el cliente es un doble que anota los upserts y
`embed` devuelve vectores fijos.
"""

from __future__ import annotations

import json

import pytest

from coach_brain.kb import index as I
from coach_brain.settings import settings

DIM = 4


class FakeQdrant:
    """Anota lo que se le pide. Lo único que importa es qué puntos entraron."""

    def __init__(self, existentes=()):
        self.colecciones = set(existentes)
        self.creadas: list[str] = []
        self.borradas: list[str] = []
        self.upserts: list[tuple[str, list]] = []

    def get_collection(self, name):
        if name not in self.colecciones:
            raise KeyError(name)
        return {"name": name}

    def create_collection(self, collection_name, vectors_config):
        self.colecciones.add(collection_name)
        self.creadas.append(collection_name)

    def delete_collection(self, name):
        self.colecciones.discard(name)
        self.borradas.append(name)

    def upsert(self, collection_name, points):
        self.upserts.append((collection_name, points))

    # --- ayudas de lectura para los tests ---
    def puntos(self, coleccion):
        return [p for c, ps in self.upserts if c == coleccion for p in ps]


@pytest.fixture(autouse=True)
def embedder_falso(monkeypatch):
    monkeypatch.setattr(I, "embed", lambda textos: [[0.1] * DIM for _ in textos])
    monkeypatch.setattr(I, "embed_dim", lambda: DIM)


def _extracto(d, stem, **items):
    cuerpo = {"principios": [], "situaciones": [], "frases": []}
    cuerpo.update(items)
    p = d / (stem + ".extracted.json")
    p.write_text(json.dumps(cuerpo, ensure_ascii=False), encoding="utf-8")
    return p


# --- 1) cero items, cero vectores -------------------------------------------


def test_una_extraccion_vacia_no_escribe_ni_un_punto(tmp_path):
    """La propiedad de la que depende todo el informe del corpus vivo."""
    c = FakeQdrant()
    counts = I.index_extracted(c, _extracto(tmp_path, "vacia"), {"source_id": "vacia"})
    assert c.upserts == []
    assert counts == {"principles": 0, "situations": 0, "style_phrases": 0}


def test_las_listas_en_null_cuentan_como_vacias(tmp_path):
    """El extractor escribe `null` cuando el modelo devolvió `null` en vez de
    `[]`. `data.get(k) or []` lo cubre; sin eso, esto explotaría en el `for`."""
    p = tmp_path / "nula.extracted.json"
    p.write_text(json.dumps({"principios": None, "situaciones": None, "frases": None}),
                 encoding="utf-8")
    c = FakeQdrant()
    assert I.index_extracted(c, p, {"source_id": "nula"}) == {
        "principles": 0, "situations": 0, "style_phrases": 0}
    assert c.upserts == []


# --- 2) un punto por item, en la colección que le toca ----------------------


def test_cada_item_va_a_su_coleccion(tmp_path):
    c = FakeQdrant()
    e = _extracto(
        tmp_path, "mix",
        principios=[{"id": "p1", "texto": "no persigas"},
                    {"id": "p2", "texto": "mantene el marco"}],
        situaciones=[{"id": "s1", "descripcion": "te deja en visto",
                      "diagnostico": "bajo interes"}],
        frases=[{"id": "f1", "texto": "me caes bien pero no te hagas ideas"}],
    )
    counts = I.index_extracted(c, e, {"source_id": "mix"})

    assert counts == {"principles": 2, "situations": 1, "style_phrases": 1}
    assert len(c.puntos(I.COL_PRINCIPLES)) == 2
    assert len(c.puntos(I.COL_SITUATIONS)) == 1
    assert len(c.puntos(I.COL_STYLE)) == 1


def test_el_payload_lleva_el_origen_y_el_tipo(tmp_path):
    """Sin `kind` la app no puede filtrar por colección cruzada, y sin el
    source_meta la respuesta no se puede citar."""
    c = FakeQdrant()
    e = _extracto(tmp_path, "uno", principios=[{"id": "p1", "texto": "algo",
                                                "tipo": "marco", "tags": ["x"]}])
    I.index_extracted(c, e, {"source_id": "uno", "title": "Clase 3"})

    pl = c.puntos(I.COL_PRINCIPLES)[0].payload
    assert pl["source_id"] == "uno"
    assert pl["title"] == "Clase 3"
    assert pl["kind"] == "principle"
    assert pl["principle_id"] == "p1"
    assert pl["texto"] == "algo"
    assert pl["tags"] == ["x"]


def test_la_situacion_se_embedea_con_descripcion_y_diagnostico(tmp_path, monkeypatch):
    """Lo que se busca de una situación no es su id: es el texto del problema.
    Si el texto embebido quedara vacío, la situación existiría en el índice y
    no la encontraría ninguna consulta."""
    vistos: list[list[str]] = []

    def espia(textos):
        vistos.append(list(textos))
        return [[0.1] * DIM for _ in textos]

    monkeypatch.setattr(I, "embed", espia)
    e = _extracto(tmp_path, "s", situaciones=[
        {"id": "s1", "descripcion": "te deja en visto", "diagnostico": "bajo interes"}])
    I.index_extracted(FakeQdrant(), e, {"source_id": "s"})
    assert vistos == [["te deja en visto — bajo interes"]]


def test_un_item_sin_texto_no_rompe_el_indexado(tmp_path):
    """El modelo a veces devuelve el objeto sin `texto`. Entra con string vacío
    (queda inútil para la búsqueda) pero no tira la corrida entera."""
    c = FakeQdrant()
    e = _extracto(tmp_path, "sin", principios=[{"id": "p1"}])
    assert I.index_extracted(c, e, {"source_id": "sin"})["principles"] == 1
    assert c.puntos(I.COL_PRINCIPLES)[0].payload["texto"] is None


# --- 3) la corrida completa -------------------------------------------------


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    edir = tmp_path / "processed" / "extracted"
    edir.mkdir(parents=True)
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    return edir


def test_el_source_id_sale_del_nombre_sin_extracted(entorno, monkeypatch):
    c = FakeQdrant()
    monkeypatch.setattr(I, "get_qdrant", lambda: c)
    _extracto(entorno, "wa_audio_0001", frases=[{"id": "f1", "texto": "hola"}])

    I.run()
    assert c.puntos(I.COL_STYLE)[0].payload["source_id"] == "wa_audio_0001"


def test_reset_borra_las_tres_colecciones_y_las_vuelve_a_crear(entorno, monkeypatch):
    """Gotcha #12: sin `--reset` el re-indexado DUPLICA, porque los ids son
    uuid4 nuevos en cada corrida y nada deduplica."""
    c = FakeQdrant(existentes=[I.COL_PRINCIPLES, I.COL_SITUATIONS, I.COL_STYLE])
    monkeypatch.setattr(I, "get_qdrant", lambda: c)
    _extracto(entorno, "uno", principios=[{"id": "p1", "texto": "algo"}])

    I.run(reset=True)
    assert set(c.borradas) == {I.COL_PRINCIPLES, I.COL_SITUATIONS, I.COL_STYLE}
    assert set(c.creadas) == {I.COL_PRINCIPLES, I.COL_SITUATIONS, I.COL_STYLE}


def test_sin_reset_no_borra_nada(entorno, monkeypatch):
    c = FakeQdrant(existentes=[I.COL_PRINCIPLES, I.COL_SITUATIONS, I.COL_STYLE])
    monkeypatch.setattr(I, "get_qdrant", lambda: c)
    _extracto(entorno, "uno", principios=[{"id": "p1", "texto": "algo"}])

    I.run()
    assert c.borradas == []
    assert c.creadas == [], "ya existian"


def test_un_extracto_ilegible_no_corta_la_corrida(entorno, monkeypatch):
    c = FakeQdrant()
    monkeypatch.setattr(I, "get_qdrant", lambda: c)
    (entorno / "a_roto.extracted.json").write_text("{", encoding="utf-8")
    _extracto(entorno, "b_sano", frases=[{"id": "f1", "texto": "hola"}])

    I.run()
    assert len(c.puntos(I.COL_STYLE)) == 1, "el sano se indexo igual"


def test_sin_extractos_no_toca_qdrant(entorno, monkeypatch):
    """Importa por el orden del paso 7: si la carpeta está vacía, abrir Qdrant
    y resetear dejaría las colecciones borradas y sin nada adentro."""
    def explota():
        raise AssertionError("no tiene que abrir Qdrant")

    monkeypatch.setattr(I, "get_qdrant", explota)
    I.run(reset=True)


# --- 4) el --reset que no reseteaba (gotcha #36) ----------------------------


def _col(tmp_path, nombre):
    d = tmp_path / "qdrant_storage" / "collection" / nombre
    d.mkdir(parents=True)
    (d / "storage.sqlite").write_text("viejo", encoding="utf-8")
    return d


def test_el_reset_borra_los_directorios_del_disco_antes_de_abrir_el_cliente(
    entorno, tmp_path, monkeypatch
):
    """Gotcha #36. `QdrantLocal.delete_collection` borra con
    `shutil.rmtree(..., ignore_errors=True)`, y en Windows el sqlite de la
    colección sigue abierto: el borrado falla en silencio y el re-indexado
    queda SUMADO encima del índice viejo. El reset tiene que borrar el
    directorio antes de construir el cliente, que es cuando nadie lo tiene
    abierto.

    Se fija también el orden: si el cliente se abriera primero, el directorio
    ya estaría tomado y volveríamos al bug.
    """
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "qdrant_storage"))
    dirs = [_col(tmp_path, n) for n in (I.COL_PRINCIPLES, I.COL_SITUATIONS, I.COL_STYLE)]
    _extracto(entorno, "uno", principios=[{"id": "p1", "texto": "algo"}])

    c = FakeQdrant(existentes=[I.COL_PRINCIPLES, I.COL_SITUATIONS, I.COL_STYLE])
    vivos_al_abrir = []

    def abrir():
        vivos_al_abrir.extend(d for d in dirs if d.exists())
        return c

    monkeypatch.setattr(I, "get_qdrant", abrir)
    I.run(reset=True)

    assert [d for d in dirs if d.exists()] == [], "quedo un directorio con datos viejos"
    assert vivos_al_abrir == [], "borro DESPUES de abrir el cliente, que es el bug"


def test_el_reset_no_toca_el_disco_si_qdrant_es_remoto(entorno, tmp_path, monkeypatch):
    """Con `QDRANT_URL` puesto el índice no está en este disco, y
    `delete_collection` contra el servidor sí funciona. Borrar acá sería borrar
    un directorio ajeno al índice que se está reseteando."""
    monkeypatch.setattr(settings, "qdrant_url", "http://qdrant.example:6333")
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "qdrant_storage"))
    d = _col(tmp_path, I.COL_PRINCIPLES)
    _extracto(entorno, "uno", principios=[{"id": "p1", "texto": "algo"}])

    monkeypatch.setattr(I, "get_qdrant", lambda: FakeQdrant())
    I.run(reset=True)
    assert d.exists()


def test_el_reset_no_aborta_si_el_directorio_no_existe(tmp_path, monkeypatch):
    """Primer índice de una máquina nueva: no hay nada que borrar."""
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "no_existe"))
    I._borrar_colecciones_del_disco()


def test_si_el_directorio_sobrevive_al_borrado_la_corrida_falla(tmp_path, monkeypatch):
    """La razón de ser del fix: un borrado que falla no puede pasar callado.
    Antes `ignore_errors=True` lo tapaba y el índice quedaba duplicado con el
    log informando los conteos nuevos como si todo hubiera salido bien."""
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "qdrant_storage"))
    _col(tmp_path, I.COL_PRINCIPLES)
    monkeypatch.setattr(I.shutil, "rmtree", lambda d: None)

    with pytest.raises(RuntimeError, match="No se pudo borrar"):
        I._borrar_colecciones_del_disco()
