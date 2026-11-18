"""Tests del escaneo del corpus vivo.

Lo que hay que fijar no es la detección — eso ya tiene 44 tests en la auditoría
sin VAD — sino tres propiedades del recorrido:

1. **No escribe nada.** Es un script que mira el corpus indexado, así que un
   write accidental cambiaría lo que devuelve el índice.
2. Un transcript vacío no es una alucinación: es el gotcha #21, y marcar los 62
   vacíos como basura sería ruido que tapa los 4 que importan.
3. Un JSON ilegible no corta la corrida. Es un problema distinto, del
   verificador, y abortar acá dejaría el corpus sin escanear por un archivo.
4. "Sin extraer" no es "0 items". El índice se arma desde `extracted/`, así que
   0 items significa que el extractor filtró ese transcript y que no aportó un
   vector; que no haya extracción significa que todavía no se sabe. Tratarlos
   igual haría que el reporte diera por limpio algo que nadie miró.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _cargar():
    spec = importlib.util.spec_from_file_location(
        "escanear_corpus", REPO / "scripts" / "escanear_corpus.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _cargar()


def _escribir(d: Path, stem: str, texto) -> Path:
    p = d / (stem + ".json")
    p.write_text(json.dumps({"full_text": texto}, ensure_ascii=False),
                 encoding="utf-8")
    return p


def test_marca_la_alucinacion_y_deja_el_habla(tmp_path):
    _escribir(tmp_path, "basura", "Gracias.")
    _escribir(tmp_path, "habla", "Hola, ¿cómo va? Te quería contar una cosa "
                                 "que me pasó ayer en el trabajo.")
    leidos, marcados, avisados = M.escanear(tmp_path)
    assert leidos == 2
    assert [s for s, _, _ in marcados] == ["basura"]
    assert avisados == []


def test_el_transcript_vacio_no_es_alucinacion(tmp_path):
    """Gotcha #21: 62 transcripts vacíos, 51 de ellos silencio real."""
    _escribir(tmp_path, "vacio", "")
    _escribir(tmp_path, "espacios", "   \n  ")
    leidos, marcados, avisados = M.escanear(tmp_path)
    assert leidos == 2, "se leyeron, aunque no se juzguen"
    assert marcados == []
    assert avisados == []


def test_un_json_roto_no_corta_la_corrida(tmp_path):
    (tmp_path / "roto.json").write_text("{esto no es json", encoding="utf-8")
    _escribir(tmp_path, "basura", "Gracias.")
    leidos, marcados, _ = M.escanear(tmp_path)
    assert leidos == 1, "el roto no cuenta como leido"
    assert len(marcados) == 1, "pero el siguiente si se escanea"


def test_no_escribe_nada(tmp_path):
    """La propiedad que hace que este script sea seguro de correr cuando sea."""
    p = _escribir(tmp_path, "basura", "Gracias.")
    antes = (p.read_bytes(), p.stat().st_mtime_ns)
    listado_antes = sorted(x.name for x in tmp_path.iterdir())

    M.escanear(tmp_path)

    assert p.read_bytes() == antes[0]
    assert p.stat().st_mtime_ns == antes[1]
    assert sorted(x.name for x in tmp_path.iterdir()) == listado_antes


def test_el_criterio_es_el_mismo_que_la_auditoria():
    """Si se separan, el escaneo del corpus y la auditoría contestan distinto
    sobre el mismo texto, y los números del informe dejan de compararse."""
    spec = importlib.util.spec_from_file_location(
        "nv_ref", REPO / "scripts" / "retranscribir_sin_vad.py")
    nv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nv)
    assert M._novad.parece_alucinacion.__doc__ == nv.parece_alucinacion.__doc__
    assert M._novad.RESTO_MINIMO == nv.RESTO_MINIMO
    assert M._novad.AVISO_COBERTURA == nv.AVISO_COBERTURA


def test_directorio_vacio_no_falla(tmp_path):
    assert M.escanear(tmp_path) == (0, [], [])


# --- items extraidos: la diferencia entre basura en el corpus y en el índice --


def _extraccion(d: Path, stem: str, **items) -> Path:
    p = d / (stem + ".extracted.json")
    cuerpo = {"principios": [], "situaciones": [], "frases": []}
    cuerpo.update(items)
    p.write_text(json.dumps(cuerpo, ensure_ascii=False), encoding="utf-8")
    return p


def test_sin_extraccion_no_es_cero(tmp_path):
    """None y 0 contestan cosas distintas.

    0 es "el extractor lo miró y no sacó nada": no aportó un solo vector, no
    hay nada que sacar del índice. None es "la etapa todavía no pasó por ahí":
    la respuesta no se sabe. Confundirlos haría que un transcript sin extraer
    pase por filtrado.
    """
    assert M.items_extraidos("no_existe", tmp_path) is None
    _extraccion(tmp_path, "vacia")
    assert M.items_extraidos("vacia", tmp_path) == 0


def test_cuenta_los_items_de_las_tres_colecciones(tmp_path):
    _extraccion(tmp_path, "algo", principios=["a", "b"], frases=["c"])
    assert M.items_extraidos("algo", tmp_path) == 3


def test_una_extraccion_rota_no_se_cuenta_como_cero(tmp_path):
    """Un JSON ilegible es desconocido, no filtrado: si se contara como 0, el
    reporte diría que no llegó al índice sin haberlo podido leer."""
    (tmp_path / "roto.extracted.json").write_text("{", encoding="utf-8")
    assert M.items_extraidos("roto", tmp_path) is None


def test_la_etiqueta_distingue_los_tres_casos():
    assert M._etiqueta_items(None) == "sin extraer"
    assert M._etiqueta_items(0) == "0 items"
    assert M._etiqueta_items(3) == "3 items"
