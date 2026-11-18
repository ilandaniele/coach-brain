"""Tests de `scripts/escanear_corpus.py`: el barrido de alucinaciones del corpus vivo.

Que el detector de alucinaciones exista para la auditoría sin VAD no significa
que el corpus con VAD esté limpio. El transcript **con VAD** de
`wa_audio_00021896` son 1907 caracteres de "¡Gracias por ver el video!"
repetido 25 veces, y lo encontró una persona leyendo un log, no el pipeline.
Este script hace esa búsqueda sobre los 8838 transcripts.

Lo que se fija:

- **El barrido es de solo lectura.** Es el paso 6 de `finalize.ps1` y corre
  sobre el corpus vivo: si escribiera, cambiaría lo que devuelve el índice sin
  que nadie lo haya decidido. El test comprueba que ningún archivo cambie.
- **Un transcript vacío no es una alucinación.** Son el gotcha #21 y tienen su
  propio informe en `verificar.py`. Contarlos acá duplicaría 62 defectos que ya
  están contados en otro lado.
- **Un transcript ilegible es problema del verificador, no de acá.** Se saltea
  sin contarse como leído: el denominador tiene que ser lo que de verdad se
  pudo juzgar (gotcha #28).
- **`None` no es `0`.** Un marcado con 0 items extraídos no aportó ni un vector
  al índice y no hay nada que sacar; uno sin extracción todavía es una pregunta
  abierta. Si los dos se devolvieran como 0, el informe diría que no hay nada
  que decidir cuando en realidad no se sabe.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _cargar():
    spec = importlib.util.spec_from_file_location(
        "escanear_corpus", REPO / "scripts" / "escanear_corpus.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _cargar()

# El caso real: 1907 caracteres de una frase que Whisper arrastra de los
# subtitulos de YouTube con los que se entreno.
ALUCINADO = "¡Gracias por ver el video! " * 25


def poner(d: Path, nombre: str, texto: str) -> Path:
    p = d / f"{nombre}.json"
    p.write_text(json.dumps({"source_id": nombre, "full_text": texto}),
                 encoding="utf-8")
    return p


@pytest.fixture
def ts(tmp_path):
    d = tmp_path / "transcripts"
    d.mkdir()
    return d


# --- qué marca y qué no -----------------------------------------------------


def test_marca_el_caso_real(ts):
    poner(ts, "wa_audio_00021896", ALUCINADO)
    leidos, marcados, _ = E.escanear(ts)
    assert leidos == 1
    assert len(marcados) == 1
    stem, chars, senal = marcados[0]
    assert stem == "wa_audio_00021896"
    assert chars == len(ALUCINADO.strip())
    assert senal


def test_el_habla_real_no_se_marca(ts):
    poner(ts, "sano", "Hoy vamos a ver como se arranca una conversacion sin que "
                      "parezca un interrogatorio. La idea es simple: preguntar "
                      "menos y contar mas, porque contar invita a contar.")
    leidos, marcados, _ = E.escanear(ts)
    assert leidos == 1
    assert marcados == []


def test_un_transcript_vacio_no_es_una_alucinacion(ts):
    """Gotcha #21: los 62 vacíos son un defecto distinto, con su propio informe
    en `verificar.py`, donde además se mide el volumen de la fuente para saber
    si se perdió habla o si el archivo estaba en silencio."""
    poner(ts, "vacio", "")
    leidos, marcados, avisados = E.escanear(ts)
    assert leidos == 1
    assert marcados == []
    assert avisados == []


def test_un_transcript_de_solo_espacios_tampoco(ts):
    poner(ts, "blancos", "   \n\t  ")
    assert E.escanear(ts)[1] == []


def test_un_json_ilegible_no_cuenta_como_leido(ts):
    """El denominador del informe tiene que ser lo que se pudo juzgar. Sumar un
    archivo que no se pudo abrir diría que se revisó algo que nadie miró."""
    (ts / "roto.json").write_text("{no es json", encoding="utf-8")
    poner(ts, "sano", "Una charla normal, con frases distintas una de otra.")
    leidos, marcados, _ = E.escanear(ts)
    assert leidos == 1
    assert marcados == []


def test_solo_mira_los_json(ts):
    poner(ts, "a", ALUCINADO)
    (ts / "notas.txt").write_text(ALUCINADO, encoding="utf-8")
    assert E.escanear(ts)[0] == 1


def test_un_marcado_no_se_avisa_dos_veces(ts):
    """El que ya está marcado no vuelve a aparecer en la lista de avisos: sería
    el mismo transcript contado en dos informes."""
    poner(ts, "a", ALUCINADO)
    _, marcados, avisados = E.escanear(ts)
    assert len(marcados) == 1
    assert [st for st, _, _ in avisados] == []


def test_el_barrido_no_escribe_nada(ts):
    """Es el paso 6 de `finalize.ps1` y corre sobre el corpus vivo. Sacar o
    re-transcribir un transcript cambia lo que devuelve el índice, y eso lo
    decide una persona que leyó el texto."""
    p1 = poner(ts, "a", ALUCINADO)
    p2 = poner(ts, "b", "Charla normal con oraciones distintas entre si, nada raro.")
    antes = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (p1, p2)}
    E.escanear(ts)
    assert sorted(x.name for x in ts.iterdir()) == ["a.json", "b.json"]
    for p, (datos, mtime) in antes.items():
        assert p.read_bytes() == datos
        assert p.stat().st_mtime_ns == mtime


def test_el_orden_del_informe_es_estable(ts):
    """El informe se compara entre corridas: si el orden dependiera del disco,
    dos barridos iguales darían diffs distintos."""
    for n in ("zzz", "aaa", "mmm"):
        poner(ts, n, ALUCINADO)
    _, marcados, _ = E.escanear(ts)
    assert [st for st, _, _ in marcados] == ["aaa", "mmm", "zzz"]


# --- items extraídos: la diferencia entre el corpus y el índice -------------


@pytest.fixture
def ex(tmp_path):
    d = tmp_path / "extracted"
    d.mkdir()
    return d


def extraccion(d: Path, stem: str, **kw) -> None:
    data = {"principios": [], "situaciones": [], "frases": []}
    data.update(kw)
    (d / f"{stem}.extracted.json").write_text(json.dumps(data), encoding="utf-8")


def test_sin_extraccion_devuelve_none_no_cero(ex):
    """None significa "la extracción todavía no pasó por ahí", que no es lo
    mismo que "no aportó nada al índice". Si fuera 0, el informe diría que no
    hay nada que decidir cuando en realidad no se sabe."""
    assert E.items_extraidos("no_existe", extracted=ex) is None


def test_una_extraccion_vacia_devuelve_cero(ex):
    """0 items es el dato que importa: ese transcript no aportó ni un vector al
    índice, así que la alucinación no llegó a las colecciones. Los 4 marcados
    de la última corrida dieron 0, incluido el de 1907 caracteres."""
    extraccion(ex, "filtrado")
    assert E.items_extraidos("filtrado", extracted=ex) == 0


def test_cuenta_los_items_de_las_tres_colecciones(ex):
    """Las tres colecciones del índice se arman de estas tres claves, así que
    las tres cuentan: mirar solo `principios` daría 0 para un transcript que
    aportó frases."""
    extraccion(ex, "a", principios=[{}, {}], situaciones=[{}], frases=[{}, {}, {}])
    assert E.items_extraidos("a", extracted=ex) == 6


def test_una_clave_en_null_no_rompe_la_cuenta(ex):
    """El extractor escribe `null` cuando el modelo devolvió la clave vacía."""
    extraccion(ex, "a", principios=None, frases=[{}])
    assert E.items_extraidos("a", extracted=ex) == 1


def test_una_extraccion_ilegible_se_informa_como_desconocida(ex):
    """Un JSON truncado no es una extracción con 0 items: no se sabe qué tiene.
    Decir 0 sería afirmar que no llegó nada al índice sin haberlo verificado."""
    (ex / "a.extracted.json").write_text("{truncado", encoding="utf-8")
    assert E.items_extraidos("a", extracted=ex) is None


@pytest.mark.parametrize("n,esperado", [(None, "sin extraer"), (0, "0 items"),
                                        (6, "6 items")])
def test_la_etiqueta_distingue_el_cero_de_lo_no_medido(n, esperado):
    assert E._etiqueta_items(n) == esperado
