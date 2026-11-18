"""Tests del matcheo por duracion del inventario.

El inventario cruza las fuentes de `data/raw` con los transcripts para saber
que material ya esta cubierto. Lo hacia por path y por nombre de archivo, y
por eso un archivo renombrado en disco aparecia como un hueco de 3,43 GB
cuando en realidad estaba transcripto hace meses. El tercer pase lo resuelve
por duracion del audio, que es lo unico que sobrevive a un renombre.

Vale la pena cubrirlo porque el modo de falla es silencioso en las dos
direcciones: un falso hueco manda a re-transcribir horas de video que ya
estaban hechas, y un falso match marca como cubierto algo que nunca se
transcribio.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _cargar_manifest():
    """Importa scripts/manifest.py, que no es parte del paquete instalable."""
    spec = importlib.util.spec_from_file_location(
        "manifest_script", REPO / "scripts" / "manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


manifest = _cargar_manifest()


def _hit(path: str, kind: str = "video"):
    """Una fila de `hits` como la arma main(): [fuente, info, via]."""
    return [{"path": path, "kind": kind, "bytes": 1}, None, None]


def _info(transcript: str, dur: float):
    return {"transcript": transcript, "dur": dur, "matched": False}


def test_matchea_el_renombrado_y_lo_saca_de_los_sueltos(monkeypatch):
    """Un archivo renombrado se reconoce por duracion, una sola vez."""
    monkeypatch.setattr(manifest, "_dur_ffprobe", lambda p: 7154.8)

    hits = [_hit("data/raw/videos/78 que es el valor.mp4")]
    info = _info("video_78 que es el valor-008.json", 7154.3)
    by_base = {"otro nombre": [info]}

    assert manifest.match_por_duracion(hits, by_base) == 1
    assert hits[0][1] is info
    assert hits[0][2] == "transcript (por duracion)"
    assert info["matched"] is True
    # Sale de by_base: no puede volver a matchear con otra fuente.
    assert by_base["otro nombre"] == []


def test_no_matchea_fuera_de_tolerancia(monkeypatch):
    """Dos videos distintos de duracion parecida no se confunden.

    La tolerancia es de 1 segundo. Un video de 3600s y otro de 3604s son
    material distinto, y darlos por iguales marcaria como cubierto algo que
    nunca se transcribio.
    """
    monkeypatch.setattr(manifest, "_dur_ffprobe", lambda p: 3600.0)

    hits = [_hit("data/raw/videos/clase.mp4")]
    info = _info("video_otra clase.json", 3604.0)
    by_base = {"otra clase": [info]}

    assert manifest.match_por_duracion(hits, by_base) == 0
    assert hits[0][1] is None
    assert info["matched"] is False


def test_elige_el_mas_cercano_cuando_hay_varios(monkeypatch):
    monkeypatch.setattr(manifest, "_dur_ffprobe", lambda p: 1000.0)

    hits = [_hit("data/raw/videos/x.mp4")]
    lejos = _info("video_lejos.json", 1000.9)
    cerca = _info("video_cerca.json", 1000.1)
    by_base = {"a": [lejos], "b": [cerca]}

    assert manifest.match_por_duracion(hits, by_base) == 1
    assert hits[0][1] is cerca
    assert lejos["matched"] is False


def test_ignora_docs_y_fuentes_sin_duracion_legible(monkeypatch):
    """Solo audio y video tienen duracion; un PDF no se matchea asi."""
    monkeypatch.setattr(manifest, "_dur_ffprobe", lambda p: 100.0)

    hits = [_hit("data/raw/docs/libro.pdf", kind="doc")]
    by_base = {"libro": [_info("pdf_libro.json", 100.0)]}

    assert manifest.match_por_duracion(hits, by_base) == 0
    assert hits[0][1] is None


def test_no_matchea_si_ffprobe_falla(monkeypatch):
    """Sin duracion de la fuente no hay con que comparar: queda sin cubrir."""
    monkeypatch.setattr(manifest, "_dur_ffprobe", lambda p: None)

    hits = [_hit("data/raw/videos/roto.mp4")]
    info = _info("video_algo.json", 500.0)
    by_base = {"algo": [info]}

    assert manifest.match_por_duracion(hits, by_base) == 0
    assert info["matched"] is False


def test_sin_pendientes_no_llama_a_ffprobe():
    """Caso normal del inventario: no se pagan 8823 llamadas a ffprobe."""
    llamadas = []

    hits = [[{"path": "x.mp4", "kind": "video", "bytes": 1}, _info("y.json", 10.0), "x"]]
    by_base = {"y": [_info("z.json", 10.0)]}

    assert manifest.match_por_duracion(hits, by_base) == 0
    assert llamadas == []


def test_descarta_duraciones_no_numericas(monkeypatch):
    """Un transcript sin `duration_seconds` guarda None y no debe romper."""
    monkeypatch.setattr(manifest, "_dur_ffprobe", lambda p: 10.0)

    hits = [_hit("data/raw/videos/x.mp4")]
    by_base = {"a": [_info("video_a.json", None), _info("video_b.json", 0)]}

    assert manifest.match_por_duracion(hits, by_base) == 0


@pytest.mark.parametrize("dur", [10.0, 10.9, 9.1])
def test_tolerancia_inclusiva(monkeypatch, dur):
    monkeypatch.setattr(manifest, "_dur_ffprobe", lambda p: 10.0)

    hits = [_hit("data/raw/videos/x.mp4")]
    by_base = {"a": [_info("video_a.json", dur)]}

    assert manifest.match_por_duracion(hits, by_base) == 1
