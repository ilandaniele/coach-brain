"""Tests de la clasificacion de los archivos sin cubrir.

`INVENTARIO.md` es el registro sobre el que se decide si se borra el material
crudo, asi que el motivo de cada archivo sin cubrir no es cosmetico: dice si
borrarlo pierde algo. Un falso "se pierde" hace desconfiar de todo el registro;
un falso "no se pierde" borra contenido sin forma de recuperarlo.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _cargar_manifest():
    """Importa scripts/manifest.py, que no es parte del paquete instalable."""
    spec = importlib.util.spec_from_file_location(
        "manifest_script_motivos", REPO / "scripts" / "manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


manifest = _cargar_manifest()


def _row(path: str, tipo: str = "video", cubierto: str = "NO"):
    return {"path": path, "tipo": tipo, "cubierto": cubierto, "bytes": 1}


def _ffprobe(stdout: str = "", returncode: int = 0, excepcion=None):
    def fake(cmd, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        if excepcion is not None:
            raise excepcion
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    return fake


def test_video_sin_pista_de_audio_no_es_hueco(monkeypatch):
    monkeypatch.setattr(manifest, "_stems_fuera_del_corpus", lambda: set())
    monkeypatch.setattr(manifest.subprocess, "run", _ffprobe(stdout="\n"))
    rows = [_row("data/raw/whatsapp/00033588-GIF.mp4")]
    assert manifest.clasificar_sin_cubrir(rows) == 0
    assert rows[0]["motivo"].startswith("sin pista de audio")


def test_contenedor_ilegible_si_es_hueco(monkeypatch):
    """ffprobe que no puede abrir el archivo: es la perdida real.

    Es el caso de `4 Hablarle a una chica del trabajo, sexualiza.mp4`, que no
    tiene atomo `moov`. Tiene que contar como hueco: es lo unico que distingue
    "no hay nada que transcribir" de "hay contenido y no lo podemos leer".
    """
    monkeypatch.setattr(manifest, "_stems_fuera_del_corpus", lambda: set())
    monkeypatch.setattr(manifest.subprocess, "run", _ffprobe(returncode=1))
    rows = [_row("data/raw/docs/truncado.mp4")]
    assert manifest.clasificar_sin_cubrir(rows) == 1
    assert "ilegible" in rows[0]["motivo"]


def test_ffprobe_ausente_cuenta_como_hueco(monkeypatch):
    """Sin ffprobe no se puede afirmar que no haya nada que perder.

    La duda tiene que resolverse del lado conservador: contarlo como hueco
    deja el archivo en la lista de lo que no se borra.
    """
    monkeypatch.setattr(manifest, "_stems_fuera_del_corpus", lambda: set())
    monkeypatch.setattr(
        manifest.subprocess, "run", _ffprobe(excepcion=OSError("no ffprobe"))
    )
    rows = [_row("data/raw/videos/clase.mp4")]
    assert manifest.clasificar_sin_cubrir(rows) == 1
    # Y no dice "ilegible": eso seria afirmar una perdida que no se comprobo.
    assert "sin verificar" in rows[0]["motivo"]
    assert "ilegible" not in rows[0]["motivo"]


def test_excluido_a_proposito_no_es_hueco(monkeypatch):
    monkeypatch.setattr(
        manifest, "_stems_fuera_del_corpus", lambda: {"00021049-cv_divertido"}
    )
    rows = [_row("data/raw/whatsapp/00021049-CV_divertido.pdf", tipo="doc")]
    assert manifest.clasificar_sin_cubrir(rows) == 0
    assert "_fuera_del_corpus" in rows[0]["motivo"]


def test_exclusion_gana_sobre_el_ffprobe(monkeypatch):
    """Un video excluido a proposito no se manda a ffprobe."""
    monkeypatch.setattr(manifest, "_stems_fuera_del_corpus", lambda: {"clip"})

    def no_llamar(*a, **kw):  # noqa: ANN002, ANN003
        raise AssertionError("no deberia consultar ffprobe")

    monkeypatch.setattr(manifest.subprocess, "run", no_llamar)
    rows = [_row("data/raw/whatsapp/clip.mp4")]
    assert manifest.clasificar_sin_cubrir(rows) == 0


def test_video_con_pista_de_audio_sin_transcript_es_hueco(monkeypatch):
    """Video que SI tiene audio y no tiene transcript: falta procesarlo.

    Es el hueco que mas importa no perder de vista, porque es el unico de los
    cuatro motivos que se arregla corriendo el pipeline otra vez. Si cayera en
    la rama de "sin pista de audio" el inventario diria que no hay nada que
    perder, y se borraria un video con habla adentro.
    """
    monkeypatch.setattr(manifest, "_stems_fuera_del_corpus", lambda: set())
    monkeypatch.setattr(manifest.subprocess, "run", _ffprobe(stdout="0\n"))
    rows = [_row("data/raw/videos/clase-nueva.mp4")]
    assert manifest.clasificar_sin_cubrir(rows) == 1
    assert rows[0]["motivo"] == "HUECO: falta procesarlo"


def test_doc_sin_transcript_es_hueco(monkeypatch):
    """Un PDF sin transcript y sin excluir es un hueco: falta procesarlo."""
    monkeypatch.setattr(manifest, "_stems_fuera_del_corpus", lambda: set())
    rows = [_row("data/raw/pdfs/libro.pdf", tipo="doc")]
    assert manifest.clasificar_sin_cubrir(rows) == 1
    assert rows[0]["motivo"].startswith("HUECO")


def test_las_filas_cubiertas_quedan_sin_motivo(monkeypatch):
    monkeypatch.setattr(manifest, "_stems_fuera_del_corpus", lambda: set())
    rows = [_row("data/raw/videos/ok.mp4", cubierto="si")]
    assert manifest.clasificar_sin_cubrir(rows) == 0
    assert rows[0]["motivo"] == ""


def test_stems_fuera_del_corpus_saca_el_prefijo_de_tipo(monkeypatch, tmp_path):
    """El transcript se llama `<tipo>_<stem>`; la fuente en disco, `<stem>`."""
    monkeypatch.setattr(manifest, "FUERA", tmp_path)
    (tmp_path / "pdf_00021049-CV_divertido_Ilan.json").write_text("{}", encoding="utf-8")
    (tmp_path / "wa_audio_00099999-AUDIO.json").write_text("{}", encoding="utf-8")
    stems = manifest._stems_fuera_del_corpus()
    assert "00021049-cv_divertido_ilan" in stems
    assert "00099999-audio" in stems


def test_stems_fuera_del_corpus_sin_carpeta(monkeypatch, tmp_path):
    monkeypatch.setattr(manifest, "FUERA", tmp_path / "no-existe")
    assert manifest._stems_fuera_del_corpus() == set()


def test_el_original_roto_no_es_hueco_si_hay_una_copia_recuperada(monkeypatch):
    """El caso de `untrunc`: el original ilegible queda sin cubrir, pero su
    contenido esta transcripto bajo el nombre de la copia reparada. Marcarlo
    HUECO seria un falso "se pierde" sobre el registro que decide el borrado."""
    monkeypatch.setattr(manifest, "_stems_fuera_del_corpus", lambda: set())
    monkeypatch.setattr(manifest.subprocess, "run", _ffprobe(returncode=1))
    rows = [
        _row("data/raw/docs/Bonus/4 Hablarle, sexualiza.mp4"),
        _row("data/raw/docs/Bonus/4 Hablarle, sexualiza REPARADO.mp4", cubierto="SI"),
    ]
    assert manifest.clasificar_sin_cubrir(rows) == 0
    assert "recuperado como" in rows[0]["motivo"]
    assert "reparado" in rows[0]["motivo"]
    assert "HUECO" not in rows[0]["motivo"]


def test_un_hermano_sin_cubrir_no_salva_al_original_roto(monkeypatch):
    """La copia tiene que estar CUBIERTA. Dos archivos rotos con nombres
    parecidos siguen siendo una perdida, y el inventario tiene que decirlo."""
    monkeypatch.setattr(manifest, "_stems_fuera_del_corpus", lambda: set())
    monkeypatch.setattr(manifest.subprocess, "run", _ffprobe(returncode=1))
    rows = [
        _row("data/raw/docs/clase.mp4"),
        _row("data/raw/docs/clase REPARADO.mp4"),
    ]
    assert manifest.clasificar_sin_cubrir(rows) == 2
    assert all("HUECO" in r["motivo"] for r in rows)


def test_un_hermano_cubierto_que_no_es_prefijo_no_salva_a_nadie(monkeypatch):
    """La regla es "mismo nombre mas un sufijo", no "esta en la misma carpeta".
    Si no, cualquier video cubierto taparia el hueco del de al lado."""
    monkeypatch.setattr(manifest, "_stems_fuera_del_corpus", lambda: set())
    monkeypatch.setattr(manifest.subprocess, "run", _ffprobe(returncode=1))
    rows = [
        _row("data/raw/docs/clase 7.mp4"),
        _row("data/raw/docs/clase 8.mp4", cubierto="SI"),
    ]
    assert manifest.clasificar_sin_cubrir(rows) == 1
    assert "HUECO" in rows[0]["motivo"]
