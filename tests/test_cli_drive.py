"""El CLI y la descarga de Drive.

El CLI es la puerta de entrada de todas las etapas: si un comando cambia de
nombre o deja de pasar un flag, los scripts de `scripts/` y lo escrito en el
README dejan de funcionar sin que falle nada.

Ninguna etapa real corre acá: cada `run` se reemplaza por un espía, y `gdown`
también. Los nombres de comando se verifican porque typer convierte los
guiones bajos en guiones medios (`ingest_pdfs` se invoca `ingest-pdfs`).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from coach_brain import cli as C
from coach_brain.ingest import docs as mod_docs
from coach_brain.ingest import drive as D
from coach_brain.ingest import pdfs as mod_pdfs
from coach_brain.ingest import transcribe as mod_transcribe
from coach_brain.ingest import whatsapp as mod_whatsapp
from coach_brain.kb import extract as mod_extract
from coach_brain.kb import index as mod_index
from coach_brain.settings import settings

runner = CliRunner()


def espia(monkeypatch, modulo, nombre="run"):
    """Reemplaza `modulo.<nombre>` y devuelve la lista de llamadas."""
    llamadas: list[dict] = []
    monkeypatch.setattr(modulo, nombre, lambda **kw: llamadas.append(kw))
    return llamadas


# --- los nombres de los comandos ------------------------------------------

def test_sin_argumentos_muestra_la_ayuda():
    """`no_args_is_help=True`: invocar `coach` sin nada no es un error silencioso."""
    r = runner.invoke(C.app, [])
    assert "coach-brain CLI" in r.output


@pytest.mark.parametrize("comando", [
    "ingest-pdfs", "ingest-docs", "ingest-whatsapp", "transcribe",
    "download-drive", "extract", "index", "app-run",
])
def test_los_ocho_comandos_estan_en_la_ayuda(comando):
    r = runner.invoke(C.app, ["--help"])
    assert comando in r.output


def test_un_comando_que_no_existe_falla():
    assert runner.invoke(C.app, ["inventado"]).exit_code != 0


# --- cada comando llama a su etapa ----------------------------------------

def test_ingest_pdfs_llama_a_pdfs(monkeypatch):
    llamadas = espia(monkeypatch, mod_pdfs)
    assert runner.invoke(C.app, ["ingest-pdfs"]).exit_code == 0
    assert llamadas == [{}]


def test_ingest_docs_llama_a_docs(monkeypatch):
    llamadas = espia(monkeypatch, mod_docs)
    assert runner.invoke(C.app, ["ingest-docs"]).exit_code == 0
    assert llamadas == [{}]


def test_transcribe_llama_a_transcribe(monkeypatch):
    llamadas = espia(monkeypatch, mod_transcribe)
    assert runner.invoke(C.app, ["transcribe"]).exit_code == 0
    assert llamadas == [{}]


def test_extract_llama_a_extract(monkeypatch):
    llamadas = espia(monkeypatch, mod_extract)
    assert runner.invoke(C.app, ["extract"]).exit_code == 0
    assert llamadas == [{}]


def test_index_llama_a_index(monkeypatch):
    llamadas = espia(monkeypatch, mod_index)
    assert runner.invoke(C.app, ["index"]).exit_code == 0
    assert llamadas == [{}]


def test_whatsapp_sin_force_no_rehace_nada(monkeypatch):
    llamadas = espia(monkeypatch, mod_whatsapp)
    assert runner.invoke(C.app, ["ingest-whatsapp"]).exit_code == 0
    assert llamadas == [{"force": False}]


def test_whatsapp_con_force_rehace(monkeypatch):
    """`--force` es lo único que vuelve a escribir un transcript ya hecho."""
    llamadas = espia(monkeypatch, mod_whatsapp)
    assert runner.invoke(C.app, ["ingest-whatsapp", "--force"]).exit_code == 0
    assert llamadas == [{"force": True}]


def test_download_drive_pasa_la_url(monkeypatch):
    llamadas: list[str] = []
    monkeypatch.setattr(D, "download", lambda url: llamadas.append(url))
    r = runner.invoke(C.app, ["download-drive", "https://drive.google.com/x"])
    assert r.exit_code == 0
    assert llamadas == ["https://drive.google.com/x"]


def test_download_drive_sin_url_falla(monkeypatch):
    monkeypatch.setattr(D, "download", lambda url: None)
    assert runner.invoke(C.app, ["download-drive"]).exit_code != 0


# --- app-run ---------------------------------------------------------------

def test_app_run_lanza_streamlit_sobre_el_script_que_existe(monkeypatch):
    """Si el archivo se renombra, el comando falla recién al usarlo."""
    cmds: list[list[str]] = []
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda c, **kw: cmds.append(c))
    assert runner.invoke(C.app, ["app-run"]).exit_code == 0
    assert cmds[0][1:4] == ["-m", "streamlit", "run"]
    assert Path(cmds[0][4]).exists()
    assert Path(cmds[0][4]).name == "streamlit_app.py"


# ===========================================================================
# drive.py
# ===========================================================================

@pytest.mark.parametrize("nombre,esperado", [
    ("libro.pdf", "pdfs"),
    ("clase.mp4", "videos"),
    ("clase.mkv", "videos"),
    ("clase.mov", "videos"),
    ("clase.avi", "videos"),
    ("clase.webm", "videos"),
    ("clase.m4v", "videos"),
    ("nota.mp3", "audios"),
    ("nota.wav", "audios"),
    ("nota.m4a", "audios"),
    ("nota.aac", "audios"),
    ("nota.flac", "audios"),
    ("nota.ogg", "audios"),
    ("nota.opus", "audios"),
    ("captura.png", "screenshots"),
    ("captura.jpg", "screenshots"),
    ("captura.jpeg", "screenshots"),
    ("captura.webp", "screenshots"),
])
def test_cada_extension_va_a_su_carpeta(nombre, esperado):
    assert D.detect_subdir(nombre) == esperado


def test_la_extension_en_mayuscula_tambien_se_reconoce():
    assert D.detect_subdir("CLASE.MP4") == "videos"


@pytest.mark.parametrize("nombre", ["notas.docx", "libro.epub", "chat.txt", "sin_extension"])
def test_una_extension_que_no_esta_en_el_mapa_no_se_clasifica(nombre):
    """Devuelve None, y el archivo queda en `_drive_dump` en vez de perderse."""
    assert D.detect_subdir(nombre) is None


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    (tmp_path / "raw").mkdir(parents=True)
    return tmp_path / "raw"


def fake_gdown(monkeypatch, archivos):
    """Simula `gdown.download_folder`: crea los archivos y devuelve sus paths."""
    def _download_folder(url, output, quiet, use_cookies):
        creados = []
        for nombre in archivos:
            p = Path(output) / nombre
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("contenido", encoding="utf-8")
            creados.append(str(p))
        return creados
    monkeypatch.setattr(D.gdown, "download_folder", _download_folder)


def test_los_archivos_se_reparten_por_extension(raw, monkeypatch):
    fake_gdown(monkeypatch, ["libro.pdf", "clase.mp4", "nota.opus", "captura.png"])
    D.download("https://drive.google.com/x")
    assert (raw / "pdfs" / "libro.pdf").exists()
    assert (raw / "videos" / "clase.mp4").exists()
    assert (raw / "audios" / "nota.opus").exists()
    assert (raw / "screenshots" / "captura.png").exists()
    assert not (raw / "_drive_dump" / "libro.pdf").exists()  # se movió, no se copió


def test_un_archivo_desconocido_queda_en_el_dump(raw, monkeypatch):
    fake_gdown(monkeypatch, ["notas.docx"])
    D.download("https://drive.google.com/x")
    assert (raw / "_drive_dump" / "notas.docx").exists()


def test_un_archivo_que_ya_estaba_no_se_sobreescribe(raw, monkeypatch):
    """Lo ya bajado gana: una re-descarga no pisa un archivo del corpus."""
    (raw / "pdfs").mkdir(parents=True)
    (raw / "pdfs" / "libro.pdf").write_text("el original", encoding="utf-8")
    fake_gdown(monkeypatch, ["libro.pdf"])
    D.download("https://drive.google.com/x")
    assert (raw / "pdfs" / "libro.pdf").read_text(encoding="utf-8") == "el original"
    assert (raw / "_drive_dump" / "libro.pdf").exists()  # la copia nueva queda aparte


def test_una_carpeta_devuelta_por_gdown_se_ignora(raw, monkeypatch):
    def _download_folder(url, output, quiet, use_cookies):
        sub = Path(output) / "subcarpeta"
        sub.mkdir(parents=True, exist_ok=True)
        return [str(sub)]
    monkeypatch.setattr(D.gdown, "download_folder", _download_folder)
    D.download("https://drive.google.com/x")  # no levanta


def test_sin_archivos_avisa_y_no_crea_carpetas(raw, monkeypatch, capsys):
    monkeypatch.setattr(D.gdown, "download_folder", lambda **kw: [])
    D.download("https://drive.google.com/x")
    assert "pública" in capsys.readouterr().out
    assert not (raw / "pdfs").exists()


def test_el_dump_se_crea_antes_de_bajar(raw, monkeypatch):
    visto: dict[str, bool] = {}

    def _download_folder(url, output, quiet, use_cookies):
        visto["existia"] = Path(output).exists()
        return []

    monkeypatch.setattr(D.gdown, "download_folder", _download_folder)
    D.download("https://drive.google.com/x")
    assert visto["existia"] is True
