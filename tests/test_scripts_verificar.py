"""Tests de `scripts/verificar.py`: los umbrales que detectan los defectos.

Este script es el único control de calidad del corpus, y los tres defectos que
ya encontró (32 PDFs que extrajeron cero, 62 transcripts vacíos, 11 PDFs
indexados como tiras de glifos) eran todos silenciosos: ninguna etapa del
pipeline había fallado. Un umbral que se desajusta no rompe nada — deja de
encontrar, y el informe sale diciendo que todo está bien.

Lo que se fija, umbral por umbral y con el caso real que lo calibró:

- **Densidad de habla.** El piso estaba en 60 chars/minuto, siete veces por
  debajo del percentil 1 del corpus (467.9 sobre 4097 notas de voz), así que un
  transcript al que el VAD le había comido el 70% del habla pasaba limpio. Hoy
  está en 250.0 y los tests marcan los dos lados del borde.
- **Proporción de espacios.** pypdf devuelve la cantidad correcta de caracteres
  cuando el PDF implementa los espacios como posicionamiento, así que el texto
  pegado pasa cualquier control de volumen. Lo único que lo distingue de prosa
  es la proporción de espacios.
- **Lo vacío no es todo igual.** De los 62 transcripts vacíos, 51 son silencio
  digital y no hay nada que recuperar. Contarlos todos como defecto hace que
  nadie mire la lista.
- **Sin VAD se lista pero no se cuenta.** Es texto a leer, no un hueco. Y un
  transcript viejo, hecho antes de que se anotara el flag, no es un transcript
  sin VAD: `vad=None` no es `vad=False`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _cargar():
    spec = importlib.util.spec_from_file_location(
        "verificar", REPO / "scripts" / "verificar.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _cargar()


def t(**kw) -> dict:
    """Una fila como la que arma `cargar()`, con los campos neutros puestos."""
    base = {
        "archivo": "x.json", "stem": "x", "tipo": "wa_audio",
        "chars": 1000, "espacios": 0.19, "cabecera": "hola",
        "dur": 60.0, "pages": 0, "engine": None, "vad": True,
        "fuente": None, "ilegible": None,
    }
    base.update(kw)
    return base


# --- los umbrales calibrados ------------------------------------------------


def test_un_transcript_normal_no_es_sospechoso():
    assert V.revisar_transcripts([t()], medir_audio=False) == 0


def test_densidad_apenas_debajo_del_piso_se_marca():
    """249 chars en 60 s. El piso es 250.0."""
    assert V.revisar_transcripts([t(chars=249, dur=60.0)], medir_audio=False) == 1


def test_densidad_apenas_arriba_del_piso_no_se_marca():
    assert V.revisar_transcripts([t(chars=251, dur=60.0)], medir_audio=False) == 0


def test_un_audio_corto_no_se_juzga_por_densidad():
    """Debajo de 30 s la densidad no dice nada: un "si" de dos segundos son
    dos caracteres y eso no es una pérdida."""
    assert V.revisar_transcripts([t(chars=2, dur=5.0)], medir_audio=False) == 0


def test_el_piso_de_densidad_quedo_arriba_del_que_dejaba_pasar_perdidas():
    """El valor viejo era 60.0 y por eso no encontraba nada. Si alguien lo baja
    de nuevo a ese orden, este test lo dice."""
    assert V.MIN_CHARS_POR_MINUTO >= 250.0


# --- PDFs con capa de texto pobre -------------------------------------------


def test_un_pdf_con_menos_de_400_chars_por_pagina_se_marca():
    fila = t(tipo="pdf", pages=10, chars=3999, dur=0.0)
    assert V.revisar_transcripts([fila], medir_audio=False) == 1


def test_un_pdf_con_capa_de_texto_sana_no_se_marca():
    fila = t(tipo="pdf", pages=10, chars=4001, dur=0.0)
    assert V.revisar_transcripts([fila], medir_audio=False) == 0


def test_un_pdf_sin_cantidad_de_paginas_no_se_juzga():
    """Sin el total de páginas no hay denominador: marcar acá sería inventar
    (gotcha #28, la fracción solo significa algo si las dos partes se midieron
    sobre el mismo texto)."""
    fila = t(tipo="pdf", pages=0, chars=5, dur=0.0)
    assert V.revisar_transcripts([fila], medir_audio=False) == 0


def test_un_audio_corto_de_texto_no_se_juzga_con_la_regla_de_pdf():
    fila = t(tipo="wa_audio", pages=10, chars=5, dur=0.0)
    assert V.revisar_transcripts([fila], medir_audio=False) == 0


# --- texto pegado, sin espacios ---------------------------------------------


def test_texto_sin_espacios_se_marca():
    """Los nueve PDFs roto daban de 0.0000 a 0.0507 de espacios. La mediana del
    corpus es 0.19."""
    fila = t(chars=5000, espacios=0.02, dur=0.0)
    assert V.revisar_transcripts([fila], medir_audio=False) == 1


def test_prosa_normal_no_se_marca_por_espacios():
    fila = t(chars=5000, espacios=0.19, dur=0.0)
    assert V.revisar_transcripts([fila], medir_audio=False) == 0


def test_el_minimo_real_del_corpus_sigue_pasando():
    """0.1081 es el OCR de capturas de chat: el texto legítimo con menos
    espacios de todo el corpus. Si el umbral sube por encima, el informe empieza
    a marcar material sano y la lista deja de servir."""
    fila = t(chars=5000, espacios=0.1081, dur=0.0)
    assert V.revisar_transcripts([fila], medir_audio=False) == 0


def test_un_texto_corto_sin_espacios_no_se_marca():
    """Un transcript de diez caracteres puede no tener ninguno legítimamente."""
    fila = t(chars=299, espacios=0.0, dur=0.0)
    assert V.revisar_transcripts([fila], medir_audio=False) == 0


# --- transcripts vacíos: tres destinos distintos ----------------------------


def test_un_vacio_con_fuente_en_silencio_no_es_un_defecto(monkeypatch, tmp_path):
    """51 de los 62 vacíos son esto: silencio digital. No hay habla perdida."""
    monkeypatch.setattr(V, "volumen_medio", lambda _p: -91.0)
    fila = t(chars=0, dur=120.0, fuente=tmp_path / "mudo.mp4")
    assert V.revisar_transcripts([fila], medir_audio=True) == 0


def test_un_vacio_con_audio_audible_si_es_un_defecto(monkeypatch, tmp_path):
    """Audio que se oye y cero segmentos transcriptos: ahí se perdió habla."""
    monkeypatch.setattr(V, "volumen_medio", lambda _p: -18.0)
    fila = t(chars=0, dur=120.0, fuente=tmp_path / "con_voz.mp4")
    assert V.revisar_transcripts([fila], medir_audio=True) == 1


def test_el_borde_del_silencio_es_el_umbral_exacto(monkeypatch, tmp_path):
    """-60.0 dB cuenta como silencio; apenas por encima, como audible."""
    fila = t(chars=0, dur=120.0, fuente=tmp_path / "a.mp4")
    monkeypatch.setattr(V, "volumen_medio", lambda _p: V.SILENCIO_DB)
    assert V.revisar_transcripts([fila], medir_audio=True) == 0
    monkeypatch.setattr(V, "volumen_medio", lambda _p: V.SILENCIO_DB + 0.1)
    assert V.revisar_transcripts([fila], medir_audio=True) == 1


def test_sin_medir_audio_no_se_llama_a_ffmpeg(monkeypatch, tmp_path):
    """La pasada rápida existe porque medir 62 fuentes con ffmpeg lleva minutos.
    Tiene que no medir nada, no medir en silencio."""
    llamadas = []
    monkeypatch.setattr(V, "volumen_medio", lambda p: llamadas.append(p) or -18.0)
    fila = t(chars=0, dur=120.0, fuente=tmp_path / "a.mp4")
    assert V.revisar_transcripts([fila], medir_audio=False) == 0
    assert llamadas == []


def test_un_vacio_sin_fuente_no_se_puede_juzgar(monkeypatch):
    """Si el archivo original no está (junction caída, material movido), no hay
    nada que medir. Queda listado como "sin medir" y no se cuenta: no se sabe."""
    monkeypatch.setattr(V, "volumen_medio", lambda _p: -18.0)
    fila = t(chars=0, dur=120.0, fuente=None)
    assert V.revisar_transcripts([fila], medir_audio=True) == 0


# --- sin VAD: se lista, no se cuenta ----------------------------------------


def test_un_transcript_sin_vad_se_lista_pero_no_se_cuenta(capsys):
    """El que lo corrió sabe lo que hizo; el que venga después no tiene otra
    forma de enterarse. Pero no es un hueco, así que no infla el contador."""
    assert V.revisar_transcripts([t(vad=False)], medir_audio=False) == 0
    assert "SIN VAD" in capsys.readouterr().out


def test_un_transcript_viejo_no_es_un_transcript_sin_vad(capsys):
    """`vad=None` son los transcripts hechos antes de que el flag se anotara.
    Tratarlos como sin VAD mandaría a leer 8800 transcripts sanos."""
    assert V.revisar_transcripts([t(vad=None)], medir_audio=False) == 0
    assert "SIN VAD" not in capsys.readouterr().out


# --- un JSON ilegible no puede voltear el verificador -----------------------


def test_un_json_ilegible_cuenta_y_no_rompe_el_resto():
    """La fila de un transcript ilegible no tiene ni `chars` ni `dur`: si
    alguna de las reglas la mirara sin la guarda, el verificador moriría con
    KeyError antes de revisar los otros 8837."""
    roto = {"archivo": "roto.json", "stem": "roto", "ilegible": "Expecting value"}
    assert V.revisar_transcripts([roto, t(chars=249, dur=60.0)], medir_audio=False) == 2


# --- cargar(): cómo se arma cada fila ---------------------------------------


@pytest.fixture
def ts_dir(tmp_path, monkeypatch):
    d = tmp_path / "transcripts"
    d.mkdir()
    monkeypatch.setattr(V, "TRANSCRIPTS", d)
    return d


def escribir(d: Path, nombre: str, data: dict) -> None:
    (d / nombre).write_text(json.dumps(data), encoding="utf-8")


def test_cargar_mide_la_proporcion_de_espacios(ts_dir):
    escribir(ts_dir, "a.json", {"full_text": "ab cd ef", "source_type": "wa_audio"})
    fila = V.cargar()[0]
    assert fila["chars"] == 8
    assert fila["espacios"] == pytest.approx(2 / 8)


def test_cargar_no_divide_por_cero_con_un_transcript_vacio(ts_dir):
    """62 transcripts del corpus tienen `full_text` vacío. Una división acá
    voltea el verificador entero."""
    escribir(ts_dir, "a.json", {"full_text": "", "source_type": "wa_audio"})
    fila = V.cargar()[0]
    assert fila["chars"] == 0
    assert fila["espacios"] == 0.0


def test_cargar_corta_la_cabecera_en_el_largo_del_criterio(ts_dir):
    escribir(ts_dir, "a.json", {"full_text": "x" * 500})
    assert len(V.cargar()[0]["cabecera"]) == V.CABECERA_DUPLICADO


def test_cargar_marca_ilegible_y_sigue(ts_dir):
    (ts_dir / "roto.json").write_text("{no es json", encoding="utf-8")
    escribir(ts_dir, "sano.json", {"full_text": "hola"})
    filas = {f["stem"]: f for f in V.cargar()}
    assert filas["roto"]["ilegible"]
    assert filas["sano"]["ilegible"] is None


def test_cargar_resuelve_la_fuente_absoluta_si_existe(ts_dir, tmp_path):
    fuente = tmp_path / "video.mp4"
    fuente.write_bytes(b"\x00")
    escribir(ts_dir, "a.json", {"full_text": "hola", "source_path": str(fuente)})
    assert V.cargar()[0]["fuente"] == fuente


def test_cargar_deja_la_fuente_en_none_si_el_archivo_no_esta(ts_dir, tmp_path):
    """Con las junctions de `data/raw` apuntando a un disco desconectado, todas
    las fuentes desaparecen. Tiene que quedar en None, no en una ruta que no
    existe, porque después se le pasa a ffmpeg."""
    escribir(ts_dir, "a.json",
             {"full_text": "hola", "source_path": str(tmp_path / "no_esta.mp4")})
    assert V.cargar()[0]["fuente"] is None


def test_cargar_lee_la_duracion_y_el_vad_del_metadata(ts_dir):
    escribir(ts_dir, "a.json", {
        "full_text": "hola",
        "metadata": {"duration_seconds": 42.5, "vad_filter": False, "engine": "x"},
    })
    fila = V.cargar()[0]
    assert fila["dur"] == 42.5
    assert fila["vad"] is False
    assert fila["engine"] == "x"


def test_cargar_sin_metadata_no_rompe(ts_dir):
    escribir(ts_dir, "a.json", {"full_text": "hola"})
    fila = V.cargar()[0]
    assert fila["dur"] == 0.0
    assert fila["vad"] is None
    assert fila["pages"] == 0
