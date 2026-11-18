"""Tests de `scripts/retranscribir_sin_vad.py`: a quién manda a re-transcribir.

Lo que estos tests protegen es el caso negativo. Re-transcribir sin filtro de
voz un archivo que no tiene habla no devuelve texto perdido: devuelve
alucinación, porque Whisper sin VAD completa sobre ruido con frases de su
corpus de entrenamiento ("La ciudad de Bogotá" nueve veces, medido). Si eso
entra al corpus es peor que el hueco que venía a tapar.

Así que el criterio tiene que negarse en los tres casos en los que negarse es
lo correcto: silencio digital, volumen que no se pudo medir, y texto que en
realidad alcanza para la duración que tiene.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parent.parent / "scripts" / "retranscribir_sin_vad.py"


def _cargar():
    spec = importlib.util.spec_from_file_location("retranscribir_sin_vad", RUTA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _cargar()


# --- la rama de los vacios: la decide el volumen ----------------------------

def test_vacio_con_audio_audible_entra():
    """0 caracteres sobre audio que se oye es exactamente una falla del VAD."""
    assert M.clasificar(0, 146.0, -19.0) == "vacio con -19.0 dB de audio"


def test_vacio_con_silencio_digital_no_entra():
    """-91 dB es un video muteado. No hay nada que recuperar ahi."""
    assert M.clasificar(0, 300.0, -91.0) is None


def test_vacio_justo_en_el_umbral_de_silencio_no_entra():
    """El umbral es inclusivo: -60.0 dB cuenta como silencio."""
    assert M.clasificar(0, 300.0, M.SILENCIO_DB) is None


def test_vacio_sin_volumen_medido_no_entra():
    """Sin medicion no se puede distinguir muteado de falla del filtro.

    Preferir no tocarlo: el costo de equivocarse es meter alucinacion al corpus.
    """
    assert M.clasificar(0, 300.0, None) is None


def test_vacio_corto_con_audio_entra_igual():
    """Para un vacio la duracion no importa, solo que haya sonido.

    El piso de 30 segundos existe para que la densidad tenga sentido
    estadistico; un transcript vacio no tiene densidad que juzgar.
    """
    assert M.clasificar(0, 4.0, -38.1) is not None


# --- la rama de la densidad -------------------------------------------------

def test_densidad_baja_entra():
    """256 caracteres sobre 964 segundos: 15.9 chars/minuto, el peor medido."""
    motivo = M.clasificar(256, 964.0, None)
    assert motivo == "15.9 chars/minuto"


def test_densidad_suficiente_no_entra():
    """Mil caracteres por minuto es una nota de voz normal."""
    assert M.clasificar(1000, 60.0, None) is None


def test_densidad_justo_en_el_piso_no_entra():
    """El piso es inclusivo: estar exactamente en el umbral no es sospechoso."""
    chars = int(M.MIN_CHARS_POR_MINUTO)  # 250 chars en 60 segundos
    assert M.clasificar(chars, 60.0, None) is None


def test_audio_corto_no_se_juzga_por_densidad():
    """Debajo de 30 segundos la densidad es ruido estadistico.

    Un "si" de 2 segundos da 60 chars/minuto y no tiene nada de malo.
    """
    assert M.clasificar(2, 2.0, None) is None


def test_con_texto_no_mira_el_volumen():
    """Si hay texto, la decision es la densidad; el volumen no entra.

    Medir volumen cuesta una pasada de ffmpeg por archivo, y para un transcript
    con texto no aporta nada.
    """
    assert M.clasificar(256, 964.0, -3.0) == M.clasificar(256, 964.0, None)


# --- duracion invalida ------------------------------------------------------

@pytest.mark.parametrize("dur", [0.0, -1.0])
def test_sin_duracion_no_entra(dur):
    """Sin duracion no hay densidad ni certeza de que sea audio."""
    assert M.clasificar(0, dur, -20.0) is None
    assert M.clasificar(100, dur, None) is None


# --- los umbrales vienen del verificador, no copiados -----------------------

def test_umbrales_vienen_de_verificar():
    """Una sola definicion de "sospechoso" en todo el proyecto.

    Si alguien recalibra el piso en `verificar.py`, esta herramienta tiene que
    moverse con el. Tener el numero escrito dos veces termina en dos
    herramientas que contestan distinto sobre el mismo archivo.
    """
    spec = importlib.util.spec_from_file_location(
        "verificar_para_test", RUTA.parent / "verificar.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    assert M.MIN_CHARS_POR_MINUTO == v.MIN_CHARS_POR_MINUTO
    assert M.DUR_MINIMA_PARA_JUZGAR == v.DUR_MINIMA_PARA_JUZGAR
    assert M.SILENCIO_DB == v.SILENCIO_DB
