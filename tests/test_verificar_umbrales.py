"""Tests de los umbrales del verificador.

Los umbrales de `scripts/verificar.py` son lo unico que separa un corpus sano
de un hueco silencioso, y ya pasaron dos veces por el mismo problema: estaban
puestos a ojo, demasiado flojos, y dejaban pasar perdidas grandes sin una sola
linea de error. Estos tests no comprueban logica: fijan los valores calibrados
contra la distribucion medida, para que bajarlos sea una decision explicita y
no un descuido.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _cargar_verificar():
    """Importa scripts/verificar.py, que no es parte del paquete instalable."""
    spec = importlib.util.spec_from_file_location(
        "verificar_script_umbrales", REPO / "scripts" / "verificar.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


verificar = _cargar_verificar()

# Percentiles medidos el 2026-11-17 sobre los 4097 wa_audio de 30 s o mas.
P1_MEDIDO = 467.9
MEDIANA_MEDIDA = 916.9


def test_el_piso_de_densidad_queda_debajo_del_percentil_1():
    """El piso tiene que estar debajo de p1 o marca transcripts sanos.

    Si sube por encima de 467,9 el verificador empieza a marcar la cola
    legitima: grabaciones de calle con minutos de ruido ambiente, que tienen
    densidad baja y estan completas.
    """
    assert verificar.MIN_CHARS_POR_MINUTO < P1_MEDIDO


def test_el_piso_de_densidad_no_es_simbolico():
    """Y tiene que estar lo bastante arriba para detectar una perdida grande.

    Estuvo en 60,0: siete veces por debajo de p1. Un transcript al que el VAD
    le habia comido el 70% del habla pasaba limpio. El piso de referencia es
    una cuarta parte de la mediana del corpus.
    """
    assert verificar.MIN_CHARS_POR_MINUTO >= MEDIANA_MEDIDA / 4


def test_hay_duracion_minima_para_juzgar_la_densidad():
    """Sin piso de duracion, una nota de voz de 3 segundos siempre es `pobre`.

    La densidad no se puede medir en fuentes cortas: un "dale, nos vemos" de
    dos segundos da una cifra sin sentido.
    """
    assert verificar.DUR_MINIMA_PARA_JUZGAR >= 30.0


def test_el_umbral_de_espacios_queda_debajo_del_minimo_real():
    """MIN_SPACE_RATIO tiene que estar debajo del minimo real del corpus.

    El minimo legitimo medido es 0,1081 (OCR de capturas de chat) y los 11 PDF
    roturados estaban entre 0,0000 y 0,0716. El umbral vive entre esos dos
    valores (gotcha #24), y esta duplicado a proposito en `ingest/pdfs.py`.
    """
    assert 0.0716 < verificar.MIN_SPACE_RATIO < 0.1081


def test_el_umbral_de_silencio_no_confunde_habla_con_silencio():
    """-91 dB es silencio digital puro; el habla real queda bien arriba de -60."""
    assert -91.0 < verificar.SILENCIO_DB <= -40.0
