"""Tests del chequeo de contenido duplicado del verificador.

Un duplicado no es un hueco: es lo contrario, sobra texto. Pero sesga la
busqueda igual, porque el mismo pasaje ocupa dos lugares entre los resultados
mas parecidos, y el indice lo arrastra en cada reconstruccion.

El riesgo de este chequeo no es dejar pasar un duplicado, es inventarlo: si
marca como repetido algo que no lo es, alguien puede borrar material unico. Por
eso los tests cargan el lado restrictivo, y uno fija que el chequeo no sume al
conteo de sospechosos, que cuenta perdidas.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _cargar_verificar():
    """Importa scripts/verificar.py, que no es parte del paquete instalable."""
    spec = importlib.util.spec_from_file_location(
        "verificar_script_duplicados", REPO / "scripts" / "verificar.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


verificar = _cargar_verificar()


def _t(stem: str, chars: int, dur: float, cabecera: str, fuente=True) -> dict:
    """Un transcript con lo justo que mira `revisar_duplicados`."""
    return {
        "stem": stem,
        "archivo": stem + ".json",
        "chars": chars,
        "dur": dur,
        "cabecera": cabecera,
        "fuente": Path("C:/m/x") if fuente else None,
        "ilegible": None,
    }


CAB_A = "a" * 200
CAB_B = "b" * 200


def test_detecta_el_mismo_video_transcripto_dos_veces():
    """Misma duracion al microsegundo, largos distintos: dos pasadas de Whisper
    sobre el mismo archivo."""
    ts = [
        _t("video_36 Clase vip-029", 83248, 5948.7201875, CAB_A, fuente=False),
        _t("video_36 Clase vip-095", 84606, 5948.7201875, CAB_A),
    ]
    assert verificar.revisar_duplicados(ts) == 1


def test_detecta_el_mismo_pdf_con_dos_nombres():
    """Sin duracion, pero el texto extraido sale identico caracter por caracter."""
    ts = [
        _t("pdf_Conversador Irresistible", 275074, 0.0, CAB_A),
        _t("pdf_Evan Cid - Conversador irresistible", 275074, 0.0, CAB_A),
    ]
    assert verificar.revisar_duplicados(ts) == 1


def test_no_marca_duplicado_por_arranque_igual_solo():
    """Dos notas de voz distintas pueden empezar con la misma muletilla.

    Sin coincidencia de largo ni de duracion no hay duplicado, y este es el caso
    que mas importa: declararlo mal puede hacer que alguien borre audio unico.
    """
    ts = [
        _t("wa_video_00038562", 2001, 350.0, CAB_A),
        _t("wa_video_00038566", 1870, 412.0, CAB_A),
    ]
    assert verificar.revisar_duplicados(ts) == 0


def test_no_marca_duplicado_con_arranques_distintos():
    ts = [
        _t("uno", 5000, 100.0, CAB_A),
        _t("dos", 5000, 100.0, CAB_B),
    ]
    assert verificar.revisar_duplicados(ts) == 0


def test_ignora_textos_mas_cortos_que_la_cabecera():
    """Una cabecera incompleta no se compara: "dale, nos vemos" repetido en dos
    notas de voz distintas no es contenido duplicado."""
    corto = "dale, nos vemos"
    ts = [
        _t("uno", len(corto), 3.0, corto),
        _t("dos", len(corto), 3.0, corto),
    ]
    assert verificar.revisar_duplicados(ts) == 0


def test_ignora_los_ilegibles():
    ts = [
        {"stem": "roto", "ilegible": "JSONDecodeError"},
        _t("uno", 5000, 100.0, CAB_A),
    ]
    assert verificar.revisar_duplicados(ts) == 0


def test_la_cabecera_es_lo_bastante_larga_para_no_dar_falsos_positivos():
    """200 caracteres son varias frases: dos fuentes distintas no coinciden ahi.

    Si baja a unas pocas decenas, los saludos y las muletillas de apertura
    empiezan a producir grupos que no son duplicados.
    """
    assert verificar.CABECERA_DUPLICADO >= 100


def test_la_tolerancia_de_duracion_es_estricta():
    """La duracion del contenedor sale del mismo archivo, asi que coincide al
    microsegundo. Aflojarla a decimas junta notas de voz sin relacion."""
    assert verificar.TOLERANCIA_DUPLICADO <= 0.1
