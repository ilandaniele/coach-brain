"""Tests del orquestador `scripts/finalize.ps1`.

Es el único script del repo que borra algo: su paso 7 corre
`kb.index --reset`, que destruye las tres colecciones de Qdrant que está
sirviendo la app. Y es PowerShell, así que no se puede importar ni ejercitar
desde pytest: estos tests leen el texto del script y fijan las propiedades de
las que depende que no se pierda nada.

Son tests de texto a propósito. No prueban que el script funcione —para eso hay
que correrlo en Windows con la GPU libre—; prueban que nadie reordene los nueve
pasos ni afloje las guardas sin que falle algo. Las tres cosas que importan:

1. El orden. Los pasos 5 y 6 son informes que una persona tiene que leer para
   decidir qué sacar del corpus, y esa decisión se aplica a mano **antes** del
   paso 7, porque el índice se reconstruye desde los transcripts. Si el reset
   se adelanta, el informe llega tarde. Y el paso 8 regenera `INVENTARIO.md`,
   que es el registro de qué está cubierto: va después del índice que describe.
2. La confirmación. Sin `-Yes` el paso destructivo pregunta y acepta sólo "SI".
3. El abort. Una extracción son días de GPU: si hay un proceso del pipeline
   corriendo, el script sale con 1 en vez de matarlo para tomar el lock de
   Qdrant (gotcha #11, un solo proceso por base embebida).

Todas las aserciones de orden corren sobre el *cuerpo*, no sobre el archivo
entero: el comentario del encabezado nombra los mismos comandos que los pasos
(`(kb.index --reset)`, `(scripts\\manifest.py)`), así que buscarlos en el texto
completo encuentra primero el comentario y el test pasa o falla por el motivo
equivocado.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
RUTA = REPO / "scripts" / "finalize.ps1"
CORTE = "param([switch]$Yes)"


@pytest.fixture(scope="module")
def ps1() -> str:
    return RUTA.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cuerpo(ps1: str) -> str:
    """El script sin el encabezado de comentarios."""
    return ps1[ps1.index(CORTE) :]


@pytest.fixture(scope="module")
def pasos(cuerpo: str) -> list[int]:
    """Los números de los carteles `[n/9]`, en el orden en que aparecen."""
    return [int(n) for n in re.findall(r"\[(\d)/9\]", cuerpo)]


def test_los_nueve_pasos_estan_numerados_en_orden(pasos):
    assert pasos == list(range(1, 10))


def test_cada_paso_corre_el_comando_que_dice(cuerpo):
    """El orden de los comandos, no el de los carteles: un cartel puede mentir."""
    comandos = [
        "coach_brain.ingest.whatsapp_images_ocr aggregate",
        "-m coach_brain.kb.extract",
        "-m coach_brain.kb.extract",  # segunda pasada: gotcha #15
        r"scripts\verificar.py",
        r"scripts\retranscribir_sin_vad.py --juzgar",
        r"scripts\escanear_corpus.py",
        "coach_brain.kb.index --reset",
        r"scripts\manifest.py",
        "streamlit run",
    ]
    posiciones = []
    desde = 0
    for c in comandos:
        i = cuerpo.find(c, desde)
        assert i != -1, f"falta el comando: {c}"
        posiciones.append(i)
        desde = i + len(c)
    assert posiciones == sorted(posiciones)


def test_el_reset_va_despues_de_los_dos_informes_que_hay_que_leer(cuerpo):
    """Lo que se saque del corpus se decide leyendo los informes, y se aplica antes."""
    juzgar = cuerpo.index("retranscribir_sin_vad.py --juzgar")
    escanear = cuerpo.index("escanear_corpus.py")
    reset = cuerpo.index("coach_brain.kb.index --reset")
    assert juzgar < reset
    assert escanear < reset


def test_el_inventario_se_regenera_despues_del_indice(cuerpo):
    """`INVENTARIO.md` es el registro de qué está cubierto: describe el índice nuevo."""
    assert cuerpo.index("coach_brain.kb.index --reset") < cuerpo.index("manifest.py")


def test_la_segunda_pasada_de_extraccion_existe(cuerpo):
    """`kb.extract` globea una sola vez al arrancar (gotcha #15).

    El paso 1 escribe un transcript nuevo *durante* la corrida, así que la
    primera pasada no lo ve. Sin la segunda, las imágenes-chat no se extraen y
    el índice sale sin ellas sin que falle nada.
    """
    assert cuerpo.count("-m coach_brain.kb.extract") == 2


def test_el_paso_destructivo_pide_confirmacion_y_solo_acepta_si(cuerpo):
    bloque = cuerpo[cuerpo.index("[7/9]") : cuerpo.index("coach_brain.kb.index --reset")]
    assert "if (-not $Yes)" in bloque
    assert "Read-Host" in bloque
    assert '-ne "SI"' in bloque
    assert "exit 0" in bloque, "cancelar tiene que salir sin tocar el índice"


def test_la_confirmacion_nombra_que_borra_y_donde_esta_el_backup(cuerpo):
    bloque = cuerpo[cuerpo.index("[7/9]") : cuerpo.index("coach_brain.kb.index --reset")]
    assert "BORRA las 3 colecciones" in bloque
    assert "qdrant_storage.bak" in bloque


def test_el_switch_yes_existe_y_es_el_unico_parametro(ps1):
    assert CORTE in ps1


def test_aborta_si_hay_un_proceso_del_pipeline_corriendo(cuerpo):
    """Matar una extracción pierde días. El script sale con 1 y no toca nada."""
    bloque = cuerpo[cuerpo.index("$pipeline = @(") : cuerpo.index("$otros = @(")]
    for etapa in (
        r"kb\.extract",
        r"kb\.index",
        r"ingest\.",
        "retranscribir_sin_vad",
        "reparar_source_path",
    ):
        assert etapa in bloque, f"el abort no cubre {etapa}"
    assert "exit 1" in cuerpo[cuerpo.index("ABORTADO") :]


def test_solo_mata_streamlit_no_cualquier_python(cuerpo):
    """La regresión: el filtro era `$_.ProcessId -ne $PID`, y `$PID` es el de
    PowerShell, no el de ningún python. O sea que no filtraba nada y se llevaba
    cualquier python de la máquina, de este proyecto o de otro."""
    bloque = cuerpo[cuerpo.index("$otros = @(") : cuerpo.index("Stop-Process")]
    assert 'match "streamlit"' in bloque


def test_el_script_fuerza_utf8_antes_de_llamar_a_python(cuerpo):
    """El corpus está en castellano y el default de esta máquina no es UTF-8."""
    assert '$env:PYTHONIOENCODING = "utf-8"' in cuerpo
    assert '$env:PYTHONUTF8 = "1"' in cuerpo
    assert cuerpo.index("PYTHONUTF8") < cuerpo.index("[1/9]")


def test_usa_el_python_del_venv_y_no_el_del_path(cuerpo):
    assert r'$py = Join-Path $root ".venv\Scripts\python.exe"' in cuerpo
    # Ningún paso invoca `python` pelado: eso agarraría el del PATH, sin torch cu130.
    assert not re.search(r"^\s*&?\s*python(\.exe)?\s+-u", cuerpo, re.M)


def test_corre_desde_la_raiz_del_repo(cuerpo):
    """Todos los pasos usan rutas relativas (`scripts\\...`, `logs\\...`)."""
    assert "$root = Split-Path -Parent $PSScriptRoot" in cuerpo
    assert "Set-Location $root" in cuerpo
    assert cuerpo.index("Set-Location $root") < cuerpo.index("[1/9]")


def test_corta_al_primer_error(cuerpo):
    assert '$ErrorActionPreference = "Stop"' in cuerpo


def test_el_log_se_trunca_una_vez_y_despues_se_appendea(cuerpo):
    """Sin -Append cada paso borraría el log del anterior y quedaría sólo el último."""
    tees = re.findall(r"Tee-Object -FilePath logs.finalize\.log( -Append)?", cuerpo)
    assert tees, "ningún paso deja log"
    assert tees[0] == "", "el primer paso tiene que truncar el log"
    assert all(t == " -Append" for t in tees[1:])


def test_el_encabezado_describe_los_nueve_pasos(ps1):
    """El comentario de arriba es lo que se lee antes de correrlo: tiene que estar."""
    encabezado = ps1[: ps1.index(CORTE)]
    for n in range(1, 10):
        assert f"{n})" in encabezado
    assert "DESTRUCTIVO" in encabezado
