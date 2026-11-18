"""Tests del piso que protege a INVENTARIO.md de un recorrido fallido.

`scan_disk()` recorre el material con `os.walk`, que no levanta excepción
cuando la ruta no se puede recorrer: devuelve cero archivos. Con las junctions
de `data/raw` apuntando a una carpeta movida, el inventario saldría diciendo
que en disco no hay material. Y `manifest.py` ahora corre solo, como paso 7 de
`finalize.ps1`, sobrescribiendo INVENTARIO.md: el registro con el que se decide
qué material se puede borrar.

Un inventario vacío es peor que ninguno, porque dice exactamente lo que
alguien querría leer antes de borrar 360 GB.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _cargar():
    spec = importlib.util.spec_from_file_location(
        "manifest_piso", REPO / "scripts" / "manifest.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_el_caso_normal_sigue_pasando():
    """17801 archivos para 8838 transcripts: el estado real del proyecto."""
    m = _cargar()
    assert m.exigir_piso(17801, 8838, argv=[]) is None


def test_mismos_numeros_pasa():
    """El invariante es `>=`, no `>`: un corpus donde todo tiene transcript."""
    m = _cargar()
    assert m.exigir_piso(8838, 8838, argv=[]) is None


def test_disco_vacio_aborta():
    """El caso que motiva todo esto: el recorrido no encontró nada."""
    m = _cargar()
    with pytest.raises(SystemExit) as e:
        m.exigir_piso(0, 8838, argv=[])
    assert "ABORTADO" in str(e.value)
    assert "No se escribio nada" in str(e.value)


def test_una_sola_junction_viva_aborta():
    """El caso parcial, que es el más fácil de no notar.

    Si sobrevive la junction de `videos` y se cae la de `whatsapp`, el
    recorrido trae unos cientos de archivos en vez de 17801. No es cero, así
    que un chequeo de "vino algo" no lo atraparía.
    """
    m = _cargar()
    with pytest.raises(SystemExit):
        m.exigir_piso(262, 8838, argv=[])


def test_sin_piso_lo_desactiva():
    """Después de un borrado autorizado, el inventario vacío es el correcto."""
    m = _cargar()
    assert m.exigir_piso(0, 8838, argv=["--sin-piso"]) is None


def test_sin_transcripts_no_aborta():
    """Un repo recién clonado, antes de la primera corrida.

    Sin transcripts el piso es 0 y cualquier disco lo cumple. Si abortara acá,
    nadie podría generar el primer inventario.
    """
    m = _cargar()
    assert m.exigir_piso(0, 0, argv=[]) is None


def test_el_mensaje_dice_donde_mirar():
    """El que lea el abort tiene que saber qué revisar sin leer el código."""
    m = _cargar()
    with pytest.raises(SystemExit) as e:
        m.exigir_piso(0, 100, argv=[])
    msg = str(e.value)
    assert "data/raw" in msg
    assert "--sin-piso" in msg
