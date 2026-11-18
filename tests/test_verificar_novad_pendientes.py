"""Tests del reporte del staging de la auditoría sin VAD.

`data/processed/_novad/` es un directorio que ninguna etapa lee, a propósito:
así una alucinación de Whisper no puede entrar al corpus sola. El costo de ese
diseño es que el staging es invisible, y puede llenarse de trabajo sin juzgar
sin que aparezca en ningún reporte.

Lo que estos tests fijan son las dos propiedades que hacen útil al reporte sin
hacerlo peligroso: que **informe** cuando hay algo esperando, y que **no sume
sospechosos**, porque el verificador devuelve 1 cuando hay sospechosos y se usa
como gate. Si un resultado sin juzgar hiciera fallar el gate, el gate se
apagaría.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _cargar(novad_dir: Path):
    """Importa `verificar.py` con su staging apuntado a un directorio de prueba.

    Se reapunta `NOVAD` en vez de escribir en el real porque el directorio real
    tiene los resultados de una corrida de GPU de horas.
    """
    spec = importlib.util.spec_from_file_location(
        "verificar_novad_pendientes", REPO / "scripts" / "verificar.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.NOVAD = novad_dir
    return mod


def _escribir(d: Path, nombre: str, texto: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / nombre).write_text(json.dumps({"full_text": texto}), encoding="utf-8")


def test_sin_directorio_no_dice_nada(tmp_path, capsys):
    """Antes de la primera corrida el directorio no existe. No es un hallazgo."""
    v = _cargar(tmp_path / "no_existe")
    v.revisar_novad_pendientes([])
    assert capsys.readouterr().out == ""


def test_directorio_vacio_no_dice_nada(tmp_path, capsys):
    """Un staging vacío es el estado normal: todo juzgado, nada pendiente."""
    d = tmp_path / "_novad"
    d.mkdir()
    v = _cargar(d)
    v.revisar_novad_pendientes([])
    assert capsys.readouterr().out == ""


def test_informa_cuantos_hay_y_cuantos_crecieron(tmp_path, capsys):
    """Dos resultados, uno con más texto que el original y otro con menos."""
    d = tmp_path / "_novad"
    _escribir(d, "a.json", "x" * 500)
    _escribir(d, "b.json", "corto")
    v = _cargar(d)
    v.revisar_novad_pendientes([
        {"archivo": "a.json", "chars": 100},
        {"archivo": "b.json", "chars": 400},
    ])
    out = capsys.readouterr().out
    assert "2 resultados" in out
    assert "1 tienen mas texto" in out
    assert "--juzgar" in out


def test_no_suma_sospechosos(tmp_path):
    """Devuelve None, no un contador.

    El verificador sale con 1 cuando hay sospechosos, y se usa como gate en
    `finalize.ps1`. Un resultado sin juzgar no es un hueco del corpus: es
    trabajo pendiente de una persona. Si rompiera el gate, el gate se apagaría.
    """
    d = tmp_path / "_novad"
    _escribir(d, "a.json", "x" * 500)
    v = _cargar(d)
    assert v.revisar_novad_pendientes([{"archivo": "a.json", "chars": 1}]) is None


def test_un_transcript_ilegible_no_lo_hace_explotar(tmp_path, capsys):
    """`cargar()` devuelve entradas sin `chars` cuando el JSON no se pudo leer.

    Son las que llevan `ilegible`. Indexar `t["chars"]` sobre una de ellas tira
    KeyError y se lleva puesto todo el reporte del verificador, incluidos los
    hallazgos reales que vienen antes.
    """
    d = tmp_path / "_novad"
    _escribir(d, "a.json", "x" * 500)
    v = _cargar(d)
    v.revisar_novad_pendientes([
        {"archivo": "a.json", "stem": "a", "ilegible": "Expecting value: line 1"},
    ])
    out = capsys.readouterr().out
    assert "1 resultados" in out
    assert "1 tienen mas texto" in out


def test_un_resultado_ilegible_no_lo_hace_explotar(tmp_path, capsys):
    """Un JSON truncado en el staging se saltea, no corta el reporte.

    Pasa de verdad: el script escribe el resultado mientras corre, así que un
    corte de luz a mitad de una escritura deja un archivo a medias.
    """
    d = tmp_path / "_novad"
    _escribir(d, "bueno.json", "x" * 500)
    d.joinpath("roto.json").write_text('{"full_text": "sin cerra', encoding="utf-8")
    v = _cargar(d)
    v.revisar_novad_pendientes([{"archivo": "bueno.json", "chars": 1}])
    out = capsys.readouterr().out
    assert "2 resultados" in out
    assert "1 tienen mas texto" in out
