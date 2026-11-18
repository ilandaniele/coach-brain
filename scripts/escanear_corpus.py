"""Pasa el detector de alucinaciones por el corpus VIVO. Solo lee.

Que el detector exista para la auditoria sin VAD no significa que el corpus
con VAD este limpio. El transcript **con VAD** de `wa_audio_00021896` son 1907
caracteres de "!Gracias por ver el video!" repetido 25 veces: esa alucinacion
ya esta indexada, y la encontro una persona leyendo un log, no el pipeline.
Este script hace esa busqueda sobre los 8838 transcripts en vez de a ojo.

No escribe nada y no esta en `finalize.ps1`. Informa, nada mas, porque sacar o
re-transcribir un transcript vivo cambia lo que devuelve el indice y eso lo
decide una persona que leyo el texto.

    .venv\\Scripts\\python.exe -I -X utf8 scripts/escanear_corpus.py

El criterio es el mismo que usa la auditoria: `parece_alucinacion()` para
marcar y `avisos()` para anotar lo que no alcanza a marcarse. Leer el reporte
sabiendo dos cosas, las dos medidas en la ultima corrida:

- Los marcados de este corpus son transcripts de 8 a 12 caracteres: ruido que
  el VAD dejo pasar.
- El aviso de cobertura sobre un texto corto casi nunca es un bucle, porque en
  un texto corto cualquier repeticion cubre mucho. Lo que vale es un resultado
  que junte las DOS senales.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
TRANSCRIPTS = REPO / "data" / "processed" / "transcripts"

# Los umbrales y el criterio viven en un solo lugar. Importar el modulo por
# path y no por nombre porque `scripts/` no es parte del paquete instalable.
_spec = importlib.util.spec_from_file_location(
    "novad_detector", REPO / "scripts" / "retranscribir_sin_vad.py")
_novad = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_novad)


def escanear(directorio: pathlib.Path):
    """Devuelve (leidos, marcados, avisados) sin tocar ningun archivo."""
    marcados: list[tuple[str, int, str]] = []
    avisados: list[tuple[str, int, list[str]]] = []
    leidos = 0
    for p in sorted(directorio.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Un transcript ilegible es un problema distinto, del verificador.
            continue
        leidos += 1
        t = (d.get("full_text") or "").strip()
        if not t:
            # Un transcript vacio no es una alucinacion: es el gotcha #21.
            continue
        senal = _novad.parece_alucinacion(t)
        if senal:
            marcados.append((p.stem, len(t), senal))
            continue
        avs = _novad.avisos(t)
        if avs:
            avisados.append((p.stem, len(t), avs))
    return leidos, marcados, avisados


def main() -> None:
    if not TRANSCRIPTS.is_dir():
        raise SystemExit("no existe %s" % TRANSCRIPTS)

    leidos, marcados, avisados = escanear(TRANSCRIPTS)
    chars = sum(n for _, n, _ in marcados)

    print("transcripts leidos: %d" % leidos)
    print("MARCADOS como alucinacion: %d (%d chars)" % (len(marcados), chars))
    print("con aviso pero no marcados: %d" % len(avisados))

    tipos = collections.Counter(
        s.split(":")[0] if ":" in s else s for _, _, s in marcados)
    for k, v in tipos.most_common():
        print("  %4d  %s" % (v, k))

    print("\n--- los 25 marcados mas grandes ---", flush=True)
    for st, n, s in sorted(marcados, key=lambda r: -r[1])[:25]:
        print("  %6d  %-55s %s" % (n, st[:55], s[:90]))

    print("\n--- los 15 avisados mas grandes ---", flush=True)
    for st, n, a in sorted(avisados, key=lambda r: -r[1])[:15]:
        print("  %6d  %-55s %s" % (n, st[:55], " | ".join(a)[:95]))

    # Las dos senales juntas son la combinacion que vale la pena leer: una sola
    # sobre un texto corto casi siempre es habla real repetida.
    dobles = [(st, n) for st, n, a in avisados if len(a) >= 2]
    print("\n--- avisados que juntan las dos senales: %d ---" % len(dobles))
    for st, n in sorted(dobles, key=lambda r: -r[1]):
        print("  %6d  %s" % (n, st))


if __name__ == "__main__":
    main()
