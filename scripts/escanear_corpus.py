"""Pasa el detector de alucinaciones por el corpus VIVO. Solo lee.

Que el detector exista para la auditoria sin VAD no significa que el corpus
con VAD este limpio. El transcript **con VAD** de `wa_audio_00021896` son 1907
caracteres de "!Gracias por ver el video!" repetido 25 veces, y lo encontro una
persona leyendo un log, no el pipeline. Este script hace esa busqueda sobre los
8838 transcripts en vez de a ojo.

Un transcript alucinado no llega al indice por si solo. Las 3 colecciones se
construyen desde `extracted/`, no desde los transcripts: un transcript entra al
indice solo a traves de los items que el extractor saco de el. Por eso el
reporte dice, al lado de cada marcado, cuantos items tiene su extraccion. 0
items significa que el extractor ya lo filtro y que no hay nada que sacar del
indice; items > 0 es lo unico que de verdad hay que mirar. Medido en la ultima
corrida: los 4 marcados dieron 0 items, incluido el de 1907 caracteres.

No escribe nada. Es el paso 6 de `finalize.ps1`, de solo lectura: informa, nada
mas, porque sacar o re-transcribir un transcript vivo cambia lo que devuelve el
indice y eso lo decide una persona que leyo el texto.

    .venv\\Scripts\\python.exe -I -X utf8 scripts/escanear_corpus.py

El criterio es el mismo que usa la auditoria: `parece_alucinacion()` para
marcar y `avisos()` para anotar lo que no alcanza a marcarse. Leer el reporte
sabiendo dos cosas, las dos medidas en la ultima corrida:

- Los marcados de este corpus son transcripts de 8 a 12 caracteres: ruido que
  el VAD dejo pasar.
- El aviso de cobertura sobre un texto corto casi nunca es un bucle, porque en
  un texto corto cualquier repeticion cubre mucho. Lo que vale es un resultado
  que junte las DOS senales.
- De los 36 avisados, 20 tienen items extraidos. Eso no los vuelve basura: son
  notas de voz reales en las que alguien repite una frase. El aviso es una
  anotacion, no un veredicto.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
TRANSCRIPTS = REPO / "data" / "processed" / "transcripts"
EXTRACTED = REPO / "data" / "processed" / "extracted"

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


def items_extraidos(stem: str, extracted: pathlib.Path = EXTRACTED) -> int | None:
    """Items que el extractor saco de ese transcript, o None si no hay extraccion.

    Es la diferencia entre "hay basura en el corpus" y "hay basura en el
    indice": el indice se arma desde `extracted/`, asi que un transcript
    alucinado con 0 items no aporto ni un vector. None no es 0: significa que
    la etapa de extraccion todavia no paso por ahi y que la respuesta no se
    sabe, no que no haya nada.
    """
    p = extracted / (stem + ".extracted.json")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return sum(len(d.get(k) or []) for k in ("principios", "situaciones", "frases"))


def _etiqueta_items(n: int | None) -> str:
    if n is None:
        return "sin extraer"
    return "%d items" % n


def main() -> None:
    if not TRANSCRIPTS.is_dir():
        raise SystemExit("no existe %s" % TRANSCRIPTS)

    leidos, marcados, avisados = escanear(TRANSCRIPTS)
    chars = sum(n for _, n, _ in marcados)

    print("transcripts leidos: %d" % leidos)
    print("MARCADOS como alucinacion: %d (%d chars)" % (len(marcados), chars))
    print("con aviso pero no marcados: %d" % len(avisados))

    # Lo unico que obliga a tocar el indice: un marcado que aporto items.
    items = {st: items_extraidos(st) for st, _, _ in marcados}
    llegaron = [st for st, n in items.items() if n]
    print("de los marcados, con items en el indice: %d" % len(llegaron))
    print("   sin extraer todavia: %d | con 0 items (filtrados): %d"
          % (sum(1 for n in items.values() if n is None),
             sum(1 for n in items.values() if n == 0)))
    for st in llegaron:
        print("   HAY QUE DECIDIR: %s (%s)" % (st, _etiqueta_items(items[st])))

    tipos = collections.Counter(
        s.split(":")[0] if ":" in s else s for _, _, s in marcados)
    for k, v in tipos.most_common():
        print("  %4d  %s" % (v, k))

    print("\n--- los 25 marcados mas grandes ---", flush=True)
    for st, n, s in sorted(marcados, key=lambda r: -r[1])[:25]:
        print("  %6d  %-12s %-45s %s"
              % (n, _etiqueta_items(items.get(st)), st[:45], s[:80]))

    print("\n--- los 15 avisados mas grandes ---", flush=True)
    for st, n, a in sorted(avisados, key=lambda r: -r[1])[:15]:
        print("  %6d  %-12s %-45s %s"
              % (n, _etiqueta_items(items_extraidos(st)), st[:45],
                 " | ".join(a)[:80]))

    # Las dos senales juntas son la combinacion que vale la pena leer: una sola
    # sobre un texto corto casi siempre es habla real repetida.
    dobles = [(st, n) for st, n, a in avisados if len(a) >= 2]
    print("\n--- avisados que juntan las dos senales: %d ---" % len(dobles))
    for st, n in sorted(dobles, key=lambda r: -r[1]):
        print("  %6d  %-12s %s" % (n, _etiqueta_items(items_extraidos(st)), st))


if __name__ == "__main__":
    main()
