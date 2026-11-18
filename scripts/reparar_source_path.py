"""Vuelve a apuntar el `source_path` de los transcripts cuyo archivo se movió.

El problema
-----------
`data/raw/docs`, `data/raw/pdfs` y `data/raw/videos` son los tres junctions al
mismo destino (`Desktop\\Ilan\\Seduccion`). Después de la ingesta, el material
se reorganizó dentro de ese destino: las clases y los libros pasaron a vivir un
nivel más abajo, dentro de `Bonus/`. Los transcripts quedaron con el path viejo,
que ya no resuelve.

Eso no borró nada y no se nota en el uso normal del corpus: el texto ya está
transcripto, extraído e indexado, y `scripts/manifest.py` cubre el material en
disco por nombre y por duración, así que el inventario sigue siendo correcto.
El daño es diferido, y por eso vale la pena arreglarlo ahora:

- Cualquier etapa que necesite volver a la fuente (re-transcribir sin VAD,
  recalcular una duración, re-extraer el texto de un PDF) falla con "no existe"
  sobre un archivo que sí está en disco, a un nivel de distancia.
- Un `source_path` que no resuelve es indistinguible, leído desde el corpus, de
  un archivo que de verdad se perdió. Es exactamente la confusión que no se
  puede tener antes de decidir qué material se borra.

Cómo resuelve
-------------
Tres pases, del más seguro al menos seguro, y ninguno adivina:

1. **Por nombre.** Si el nombre de archivo aparece exactamente una vez bajo
   `data/raw`, ese es el archivo. Es el caso de 149 de las 154 clases: cambió la
   carpeta, no el nombre.
2. **Por nombre con el sufijo del pipeline.** Varias fuentes quedaron
   registradas con un sufijo que el archivo en disco no tiene
   (`78 que es el valor-008.mp4` en el transcript contra
   `78 que es el valor.mp4` en disco). Se prueba sacando un sufijo `-NNN` final.
3. **Por duración.** Último recurso y sólo para audio y video: `ffprobe` lee la
   duración del contenedor sin decodificar el archivo, y el transcript ya la
   tiene guardada. Es la única huella que sobrevive a un renombre completo
   (gotcha #23). Tolerancia de 1 segundo, y sólo contra archivos que ningún
   transcript reclama por nombre.

Un nombre que aparece en dos lugares distintos NO se resuelve: se reporta como
ambiguo y queda para que lo mire una persona. Preferir un candidato cualquiera
sería inventar una respuesta.

Lo que queda sin resolver no es necesariamente una pérdida
-----------------------------------------------------------
Un cuarto pase, que no escribe nada, explica el resto: busca si otro transcript
tiene el MISMO contenido y sí tiene archivo en disco. Cuando lo encuentra, el
transcript sin fuente es un duplicado: la misma clase o el mismo libro entró al
pipeline dos veces con nombres distintos, y después se borró una de las dos
copias del archivo.

Eso cambia por completo lo que hay que hacer. Un transcript sin fuente y sin
gemelo es material que ya no se puede volver a procesar. Un transcript sin
fuente pero con gemelo vivo es texto repetido dentro del corpus: no falta nada,
sobra. El pase lo reporta y no toca nada, porque sacar un duplicado del corpus
borra datos y eso lo decide una persona.

Uso
---
    python scripts/reparar_source_path.py            # pase seco, no escribe nada
    python scripts/reparar_source_path.py --apply    # escribe, con backup

Con `--apply` cada transcript modificado se respalda en
`data/processed/_transcripts_backup_source_path/` antes de tocarlo, y el path
viejo queda guardado en `metadata.source_path_previo` para que el cambio sea
auditable desde el propio archivo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
TRANSCRIPTS = REPO / "data" / "processed" / "transcripts"
BACKUP = REPO / "data" / "processed" / "_transcripts_backup_source_path"

MEDIA_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v",
             ".opus", ".m4a", ".mp3", ".ogg", ".wav", ".aac"}

# Sufijo que el pipeline le agrego a algunos nombres al registrarlos:
# "78 que es el valor-008.mp4" en el transcript, "78 que es el valor.mp4" en disco.
SUFIJO_PIPELINE = re.compile(r"-\d{3}$")

TOLERANCIA_DURACION = 1.0

# Para declarar que dos transcripts son el mismo contenido. El arranque tiene que
# coincidir caracter por caracter, y encima tiene que coincidir la duracion o el
# largo total. Dos transcripts de Whisper de un mismo video no dan el mismo
# largo (cada corrida reparte distinto el texto), pero la duracion del
# contenedor es identica al microsegundo. Dos PDF iguales dan el mismo largo.
CABECERA_DUPLICADO = 200
TOLERANCIA_DUPLICADO = 0.01


def _real(p: Path) -> str:
    """realpath en minusculas, para deduplicar los tres junctions al mismo destino."""
    try:
        return os.path.realpath(str(p)).lower()
    except OSError:
        return str(p).lower()


def indexar_disco() -> dict[str, list[Path]]:
    """Nombre de archivo en minusculas -> rutas reales distintas bajo data/raw.

    Deduplica directorios por realpath: sin eso los tres junctions hacen que
    cada archivo aparezca tres veces y todo nombre quede "ambiguo".
    """
    vistos: set[str] = set()
    idx: dict[str, list[Path]] = defaultdict(list)
    ya: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(RAW):
        rp = _real(Path(dirpath))
        if rp in vistos:
            dirnames[:] = []
            continue
        vistos.add(rp)
        for fn in filenames:
            p = Path(dirpath) / fn
            k = _real(p)
            if k in ya:
                continue
            ya.add(k)
            idx[fn.lower()].append(p)
    return idx


def _duracion(path: Path) -> float | None:
    """Duracion del contenedor segun ffprobe, sin decodificar el archivo."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    try:
        return float((r.stdout or "").strip())
    except ValueError:
        return None


def resolver(roto: Path, idx: dict[str, list[Path]]) -> tuple[Path | None, str]:
    """Devuelve (ruta nueva, como se resolvio) o (None, motivo)."""
    nombre = roto.name.lower()
    cands = idx.get(nombre) or []
    if len(cands) == 1:
        return cands[0], "nombre"
    if len(cands) > 1:
        return None, "ambiguo: %d candidatos" % len(cands)

    base = SUFIJO_PIPELINE.sub("", roto.stem)
    if base != roto.stem:
        alt = (base + roto.suffix).lower()
        cands = idx.get(alt) or []
        if len(cands) == 1:
            return cands[0], "nombre sin sufijo del pipeline"
        if len(cands) > 1:
            return None, "ambiguo sin sufijo: %d candidatos" % len(cands)
    return None, "sin candidato por nombre"


def huella(d: dict) -> tuple[float | None, int, str]:
    """(duracion, largo del texto, arranque) para comparar dos transcripts."""
    t = d.get("full_text") or ""
    dur = (d.get("metadata") or {}).get("duration_seconds")
    if not isinstance(dur, (int, float)) or not dur or dur <= 0:
        dur = None
    return (float(dur) if dur is not None else None, len(t), t[:CABECERA_DUPLICADO])


def es_duplicado(a: tuple[float | None, int, str], b: tuple[float | None, int, str]) -> bool:
    """Mismo contenido: arranque identico y ademas misma duracion o mismo largo."""
    if not a[2] or a[2] != b[2]:
        return False
    if a[1] == b[1]:
        return True
    if a[0] is not None and b[0] is not None:
        return abs(a[0] - b[0]) <= TOLERANCIA_DUPLICADO
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="escribe los transcripts (por defecto es un pase seco)")
    ap.add_argument("--sin-ffprobe", action="store_true",
                    help="saltea el pase por duracion")
    args = ap.parse_args()

    if not TRANSCRIPTS.is_dir():
        print("no existe %s" % TRANSCRIPTS)
        return 1

    print("indexando %s..." % RAW, flush=True)
    idx = indexar_disco()
    print("  %d nombres distintos" % len(idx))

    rotos: list[tuple[Path, dict, Path]] = []
    reclamados: set[str] = set()
    huellas: dict[str, tuple[float | None, int, str]] = {}
    con_fuente: set[str] = set()
    total = 0
    for p in sorted(TRANSCRIPTS.glob("*.json")):
        total += 1
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print("ILEGIBLE %s: %s" % (p.name, e))
            continue
        huellas[p.stem] = huella(d)
        sp = d.get("source_path")
        if not sp:
            continue
        src = Path(sp)
        if src.exists():
            reclamados.add(src.name.lower())
            con_fuente.add(p.stem)
            continue
        rotos.append((p, d, src))

    print("transcripts: %d | con source_path roto: %d" % (total, len(rotos)))
    if not rotos:
        return 0

    arreglos: list[tuple[Path, dict, Path, Path, str]] = []
    pendientes: list[tuple[Path, dict, Path, str]] = []
    for p, d, src in rotos:
        nueva, como = resolver(src, idx)
        if nueva is not None:
            arreglos.append((p, d, src, nueva, como))
            reclamados.add(nueva.name.lower())
        else:
            pendientes.append((p, d, src, como))

    # Pase por duracion, solo sobre lo que quedo sin resolver y solo para medios.
    if pendientes and not args.sin_ffprobe:
        huerfanos = [
            c
            for nombre, lst in idx.items()
            if nombre not in reclamados and Path(nombre).suffix.lower() in MEDIA_EXT
            for c in lst
        ]
        medios = [t for t in pendientes if t[2].suffix.lower() in MEDIA_EXT]
        if medios and huerfanos:
            print("\npase por duracion: %d sin resolver contra %d archivo(s) que "
                  "ningun transcript reclama..." % (len(medios), len(huerfanos)))
            durs: dict[str, float | None] = {}
            for p, d, src, _como in medios:
                esperada = ((d.get("metadata") or {}).get("duration_seconds"))
                if not isinstance(esperada, (int, float)) or esperada <= 0:
                    continue
                mejor: tuple[float, Path] | None = None
                for h in huerfanos:
                    k = str(h)
                    if k not in durs:
                        durs[k] = _duracion(h)
                    dh = durs[k]
                    if dh is None:
                        continue
                    delta = abs(dh - float(esperada))
                    if mejor is None or delta < mejor[0]:
                        mejor = (delta, h)
                if mejor is not None and mejor[0] <= TOLERANCIA_DURACION:
                    arreglos.append((p, d, src, mejor[1],
                                     "duracion (%.1fs, delta %.2fs)" % (esperada, mejor[0])))
                    huerfanos.remove(mejor[1])
                    pendientes = [t for t in pendientes if t[0] != p]

    print("\n=== RESUELTOS: %d ===" % len(arreglos))
    por_como: dict[str, int] = defaultdict(int)
    for _p, _d, src, nueva, como in arreglos:
        por_como[como.split(" (")[0]] += 1
    for k, v in sorted(por_como.items(), key=lambda t: -t[1]):
        print("  %-34s %d" % (k, v))
    for _p, _d, src, nueva, como in arreglos[:5]:
        print("  ej: %s\n      -> %s  [%s]" % (src, nueva, como))

    # Cuarto pase, solo informativo: lo que quedo sin fuente, ¿tiene un gemelo
    # con el mismo contenido que SI tiene archivo en disco?
    for _p, _d, _src, nueva, _como in arreglos:
        con_fuente.add(_p.stem)
    duplicados: list[tuple[Path, Path, str]] = []
    huerfanos_reales: list[tuple[Path, Path, str]] = []
    for p, _d, src, como in pendientes:
        mia = huellas.get(p.stem)
        gemelo = None
        if mia:
            for stem in con_fuente:
                if stem != p.stem and es_duplicado(mia, huellas.get(stem, (None, -1, ""))):
                    gemelo = stem
                    break
        if gemelo:
            duplicados.append((p, src, gemelo))
        else:
            huerfanos_reales.append((p, src, como))

    if duplicados:
        print("\n=== SIN FUENTE PERO DUPLICADOS: %d ===" % len(duplicados))
        print("El mismo contenido entro al pipeline dos veces con nombres distintos")
        print("y despues se borro una de las dos copias del archivo. No falta nada:")
        print("sobra texto repetido en el corpus. Sacarlo lo decide una persona.")
        for p, src, gemelo in duplicados:
            print("  %-44s == %s" % (p.stem[:44], gemelo))
            print("      su archivo era: %s" % src)

    if huerfanos_reales:
        print("\n=== SIN RESOLVER Y SIN GEMELO: %d ===" % len(huerfanos_reales))
        print("Transcripts cuya fuente no esta en disco y cuyo contenido no esta")
        print("repetido en ningun otro transcript: no se pueden volver a procesar.")
        for p, src, como in huerfanos_reales:
            print("  %-44s %s  (%s)" % (p.stem[:44], src, como))

    if not args.apply:
        print("\npase seco. nada escrito. correr con --apply para aplicar.")
        return 0

    BACKUP.mkdir(parents=True, exist_ok=True)
    n = 0
    for p, d, src, nueva, como in arreglos:
        shutil.copy2(p, BACKUP / p.name)
        d.setdefault("metadata", {})["source_path_previo"] = str(src)
        d["source_path"] = str(nueva)
        # Escritura atomica: `kb.extract` puede estar leyendo estos mismos
        # archivos en paralelo, y un write_text no es atomico. Con os.replace
        # el lector ve el archivo viejo completo o el nuevo completo, nunca uno
        # truncado a medias.
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
        n += 1
    print("\n%d transcript(s) reescritos. backup en %s" % (n, BACKUP))
    return 0


if __name__ == "__main__":
    sys.exit(main())
