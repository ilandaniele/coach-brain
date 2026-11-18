"""Verificador de huecos silenciosos del corpus.

Todas las etapas del pipeline son resumables por existencia del archivo de
salida: si el JSON ya existe, la etapa saltea la fuente sin mirar lo que hay
adentro. Eso hace que una salida vacia o casi vacia sea permanente — nadie la
reintenta nunca. Este script busca justamente esas salidas y, lo mas
importante, separa las que son legitimas de las que son un hueco de verdad.

Que revisa:
  1. Transcripts con `full_text` vacio. Mide el volumen de la fuente: a
     -60 dB o menos es silencio digital (video muteado, nota de voz muda) y
     no hay nada que recuperar. Con audio audible y 0 segmentos, es un hueco.
  2. Transcripts con muy pocos caracteres por minuto de audio. Una nota de
     voz de 5 segundos con tres palabras es normal; 128 segundos que dan
     solamente "Gracias." no lo es.
  3. PDFs cuya capa de texto no alcanza (menos de OCR_MIN_CHARS_PER_PAGE
     caracteres por pagina). Se arreglan con `ingest.pdfs --ocr`.
  4. Texto sin espacios. Un PDF puede dar la cantidad correcta de caracteres
     y ser ilegible igual: "Estoesparaquevayasdiaadia". Pasa cualquier
     control de volumen y no sirve para nada.
  5. Transcripts sin extraccion, y extracciones con 0 items.
  6. Extracciones cuyo transcript ya no existe.

Uso:
  .venv\\Scripts\\python.exe scripts/verificar.py
  .venv\\Scripts\\python.exe scripts/verificar.py --sin-audio   (no mide volumen)

Devuelve 0 si no hay nada sospechoso y 1 si hay. Solo lee: no escribe ni
borra nada.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRANSCRIPTS = REPO / "data" / "processed" / "transcripts"
EXTRACTED = REPO / "data" / "processed" / "extracted"

# Un video muteado o una nota de voz muda dan -91 dB (silencio digital puro).
# Habla real, incluso lejana o de fondo, queda bien arriba de -60.
SILENCIO_DB = -60.0

# Habla normal ronda los 800-1000 caracteres por minuto. Por debajo de este
# piso, con audio suficientemente largo, algo se perdio.
MIN_CHARS_POR_MINUTO = 60.0
DUR_MINIMA_PARA_JUZGAR = 30.0

# El mismo umbral que usa ingest/pdfs.py para decidir si rasteriza.
OCR_MIN_CHARS_PER_PAGE = 400

# Proporcion de espacios sobre el total de caracteres. Es el test que
# distingue prosa de una tira de glifos pegados: pypdf no reconstruye los
# espacios cuando el PDF los implementa como posicionamiento, y devuelve la
# cantidad correcta de caracteres, asi que el texto roto pasa cualquier
# control de volumen. La mediana del corpus es 0,19 y el minimo real es
# 0,1081 (OCR de capturas de chat); los nueve PDFs roto daban de 0,0000 a
# 0,0507. El mismo umbral vive en ingest/pdfs.py.
MIN_SPACE_RATIO = 0.08

# Por debajo de este largo la proporcion de espacios no dice nada: un
# transcript de diez caracteres puede no tener ninguno legitimamente.
CHARS_MINIMO_PARA_ESPACIOS = 300

# Un transcript de este largo que extrae 0 items no es un audio corto sin
# contenido: es una extraccion que fallo.
CHARS_EXTRACCION_EXIGIBLE = 1500

_MEAN_RE = re.compile(r"mean_volume:\s*(-?[\d.]+)")


def volumen_medio(path: Path):
    """mean_volume en dB segun ffmpeg, o None si no se pudo medir."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
             "-vn", "-ac", "1", "-ar", "16000", "-af", "volumedetect",
             "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=600,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    m = _MEAN_RE.search(r.stderr or "")
    return float(m.group(1)) if m else None


def cargar():
    """Los transcripts, con lo justo para poder juzgarlos."""
    out = []
    for tp in sorted(TRANSCRIPTS.glob("*.json")):
        try:
            d = json.loads(tp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            out.append({"archivo": tp.name, "stem": tp.stem, "ilegible": str(e)})
            continue
        md = d.get("metadata") or {}
        sp = d.get("source_path") or ""
        if sp:
            abs_src = Path(sp) if os.path.isabs(sp) else REPO / sp
        else:
            abs_src = None
        txt = (d.get("full_text") or "").strip()
        out.append({
            "archivo": tp.name,
            "stem": tp.stem,
            "tipo": d.get("source_type") or "",
            "chars": len(txt),
            "espacios": (txt.count(" ") / len(txt)) if txt else 0.0,
            "dur": md.get("duration_seconds") or 0.0,
            "pages": md.get("pages") or 0,
            "engine": md.get("engine"),
            "fuente": abs_src if abs_src is not None and abs_src.exists() else None,
            "ilegible": None,
        })
    return out


def revisar_transcripts(ts, medir_audio: bool) -> int:
    """Imprime el estado de la etapa de transcripcion. Devuelve sospechosos."""
    sospechosos = 0

    ilegibles = [t for t in ts if t.get("ilegible")]
    if ilegibles:
        sospechosos += len(ilegibles)
        print("\nJSON ILEGIBLE: %d" % len(ilegibles))
        for t in ilegibles:
            print("   %s  %s" % (t["archivo"], t["ilegible"]))

    vacios = [t for t in ts if not t.get("ilegible") and t["chars"] == 0]
    print("\nTranscripts vacios: %d" % len(vacios))
    mudos, con_audio, sin_medir = [], [], []
    if vacios and medir_audio:
        print("   midiendo volumen de las fuentes con ffmpeg...", flush=True)
    for t in vacios:
        db = volumen_medio(t["fuente"]) if (medir_audio and t["fuente"]) else None
        if db is None:
            sin_medir.append(t)
        elif db <= SILENCIO_DB:
            mudos.append((t, db))
        else:
            con_audio.append((t, db))
    print("   silencio digital, nada que recuperar: %d" % len(mudos))
    print("   CON AUDIO AUDIBLE y 0 segmentos: %d" % len(con_audio))
    for t, db in sorted(con_audio, key=lambda x: -x[1]):
        print("      %6.1f dB  %6.0fs  %s" % (db, t["dur"], t["archivo"]))
    if sin_medir:
        print("   sin medir, fuente ausente o ffmpeg no disponible: %d" % len(sin_medir))
    sospechosos += len(con_audio)

    pobres = [
        t for t in ts
        if not t.get("ilegible") and t["chars"] > 0
        and t["dur"] >= DUR_MINIMA_PARA_JUZGAR
        and t["chars"] / (t["dur"] / 60.0) < MIN_CHARS_POR_MINUTO
    ]
    print("\nTranscripts con menos de %.0f chars por minuto (y mas de %.0fs): %d"
          % (MIN_CHARS_POR_MINUTO, DUR_MINIMA_PARA_JUZGAR, len(pobres)))
    for t in sorted(pobres, key=lambda t: -t["dur"]):
        print("   %6.0fs %5d chars  %s" % (t["dur"], t["chars"], t["archivo"]))
    sospechosos += len(pobres)

    pdf_pobres = [
        t for t in ts
        if not t.get("ilegible") and t["tipo"] == "pdf" and t["pages"]
        and t["chars"] < OCR_MIN_CHARS_PER_PAGE * t["pages"]
    ]
    print("\nPDFs con capa de texto pobre, menos de %d chars por pagina: %d"
          % (OCR_MIN_CHARS_PER_PAGE, len(pdf_pobres)))
    for t in sorted(pdf_pobres, key=lambda t: t["chars"] / max(1, t["pages"])):
        print("   %6.0f ch/pg  %4d pg  engine=%s  %s"
              % (t["chars"] / max(1, t["pages"]), t["pages"], t["engine"], t["archivo"]))
    sospechosos += len(pdf_pobres)

    pegados = [
        t for t in ts
        if not t.get("ilegible") and t["chars"] >= CHARS_MINIMO_PARA_ESPACIOS
        and t["espacios"] < MIN_SPACE_RATIO
    ]
    print("\nTexto sin espacios, menos de %.2f de espacios sobre el total: %d"
          % (MIN_SPACE_RATIO, len(pegados)))
    for t in sorted(pegados, key=lambda t: -t["chars"]):
        print("   esp=%.4f %8d chars  engine=%s  %s"
              % (t["espacios"], t["chars"], t["engine"], t["archivo"]))
    sospechosos += len(pegados)

    return sospechosos


def revisar_extraccion(ts) -> int:
    """Imprime el estado de la etapa de extraccion. Devuelve sospechosos."""
    stems = {t["stem"] for t in ts if not t.get("ilegible")}
    sin_extraer, vacias = [], []
    for t in ts:
        if t.get("ilegible"):
            continue
        ep = EXTRACTED / (t["stem"] + ".extracted.json")
        if not ep.exists():
            sin_extraer.append(t)
            continue
        try:
            ed = json.loads(ep.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            vacias.append(t)
            continue
        if sum(len(ed.get(k) or []) for k in ("principios", "situaciones", "frases")) == 0:
            vacias.append(t)

    sufijo = ".extracted.json"
    huerfanas = [
        f for f in (sorted(EXTRACTED.glob("*" + sufijo)) if EXTRACTED.exists() else [])
        if f.name[: -len(sufijo)] not in stems
    ]

    print("\nSin extraccion: %d" % len(sin_extraer))
    print("Extraccion con 0 items: %d (en audios cortos suele ser legitimo)" % len(vacias))
    exigibles = [t for t in vacias if t["chars"] >= CHARS_EXTRACCION_EXIGIBLE]
    print("   de esas, con transcript de %d+ chars: %d"
          % (CHARS_EXTRACCION_EXIGIBLE, len(exigibles)))
    for t in sorted(exigibles, key=lambda t: -t["chars"])[:30]:
        print("      %7d chars  %s" % (t["chars"], t["archivo"]))
    print("Extracciones sin transcript: %d" % len(huerfanas))
    for f in huerfanas[:20]:
        print("   %s" % f.name)

    return len(exigibles)


def main() -> int:
    ap = argparse.ArgumentParser(description="Busca huecos silenciosos en el corpus.")
    ap.add_argument(
        "--sin-audio", action="store_true",
        help="No mide el volumen de las fuentes vacias: mas rapido, menos concluyente",
    )
    args = ap.parse_args()

    ts = cargar()
    print("transcripts: %d" % len(ts))
    sospechosos = revisar_transcripts(ts, medir_audio=not args.sin_audio)
    sospechosos += revisar_extraccion(ts)
    print("\nSOSPECHOSOS: %d" % sospechosos)
    return 1 if sospechosos else 0


if __name__ == "__main__":
    sys.exit(main())
