"""Inventario de fuentes: cruza lo que hay en disco con lo que entro al pipeline.

Genera:
  data/processed/manifest_fuentes.csv  - una fila por archivo fuente
  INVENTARIO.md                        - resumen legible

Sirve para decidir que material crudo se puede borrar sin perder nada que el
pipeline todavia necesite. Solo lee: no borra ni mueve nada.

Uso:  .venv\\Scripts\\python.exe scripts/manifest.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
TRANSCRIPTS = REPO / "data" / "processed" / "transcripts"
EXTRACTED = REPO / "data" / "processed" / "extracted"
OCR = REPO / "data" / "processed" / "whatsapp_images_ocr"
FUERA = REPO / "data" / "processed" / "_fuera_del_corpus"

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".m4v"}
AUDIO_EXT = {".opus", ".m4a", ".mp3", ".ogg", ".wav", ".aac", ".oga"}
DOC_EXT = {".pdf", ".docx", ".doc", ".epub", ".txt", ".rtf", ".pptx", ".xlsx"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic"}
MEDIA_EXT = VIDEO_EXT | AUDIO_EXT | DOC_EXT | IMG_EXT


def kind(ext: str) -> str:
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    if ext in DOC_EXT:
        return "doc"
    if ext in IMG_EXT:
        return "imagen"
    return "otro"


def real(p: Path) -> str:
    """Path canonico en minusculas, para comparar a traves de las junctions."""
    try:
        return os.path.normcase(os.path.realpath(str(p)))
    except OSError:
        return os.path.normcase(str(p))


def load_transcripts():
    """Indexa los transcripts por path de fuente y por nombre de archivo.

    El `source_path` guardado apunta al layout que tenian las junctions de
    `data/raw` cuando corrio la transcripcion. Esas junctions se re-apuntaron
    despues, asi que para el material que no es de WhatsApp el path ya no
    resuelve. Por eso ademas indexamos por nombre de archivo.
    """
    by_src = {}
    by_base = defaultdict(list)
    huerfanos = []
    for tp in sorted(TRANSCRIPTS.glob("*.json")):
        try:
            d = json.loads(tp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            huerfanos.append({"transcript": tp.name, "motivo": "JSON ilegible: %s" % e})
            continue
        sp = (d.get("source_path") or "").replace("\\", "/")
        chars = len(d.get("full_text") or "")
        ep = EXTRACTED / (tp.stem + ".extracted.json")
        items = None
        if ep.exists():
            try:
                ed = json.loads(ep.read_text(encoding="utf-8"))
                items = sum(
                    len(ed.get(k) or []) for k in ("principios", "situaciones", "frases")
                )
            except Exception:  # noqa: BLE001
                items = -1
        md = d.get("metadata") or {}
        info = {
            "transcript": tp.name,
            "source_type": d.get("source_type") or "",
            "chars": chars,
            "extraido": ep.exists(),
            "items": items,
            "dur": md.get("duration_seconds"),
        }
        if not sp:
            huerfanos.append({"transcript": tp.name, "motivo": "sin source_path"})
            continue
        info["source_path_declarado"] = sp
        abs_src = Path(sp) if os.path.isabs(sp) else (REPO / sp)
        if abs_src.exists():
            by_src[real(abs_src)] = info
        else:
            by_base[os.path.basename(sp).lower()].append(info)
    return by_src, by_base, huerfanos


def scan_disk():
    """Recorre el material crudo una sola vez, deduplicando por realpath.

    Devuelve (fuentes, ignorados). `ignorados` lleva la cuenta de las
    extensiones que no estan en MEDIA_EXT. Sin eso, un formato que el
    pipeline no maneja no aparece en ninguna parte del inventario y parece
    que no existe: fue exactamente lo que paso con un .zip que traia un
    .pptx y un .pdf adentro.
    """
    seen_dirs = set()
    out = {}
    ignorados = {}
    for dirpath, dirnames, filenames in os.walk(RAW):
        rp = real(Path(dirpath))
        if rp in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(rp)
        for fn in filenames:
            p = Path(dirpath) / fn
            ext = p.suffix.lower()
            if ext not in MEDIA_EXT:
                rec = ignorados.setdefault(ext, {"n": 0, "bytes": 0, "ej": []})
                try:
                    rec["bytes"] += p.stat().st_size
                except OSError:
                    pass
                rec["n"] += 1
                if len(rec["ej"]) < 3:
                    rec["ej"].append(str(p))
                continue
            k = real(p)
            if k in out:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            out[k] = {"path": str(p), "ext": ext, "kind": kind(ext), "size": size}
    return out, ignorados


def ocr_coverage():
    """Nombres base (sin extension) de las imagenes que ya pasaron por OCR.

    Las imagenes que estan fuera de data/raw/whatsapp se guardan con un
    sufijo "__<hash8>" para evitar colisiones de nombre (ver _out_name en
    ingest/whatsapp_images_ocr.py). Hay que registrar las dos formas, porque
    la cobertura se matchea contra el nombre base de la imagen original.
    """
    if not OCR.exists():
        return set()
    names = set()
    for f in os.listdir(OCR):
        stem = Path(f).stem.lower()
        names.add(stem)
        m = re.match(r"^(.*)__[0-9a-f]{8}$", stem)
        if m:
            names.add(m.group(1))
    return names


def _dur_ffprobe(path: str):
    """Duracion en segundos segun ffprobe, o None si no se puede leer."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return float((r.stdout or "").strip())
    except ValueError:
        return None


def match_por_duracion(hits, by_base, tol: float = 1.0) -> int:
    """Tercer pase de matcheo de fuentes: por duracion del audio.

    Un archivo renombrado en disco no matchea ni por path ni por nombre, y el
    inventario lo cuenta como material sin transcribir. Eso paso con una clase
    de 2 horas y 3.4 GB que SI estaba transcripta: en disco se llama
    "78 que es el valor.mp4" y el transcript apunta a
    "78 que es el valor-008.mp4". Un falso "no lo tengo" de ese tamano es
    justo el error que no se puede cometer antes de borrar material.

    La duracion es la huella mas barata disponible: el transcript ya la
    guarda y ffprobe la lee del contenedor sin decodificar el archivo. Solo
    se prueba sobre lo que quedo sin cubrir, que son unos pocos archivos.
    """
    pend = [h for h in hits if h[1] is None and h[0]["kind"] in ("video", "audio")]
    sueltos = [
        info
        for lst in by_base.values()
        for info in lst
        if isinstance(info.get("dur"), (int, float)) and info["dur"] > 0
    ]
    if not pend or not sueltos:
        return 0
    print("  %d sin cubrir y %d transcript(s) sin fuente: pruebo por duracion..."
          % (len(pend), len(sueltos)))
    n = 0
    for h in pend:
        d = _dur_ffprobe(h[0]["path"])
        if d is None:
            continue
        mejor = min(sueltos, key=lambda i: abs(i["dur"] - d))
        if abs(mejor["dur"] - d) > tol:
            continue
        mejor["matched"] = True
        sueltos.remove(mejor)
        for lst in by_base.values():
            if mejor in lst:
                lst.remove(mejor)
                break
        h[1] = mejor
        h[2] = "transcript (por duracion)"
        n += 1
        print("    %s  <->  %s  (%.1fs)" % (
            os.path.basename(h[0]["path"]), mejor["transcript"], d))
    return n


def _tiene_audio(path: str):
    """True/False si el contenedor declara pista de audio, None si no se pudo leer.

    None es informacion, no un error: un contenedor que ffprobe no puede abrir
    esta roto, y eso es justamente lo que hay que distinguir de un archivo sano
    que todavia no se transcribio.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return bool((r.stdout or "").strip())


def _stems_fuera_del_corpus() -> set[str]:
    """Stems de las fuentes cuyo transcript se movio a _fuera_del_corpus/.

    Son exclusiones deliberadas (por ejemplo los CV de Ilan que llegaron por
    WhatsApp): el material no es parte del corpus, asi que figurar como "sin
    cubrir" es correcto pero NO es un hueco.
    """
    out: set[str] = set()
    if not FUERA.exists():
        return out
    for p in FUERA.glob("*.json"):
        stem = p.stem
        # El transcript se llama "<tipo>_<stem de la fuente>".
        for pref in ("pdf_", "docx_", "epub_", "video_", "audio_",
                     "wa_video_", "wa_audio_", "whatsapp_"):
            if stem.startswith(pref):
                out.add(stem[len(pref):].lower())
                break
        else:
            out.add(stem.lower())
    return out


def clasificar_sin_cubrir(rows) -> int:
    """Anota en cada fila sin cubrir POR QUE lo esta. Devuelve los huecos reales.

    El inventario es el registro sobre el que se decide si el material crudo se
    puede borrar, asi que "sin cubrir" no puede ser una sola bolsa: un video
    grabado con el microfono apagado no tiene habla que perder, y un transcript
    excluido a proposito tampoco es una perdida. Mezclarlos con el archivo roto
    que SI hay que recuperar hace que el registro no sirva para decidir.
    """
    fuera = _stems_fuera_del_corpus()
    huecos = 0
    for r in rows:
        if r["cubierto"] != "NO":
            r["motivo"] = ""
            continue
        stem = Path(r["path"]).stem.lower()
        if stem in fuera:
            r["motivo"] = "excluido a proposito (transcript en _fuera_del_corpus/)"
            continue
        if r["tipo"] in ("video", "audio"):
            audio = _tiene_audio(r["path"])
            if audio is False:
                r["motivo"] = "sin pista de audio: no hay habla que transcribir"
                continue
            if audio is None:
                r["motivo"] = "contenedor ilegible: ffprobe no lo puede abrir"
                huecos += 1
                continue
        r["motivo"] = "HUECO: falta procesarlo"
        huecos += 1
    return huecos


def gb(b: int) -> str:
    return "%.2f GB" % (b / 1024.0 ** 3)


def main() -> None:
    print("Leyendo transcripts...")
    by_src, by_base, huerfanos = load_transcripts()
    pend = sum(len(v) for v in by_base.values())
    print("  %d con path exacto, %d a resolver por nombre, %d sin source_path"
          % (len(by_src), pend, len(huerfanos)))

    print("Recorriendo material crudo (son cientos de GB, puede tardar)...")
    disk, ignorados = scan_disk()
    print("  %d archivos fuente en disco" % len(disk))

    ocr = ocr_coverage()

    hits = []
    for k, f in sorted(disk.items(), key=lambda kv: kv[1]["path"]):
        info = by_src.get(k)
        via = "transcript"
        if info is None:
            cand = by_base.get(os.path.basename(f["path"]).lower())
            if cand:
                info = cand.pop(0)
                info["matched"] = True
                via = "transcript (por nombre)"
        hits.append([f, info, via])

    n_dur = match_por_duracion(hits, by_base)

    rows = []
    stats = defaultdict(lambda: {"n": 0, "bytes": 0, "cub": 0, "cub_bytes": 0})
    for f, info, via in hits:
        cubierto = info is not None
        if not cubierto and f["kind"] == "imagen":
            if Path(f["path"]).stem.lower() in ocr:
                cubierto = True
                via = "ocr"
        s = stats[f["kind"]]
        s["n"] += 1
        s["bytes"] += max(0, f["size"])
        if cubierto:
            s["cub"] += 1
            s["cub_bytes"] += max(0, f["size"])
        rows.append(
            {
                "path": f["path"],
                "tipo": f["kind"],
                "ext": f["ext"],
                "bytes": f["size"],
                "cubierto": "si" if cubierto else "NO",
                "via": via if cubierto else "",
                "transcript": info["transcript"] if info else "",
                "chars": info["chars"] if info else "",
                "extraido": ("si" if info["extraido"] else "no") if info else "",
                "items": (info["items"] if info["items"] is not None else "") if info else "",
            }
        )

    n_huecos = clasificar_sin_cubrir(rows)

    out_csv = REPO / "data" / "processed" / "manifest_fuentes.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "path", "tipo", "ext", "bytes", "cubierto", "via", "motivo",
        "transcript", "chars", "extraido", "items",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("Escrito %s" % out_csv)

    for lst in by_base.values():
        for info in lst:
            if not info.get("matched"):
                huerfanos.append({
                    "transcript": info["transcript"],
                    "motivo": "fuente no esta en disco: " + info.get("source_path_declarado", "?"),
                })

    sin_cubrir = [r for r in rows if r["cubierto"] == "NO"]
    sin_extraer = [r for r in rows if r["cubierto"] == "si" and r["extraido"] == "no"]
    vacios = [r for r in rows if r["cubierto"] == "si" and r["items"] == 0]
    total_b = sum(max(0, r["bytes"]) for r in rows)

    L = []
    L.append("# INVENTARIO de fuentes")
    L.append("")
    L.append("Generado por `scripts/manifest.py`. El detalle fila por fila esta en")
    L.append("`data/processed/manifest_fuentes.csv`, una fila por archivo.")
    L.append("")
    L.append("**Total en disco:** %d archivos, %s" % (len(rows), gb(total_b)))
    L.append("")
    L.append("| Tipo | Archivos | Tamano | Cubiertos | Tamano cubierto |")
    L.append("|---|---|---|---|---|")
    for k in sorted(stats):
        s = stats[k]
        L.append("| %s | %d | %s | %d | %s |" % (k, s["n"], gb(s["bytes"]), s["cub"], gb(s["cub_bytes"])))
    L.append("")
    L.append("## Archivos SIN cubrir")
    L.append("")
    L.append("Estos no tienen transcript ni OCR. El motivo de cada uno esta al lado,")
    L.append("y es lo que decide si borrar el crudo pierde algo: un video sin pista de")
    L.append("audio no tiene habla que perder, y un transcript excluido a proposito")
    L.append("tampoco es una perdida. Solo los marcados `HUECO` o `ilegible` son")
    L.append("contenido que se pierde sin forma de recuperarlo.")
    L.append("")
    if sin_cubrir:
        L.append("**%d archivos, %s — de los cuales %d %s hueco%s real%s**"
                 % (len(sin_cubrir),
                    gb(sum(max(0, r["bytes"]) for r in sin_cubrir)),
                    n_huecos,
                    "es" if n_huecos == 1 else "son",
                    "" if n_huecos == 1 else "s",
                    "" if n_huecos == 1 else "es"))
        if n_dur:
            L.append("")
            L.append("(%d archivo(s) mas matchearon por duracion: estan "
                     "renombrados en disco respecto del `source_path` del "
                     "transcript, pero su contenido YA esta en el corpus. "
                     "Columna `via` del CSV.)" % n_dur)
        L.append("")
        for r in sorted(sin_cubrir, key=lambda r: -max(0, r["bytes"]))[:200]:
            L.append("- `%s` (%s, %s) — %s" % (
                r["path"], gb(max(0, r["bytes"])), r["tipo"], r.get("motivo", "")))
        if len(sin_cubrir) > 200:
            L.append("- ... y %d mas, ver el CSV" % (len(sin_cubrir) - 200))
    else:
        L.append("Ninguno. Todo el material crudo tiene transcript u OCR.")
    L.append("")
    L.append("## Con transcript pero sin extraccion")
    L.append("")
    if sin_extraer:
        L.append("%d archivos. El texto ya esta guardado en `data/processed/transcripts/`;" % len(sin_extraer))
        L.append("falta destilarlo. Para eso no hace falta el archivo original.")
    else:
        L.append("Ninguno.")
    L.append("")
    L.append("## Con extraccion vacia (0 items)")
    L.append("")
    if vacios:
        L.append("%d archivos devolvieron 0 principios, 0 situaciones y 0 frases." % len(vacios))
        L.append("En audios cortos eso suele ser legitimo. Re-extraer no necesita el")
        L.append("archivo original, solo el transcript.")
    else:
        L.append("Ninguno.")
    L.append("")
    L.append("## Formatos que el inventario no clasifica")
    L.append("")
    L.append("El recorrido solo cuenta las extensiones de MEDIA_EXT. Lo que sigue")
    L.append("quedo afuera: puede ser basura, o puede ser material real en un formato")
    L.append("que el pipeline todavia no lee. Conviene revisarlo antes de borrar nada.")
    L.append("")
    if ignorados:
        tot_n = sum(v["n"] for v in ignorados.values())
        tot_b = sum(v["bytes"] for v in ignorados.values())
        L.append("**%d archivos, %s**" % (tot_n, gb(tot_b)))
        L.append("")
        L.append("| Extension | Archivos | Tamano | Ejemplo |")
        L.append("|---|---|---|---|")
        for ext, v in sorted(ignorados.items(), key=lambda kv: -kv[1]["bytes"]):
            ej = v["ej"][0] if v["ej"] else ""
            L.append("| `%s` | %d | %s | `%s` |" % (ext or "(sin extension)", v["n"], gb(v["bytes"]), ej))
    else:
        L.append("Ninguno.")
    L.append("")
    L.append("## Transcripts cuya fuente ya no esta en disco")
    L.append("")
    if huerfanos:
        L.append("%d transcripts. El texto esta a salvo; el original no." % len(huerfanos))
        for h in huerfanos[:50]:
            L.append("- `%s` - %s" % (h["transcript"], h["motivo"]))
        if len(huerfanos) > 50:
            L.append("- ... y %d mas" % (len(huerfanos) - 50))
    else:
        L.append("Ninguno.")
    L.append("")
    (REPO / "INVENTARIO.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Escrito %s" % (REPO / "INVENTARIO.md"))
    print("Sin cubrir: %d archivos (%d huecos reales)" % (len(sin_cubrir), n_huecos))


if __name__ == "__main__":
    main()
