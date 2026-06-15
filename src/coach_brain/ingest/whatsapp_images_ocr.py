"""OCR local (RapidOCR) sobre las imágenes del export de WhatsApp.

Alternativa GRATIS y determinista a la vision con Haiku: la mayoría de las
imágenes son screenshots de chats (texto), así que con OCR local sacamos el
contenido sin gastar API, sin rate limits y SIEMPRE devuelve algo parseable.

RapidOCR devuelve, por línea, la caja (4 puntos) + texto + confianza. Usamos la
coordenada X del borde izquierdo para inferir lado (YO = derecha / OTRO =
izquierda), y filtramos ruido de UI (hora, batería, "Escribe un mensaje", etc.).
El etiquetado de hablante es aproximado; el texto crudo es lo fiable.

Resumable: un JSON por imagen en data/processed/whatsapp_images_ocr/.
`aggregate()` junta los chats en transcripts/whatsapp_images_chats.json (mismo
target que la versión vision, así el resto del pipeline no cambia).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import typer
from PIL import Image
from rich.console import Console
from rich.progress import track

from coach_brain.settings import settings

console = Console()

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Líneas que son chrome de WhatsApp / iOS, no contenido del chat.
_NOISE_RE = re.compile(
    r"^\s*("
    r"\d{1,2}:\d{2}(\s*[ap]\.?\s*m\.?)?"          # hora 8:33 / 2:11 a.m.
    r"|\d{1,3}\s*%"                                  # batería 29%
    r"|enviar|enviado|entregado|le[ií]do|visto"
    r"|escrib[ie].*mensaje"
    r"|en l[ií]nea|online|escribiendo\.{0,3}"
    r"|hoy|ayer|previa|lun|mar|mi[eé]r|jue|vie|s[áa]b|dom"
    r"|[<>«»·∙•|=三▲►◄√x×#@]+"                       # glifos/iconos sueltos
    r")\s*$",
    re.IGNORECASE,
)


def _is_noise(txt: str) -> bool:
    t = txt.strip()
    if len(t) <= 1:
        return True
    if _NOISE_RE.match(t):
        return True
    # Pura puntuación/dígitos cortos
    if len(t) <= 3 and not any(c.isalpha() for c in t):
        return True
    return False


def _ocr_image(ocr, path: Path) -> dict:
    """OCR de una imagen → dict con transcripcion (YO/OTRO aprox) + texto crudo."""
    try:
        W = Image.open(path).size[0]
    except Exception:  # noqa: BLE001
        return {"es_chat": False, "transcripcion": "", "texto_crudo": "",
                "descripcion": "no se pudo abrir", "engine": "rapidocr"}

    res, _ = ocr(str(path))
    if not res:
        return {"es_chat": False, "transcripcion": "", "texto_crudo": "",
                "descripcion": "sin texto detectado", "engine": "rapidocr"}

    # Ordenar arriba->abajo por Y del borde superior.
    items = sorted(res, key=lambda r: r[0][0][1])
    raw_lines: list[str] = []
    turns: list[tuple[str, str]] = []  # (lado, texto)
    for box, txt, _conf in items:
        txt = (txt or "").strip()
        if not txt:
            continue
        raw_lines.append(txt)
        if _is_noise(txt):
            continue
        left_x = box[0][0]
        lado = "YO" if left_x > W * 0.5 else "OTRO"
        if turns and turns[-1][0] == lado:
            turns[-1] = (lado, turns[-1][1] + " " + txt)
        else:
            turns.append((lado, txt))

    texto_crudo = "\n".join(raw_lines)
    transcripcion = "\n".join(f"{lado}: {t}" for lado, t in turns)
    # Heurística es_chat: varias líneas de contenido real con letras.
    content_lines = [t for _, t in turns if any(c.isalpha() for c in t)]
    es_chat = len(content_lines) >= 3
    return {
        "es_chat": es_chat,
        "transcripcion": transcripcion,
        "texto_crudo": texto_crudo,
        "descripcion": "screenshot de chat (OCR)" if es_chat else "imagen con poco texto (OCR)",
        "engine": "rapidocr",
    }


def run(limit: int = 0) -> None:
    from rapidocr_onnxruntime import RapidOCR

    wa_dir = settings.raw_path / "whatsapp"
    if not wa_dir.exists():
        console.print(f"[yellow]No existe {wa_dir}[/yellow]")
        return

    images = sorted(p for p in wa_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    out_dir = settings.processed_path / "whatsapp_images_ocr"
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = [p for p in images if not (out_dir / f"{p.stem}.json").exists()]
    if limit > 0:
        pending = pending[:limit]
    console.print(f"[cyan]Imágenes: {len(images)} total, {len(pending)} pendientes (OCR local)[/cyan]")
    if not pending:
        return

    ocr = RapidOCR()
    counts = {"chat": 0, "other": 0, "err": 0}
    total = len(pending)
    for i, p in enumerate(pending, 1):
        out_path = out_dir / f"{p.stem}.json"
        try:
            data = _ocr_image(ocr, p)
            data["source_file"] = p.name
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            counts["chat" if data["es_chat"] else "other"] += 1
        except Exception as e:  # noqa: BLE001
            counts["err"] += 1
            console.print(f"[red]err {p.name}: {type(e).__name__}[/red]")
        # Progreso en texto plano (se ve en el log/ventana, no como barra rich).
        if i % 25 == 0 or i == total:
            console.print(f"OCR {i}/{total} — {counts}", highlight=False)
    console.print(f"[green]Listo[/green] — {counts}")


def aggregate() -> None:
    """Junta las imágenes-chat (OCR) en un transcript para el pipeline."""
    out_dir = settings.processed_path / "whatsapp_images_ocr"
    results = sorted(out_dir.glob("*.json"))
    chats = []
    for r in results:
        try:
            d = json.loads(r.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if d.get("es_chat") and d.get("transcripcion", "").strip():
            chats.append(d)

    if not chats:
        console.print("[yellow]No hay imágenes-chat (OCR) para agregar todavía.[/yellow]")
        return

    parts, segments = [], []
    for i, c in enumerate(chats):
        body = c["transcripcion"].strip()
        parts.append(f"### Screenshot: {c.get('source_file', '')}\n{body}")
        segments.append({"id": i, "text": body, "source_file": c.get("source_file"),
                         "descripcion": c.get("descripcion"), "page": None})

    data = {
        "source_id": "whatsapp_images_chats",
        "source_type": "whatsapp_images",
        "source_path": "data/raw/whatsapp",
        "title": "Screenshots de chats (WhatsApp) — OCR local",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": "es",
        "metadata": {"chats": len(chats), "total_imagenes_procesadas": len(results), "engine": "rapidocr"},
        "segments": segments,
        "full_text": "\n\n".join(parts),
    }
    out_path = settings.transcripts_path / "whatsapp_images_chats.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/green] {len(chats)} chats de {len(results)} imágenes -> {out_path.name}")


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(run)
    app.command()(aggregate)
    app()
