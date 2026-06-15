"""Vision sobre las imágenes del export de WhatsApp (Haiku).

Muchas son screenshots de chats reales de Ilan (textgame en vivo) que le pasaba
a la coach. Cada imagen se manda a Haiku: si es un chat, transcribe los mensajes
con quién habla; si no, una descripción corta. Resumable (un JSON por imagen),
con downscale para abaratar tokens y backoff ante rate limits.

Después `aggregate()` junta las que son chat en un transcript para el pipeline.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import typer
from PIL import Image
from rich.console import Console

from coach_brain.settings import settings

console = Console()

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_DIM = 1568  # recomendado por Anthropic; más grande no mejora y cuesta más

PROMPT = """\
Esta imagen viene de un export de WhatsApp de un coaching de seducción/textgame.
Puede ser: (a) un SCREENSHOT de una conversación de chat (con otra persona), o
(b) otra cosa (foto, meme, captura suelta, imagen de perfil).

Devolvé SOLO un JSON válido, sin texto extra, con esta forma:
{
  "es_chat": true|false,
  "transcripcion": "si es_chat: los mensajes en orden, prefijando 'EL:' o 'ELLA:' (o 'YO:'/'OTRO:' si no se distingue género). Si no es chat: \\"\\"",
  "descripcion": "una línea describiendo qué es (útil siempre)"
}
Sé fiel al texto del chat, no inventes. Si la imagen no tiene texto legible, es_chat=false."""


def _encode(path: Path) -> tuple[str, str] | None:
    """Downscalea y devuelve (media_type, base64). None si no se puede abrir."""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        if max(img.size) > MAX_DIM:
            img.thumbnail((MAX_DIM, MAX_DIM))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "image/jpeg", base64.standard_b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].lstrip("json").strip() if "```" in text else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"es_chat": False, "transcripcion": "", "descripcion": text[:200]}


def _process_one(client, model: str, path: Path, out_dir: Path) -> str:
    out_path = out_dir / f"{path.stem}.json"
    if out_path.exists():
        return "skip"
    enc = _encode(path)
    if enc is None:
        return "err_open"
    media_type, b64 = enc

    for attempt in range(5):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
            )
            parsed = _parse_json(resp.content[0].text)
            parsed["source_file"] = path.name
            out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            return "chat" if parsed.get("es_chat") else "other"
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "overload" in msg or "529" in msg:
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            return f"err:{type(e).__name__}"
    return "err_ratelimit"


def run(workers: int = 4, limit: int = 0) -> None:
    from anthropic import Anthropic

    wa_dir = settings.raw_path / "whatsapp"
    if not wa_dir.exists():
        console.print(f"[yellow]No existe {wa_dir}[/yellow]")
        return

    images = sorted(p for p in wa_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        console.print("[yellow]No hay imágenes en data/raw/whatsapp/[/yellow]")
        return

    out_dir = settings.processed_path / "whatsapp_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = [p for p in images if not (out_dir / f"{p.stem}.json").exists()]
    if limit > 0:
        pending = pending[:limit]
    console.print(f"[cyan]Imágenes: {len(images)} total, {len(pending)} pendientes (workers={workers})[/cyan]")

    client = Anthropic(api_key=settings.anthropic_api_key)
    model = settings.model_fast

    counts: dict[str, int] = {}
    lock = threading.Lock()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_process_one, client, model, p, out_dir): p for p in pending}
        for fut in as_completed(futs):
            res = fut.result()
            key = res if res in ("chat", "other", "skip") or res.startswith("err") else "other"
            with lock:
                counts[key] = counts.get(key, 0) + 1
                done += 1
                if done % 100 == 0:
                    console.print(f"  {done}/{len(pending)} — {counts}")

    console.print(f"[green]Listo[/green] — {counts}")


def aggregate() -> None:
    """Junta las imágenes que son chat en un transcript para el pipeline."""
    out_dir = settings.processed_path / "whatsapp_images"
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
        console.print("[yellow]No hay imágenes-chat para agregar todavía.[/yellow]")
        return

    parts = []
    segments = []
    for i, c in enumerate(chats):
        header = f"### Screenshot: {c.get('source_file', '')}"
        body = c["transcripcion"].strip()
        parts.append(f"{header}\n{body}")
        segments.append({"id": i, "text": body, "source_file": c.get("source_file"),
                         "descripcion": c.get("descripcion"), "page": None})

    full_text = "\n\n".join(parts)
    data = {
        "source_id": "whatsapp_images_chats",
        "source_type": "whatsapp_images",
        "source_path": "data/raw/whatsapp",
        "title": "Screenshots de chats (WhatsApp) — vision",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": "es",
        "metadata": {"chats": len(chats), "total_imagenes_procesadas": len(results)},
        "segments": segments,
        "full_text": full_text,
    }
    out_path = settings.transcripts_path / "whatsapp_images_chats.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/green] {len(chats)} chats de {len(results)} imágenes -> {out_path.name}")


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(run)
    app.command()(aggregate)
    app()
