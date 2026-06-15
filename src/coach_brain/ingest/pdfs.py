"""Ingestor de PDFs → JSON transcripts unificados."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from pypdf import PdfReader
from rich.console import Console
from rich.progress import track

from coach_brain.settings import REPO_ROOT, settings

console = Console()


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def extract_pdf(path: Path) -> dict:
    """Extrae texto de un PDF, con fallback a pymupdf si pypdf no obtiene texto."""
    reader = PdfReader(path)
    n_pages = len(reader.pages)
    segments: list[dict] = []
    full_text_parts: list[str] = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            segments.append({"id": i, "text": text, "start": None, "end": None, "page": i + 1})
            full_text_parts.append(text)

    if not segments:
        # Fallback con pymupdf (mejor con layouts raros)
        try:
            import pymupdf  # type: ignore

            doc = pymupdf.open(path)
            for i, page in enumerate(doc):
                text = (page.get_text() or "").strip()
                if text:
                    segments.append(
                        {"id": i, "text": text, "start": None, "end": None, "page": i + 1}
                    )
                    full_text_parts.append(text)
            doc.close()
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]pymupdf fallback falló: {e}[/yellow]")

    return {
        "source_id": f"pdf_{path.stem}",
        "source_type": "pdf",
        "source_path": _relpath(path),
        "title": path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": settings.whisper_language,
        "metadata": {"pages": n_pages, "duration_seconds": None},
        "segments": segments,
        "full_text": "\n\n".join(full_text_parts),
    }


def run(force: bool = False) -> None:
    pdf_dir = settings.raw_path / "pdfs"
    out_dir = settings.transcripts_path
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in pdf_dir.rglob("*") if p.suffix.lower() == ".pdf")
    if not pdfs:
        console.print(f"[yellow]No hay PDFs en {pdf_dir}[/yellow]")
        return

    console.print(f"[cyan]Procesando {len(pdfs)} PDF(s)...[/cyan]")
    ok = skip = err = empty = 0

    for pdf_path in track(pdfs, description="PDFs"):
        out_path = out_dir / f"pdf_{pdf_path.stem}.json"
        if out_path.exists() and not force:
            skip += 1
            continue
        try:
            data = extract_pdf(pdf_path)
            if not data["segments"]:
                console.print(f"[yellow]Sin texto extraíble (¿escaneado?): {pdf_path.name}[/yellow]")
                empty += 1
                continue
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            ok += 1
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error en {pdf_path.name}: {e}[/red]")
            err += 1

    console.print(
        f"[green]Listo[/green] — ok={ok} skip={skip} sin_texto={empty} errores={err}"
    )


if __name__ == "__main__":
    typer.run(run)
