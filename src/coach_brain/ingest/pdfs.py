"""Ingestor de PDFs → JSON transcripts unificados."""

from __future__ import annotations

import json
import os
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


_OCR_ENGINE = None

# Debajo de este promedio de caracteres por pagina se considera que la capa de
# texto del PDF no alcanza y vale la pena rasterizar. Un PDF de texto real
# ronda los 1500-3000 por pagina; una exportacion de Canva o una presentacion
# hecha de imagenes cae bajo los 300.
OCR_MIN_CHARS_PER_PAGE = 400


def _ocr_pages(path: Path, dpi: int = 200) -> list[str]:
    """Rasteriza cada pagina y le pasa OCR local (RapidOCR).

    Es el ultimo recurso, para los PDFs escaneados: pypdf y pymupdf no
    devuelven nada porque el archivo son imagenes, no texto. Cuesta varios
    segundos por pagina, asi que solo corre con --ocr.
    """
    global _OCR_ENGINE

    import numpy as np
    import pymupdf  # type: ignore

    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_ENGINE = RapidOCR()

    zoom = dpi / 72.0
    out: list[str] = []
    doc = pymupdf.open(path)
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            if pix.n == 4:  # RGBA -> RGB, RapidOCR no acepta canal alfa
                img = img[:, :, :3]
            result, _ = _OCR_ENGINE(img)
            lines = [r[1] for r in (result or []) if r and len(r) > 1 and r[1]]
            out.append("\n".join(lines).strip())
    finally:
        doc.close()
    return out


def extract_pdf(path: Path, ocr: bool = False) -> dict:
    """Extrae texto de un PDF: pypdf, luego pymupdf, y con --ocr rasteriza."""
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

    engine = "texto"
    # El OCR no se dispara solo cuando no hay NADA de texto. Un PDF puede traer
    # una capa de texto casi vacia y seguir siendo ilegible: una exportacion de
    # Canva, por ejemplo, trae los titulos de los cuadros pero el contenido es
    # imagen. Por eso tambien se rasteriza cuando el texto extraido es
    # sospechosamente pobre para la cantidad de paginas.
    chars = sum(len(seg["text"]) for seg in segments)
    thin = n_pages > 0 and chars < OCR_MIN_CHARS_PER_PAGE * n_pages
    if ocr and (not segments or thin):
        console.print(
            f"[cyan]OCR sobre {n_pages} páginas de {path.name} "
            f"({chars} chars de capa de texto)...[/cyan]"
        )
        try:
            by_page = {seg["page"]: seg for seg in segments}
            used = False
            for i, text in enumerate(_ocr_pages(path)):
                if not text:
                    continue
                prev = by_page.get(i + 1)
                # Se queda el que mas texto trae. Si la pagina ya tenia una capa
                # de texto buena, el OCR no la puede empeorar.
                if prev is not None and len(prev["text"]) >= len(text):
                    continue
                by_page[i + 1] = {
                    "id": i,
                    "text": text,
                    "start": None,
                    "end": None,
                    "page": i + 1,
                }
                used = True
            if used:
                segments = [by_page[page] for page in sorted(by_page)]
                full_text_parts = [seg["text"] for seg in segments]
                engine = "rapidocr" if chars == 0 else "texto+rapidocr"
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]OCR falló en {path.name}: {e}[/red]")

    return {
        "source_id": f"pdf_{path.stem}",
        "source_type": "pdf",
        "source_path": _relpath(path),
        "title": path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": settings.whisper_language,
        "metadata": {"pages": n_pages, "duration_seconds": None, "engine": engine},
        "segments": segments,
        "full_text": "\n\n".join(full_text_parts),
    }


def run(force: bool = False, ocr: bool = False) -> None:
    # `pdfs` es una junction al material; `whatsapp` es un directorio real y
    # tambien tiene PDFs, los que llegaron adjuntos en los chats. Sin esta
    # segunda raiz esos adjuntos nunca entraban al pipeline.
    roots = [settings.raw_path / "pdfs", settings.raw_path / "whatsapp"]
    out_dir = settings.transcripts_path
    out_dir.mkdir(parents=True, exist_ok=True)

    # Las junctions de data/raw pueden apuntar al mismo arbol, asi que
    # deduplicamos por path canonico antes de procesar.
    found: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() != ".pdf":
                continue
            found.setdefault(os.path.normcase(os.path.realpath(str(p))), p)
    pdfs = sorted(found.values())
    if not pdfs:
        console.print(f"[yellow]No hay PDFs en {', '.join(str(r) for r in roots)}[/yellow]")
        return

    console.print(f"[cyan]Procesando {len(pdfs)} PDF(s)...[/cyan]")
    ok = skip = err = empty = 0

    for pdf_path in track(pdfs, description="PDFs"):
        out_path = out_dir / f"pdf_{pdf_path.stem}.json"
        if out_path.exists() and not force:
            skip += 1
            continue
        try:
            data = extract_pdf(pdf_path, ocr=ocr)
            if not data["segments"]:
                console.print(
                    f"[yellow]Sin texto extraíble (¿escaneado?): {pdf_path.name}"
                    f"{'' if ocr else ' — reintentá con --ocr'}[/yellow]"
                )
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
