"""Ingestor de documentos (.docx, .epub, .pptx) → JSON transcripts unificados."""

from __future__ import annotations

import html
import json
import os
import posixpath
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track

from coach_brain.settings import REPO_ROOT, settings

console = Console()

DOCX_EXT = {".docx"}
EPUB_EXT = {".epub"}
PPTX_EXT = {".pptx"}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\f\v]+")


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def extract_docx(path: Path) -> list[str]:
    """Devuelve párrafos no vacíos de un .docx (incluye celdas de tablas)."""
    from docx import Document

    doc = Document(str(path))
    parts: list[str] = []
    for para in doc.paragraphs:
        t = _clean(para.text)
        if t:
            parts.append(t)
    for table in doc.tables:
        for row in table.rows:
            cells = [_clean(c.text) for c in row.cells]
            line = " | ".join(c for c in cells if c)
            if line:
                parts.append(line)
    return parts


def extract_epub(path: Path) -> list[str]:
    """Devuelve bloques de texto (uno por capítulo) de un .epub."""
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path))
    parts: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        raw = item.get_content().decode("utf-8", errors="ignore")
        text = html.unescape(_TAG_RE.sub(" ", raw))
        text = "\n".join(_clean(line) for line in text.splitlines())
        text = re.sub(r"\n{2,}", "\n\n", text).strip()
        if text:
            parts.append(text)
    return parts


_SLIDE_RE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")
_NOTES_TARGET_RE = re.compile(r'Target="([^"]*notesSlide\d+\.xml)"')
_RUN_RE = re.compile(r"<a:t>(.*?)</a:t>", re.S)
_PARA_END_RE = re.compile(r"</a:p>")


def _pptx_slide_text(xml: str) -> str:
    """Texto de una diapositiva, un parrafo por linea.

    Un .pptx es un zip de XML: cada corrida de texto es un <a:t> y cada
    parrafo un <a:p>. Se parte por parrafo para no pegar el titulo con el
    cuerpo, y se concatenan las corridas dentro de cada parrafo porque
    PowerPoint parte una misma frase en varias cuando cambia el formato.
    """
    lines: list[str] = []
    for chunk in _PARA_END_RE.split(xml):
        runs = [html.unescape(m.group(1)) for m in _RUN_RE.finditer(chunk)]
        line = _clean("".join(runs))
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_pptx(path: Path) -> list[str]:
    """Devuelve un bloque de texto por diapositiva de un .pptx.

    Las notas del orador van pegadas a su diapositiva, prefijadas con
    "Notas:", porque suelen tener el desarrollo de lo que el titulo resume.
    No se usa python-pptx a proposito: el formato es XML en un zip y el
    stdlib alcanza, igual que para el .epub.
    """
    parts: list[str] = []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        slides = {}
        for name in names:
            m = _SLIDE_RE.match(name)
            if m:
                slides[int(m.group(1))] = name
        for idx in sorted(slides):
            slide = slides[idx]
            text = _pptx_slide_text(z.read(slide).decode("utf-8", errors="ignore"))
            ntext = ""
            rels = f"ppt/slides/_rels/{slide.rsplit('/', 1)[1]}.rels"
            if rels in names:
                m = _NOTES_TARGET_RE.search(z.read(rels).decode("utf-8", errors="ignore"))
                if m:
                    target = posixpath.normpath(
                        posixpath.join("ppt/slides", m.group(1))
                    )
                    if target in names:
                        raw = z.read(target).decode("utf-8", errors="ignore")
                        ntext = _pptx_slide_text(raw)
            if ntext:
                text = (text + "\n\nNotas: " + ntext).strip()
            if text:
                parts.append(text)
    return parts


_EXTRACTORS = {
    "docx": lambda p: extract_docx(p),
    "epub": lambda p: extract_epub(p),
    "pptx": lambda p: extract_pptx(p),
}


def build_record(path: Path, source_type: str, parts: list[str]) -> dict:
    segments = [
        {"id": i, "text": t, "start": None, "end": None, "page": None}
        for i, t in enumerate(parts)
    ]
    return {
        "source_id": f"{source_type}_{path.stem}",
        "source_type": source_type,
        "source_path": _relpath(path),
        "title": path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": settings.whisper_language,
        "metadata": {"pages": None, "duration_seconds": None, "blocks": len(parts)},
        "segments": segments,
        "full_text": "\n\n".join(parts),
    }


def _collect_targets() -> list[tuple[Path, str]]:
    """Documentos a ingestar, de todos los roots donde puede haber.

    Hay que mirar `data/raw/whatsapp` además de `data/raw/docs`, porque hubo
    material que llegó como adjunto de WhatsApp y esta etapa nunca lo vio.
    Los tres junctions de `data/raw` apuntan al mismo árbol, así que se
    deduplica por path canónico o se procesa todo varias veces.
    """
    roots = [settings.raw_path / "docs", settings.raw_path / "whatsapp"]
    by_real: dict[str, tuple[Path, str]] = {}
    for base in roots:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            suf = p.suffix.lower()
            if suf in DOCX_EXT:
                kind = "docx"
            elif suf in EPUB_EXT:
                kind = "epub"
            elif suf in PPTX_EXT:
                kind = "pptx"
            else:
                continue
            key = os.path.normcase(os.path.realpath(str(p)))
            by_real.setdefault(key, (p, kind))
    return sorted(by_real.values(), key=lambda t: t[0].name)


def run(force: bool = False) -> None:
    targets = _collect_targets()
    if not targets:
        console.print(f"[yellow]No hay .docx/.epub en {settings.raw_path / 'docs'}[/yellow]")
        return

    out_dir = settings.transcripts_path
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Procesando {len(targets)} documento(s)...[/cyan]")
    ok = skip = err = empty = 0

    for path, source_type in track(targets, description="Docs"):
        out_path = out_dir / f"{source_type}_{path.stem}.json"
        if out_path.exists() and not force:
            skip += 1
            continue
        try:
            parts = _EXTRACTORS[source_type](path)
            if not parts:
                console.print(f"[yellow]Sin texto extraíble: {path.name}[/yellow]")
                empty += 1
                continue
            data = build_record(path, source_type, parts)
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            ok += 1
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error en {path.name}: {e}[/red]")
            err += 1

    console.print(
        f"[green]Listo[/green] — ok={ok} skip={skip} sin_texto={empty} errores={err}"
    )


if __name__ == "__main__":
    typer.run(run)
