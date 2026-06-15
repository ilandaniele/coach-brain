"""Descarga de carpeta pública de Google Drive con gdown."""

from __future__ import annotations

from pathlib import Path

import gdown
import typer
from rich.console import Console

from coach_brain.settings import settings

console = Console()


EXT_MAP: dict[str, set[str]] = {
    "pdfs": {".pdf"},
    "videos": {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"},
    "audios": {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"},
    "screenshots": {".png", ".jpg", ".jpeg", ".webp"},
}


def detect_subdir(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    for subdir, exts in EXT_MAP.items():
        if ext in exts:
            return subdir
    return None


def download(folder_url: str) -> None:
    """Baja la carpeta de Drive y clasifica archivos por extensión en data/raw/*."""
    tmp_dir = settings.raw_path / "_drive_dump"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Bajando carpeta de Drive...[/cyan]")
    console.print(f"[dim]URL: {folder_url}[/dim]")
    console.print(f"[dim]Dump temporal: {tmp_dir}[/dim]\n")

    paths = gdown.download_folder(
        url=folder_url,
        output=str(tmp_dir),
        quiet=False,
        use_cookies=False,
    )

    if not paths:
        console.print("[red]gdown no devolvió archivos. ¿La carpeta es pública?[/red]")
        return

    moved = 0
    skipped = 0
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        subdir = detect_subdir(path.name)
        if subdir is None:
            console.print(
                f"[yellow]Extensión no reconocida ({path.suffix}), queda en _drive_dump: "
                f"{path.name}[/yellow]"
            )
            skipped += 1
            continue
        dest = settings.raw_path / subdir / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            console.print(f"[dim]ya existe: {dest}[/dim]")
            skipped += 1
            continue
        path.rename(dest)
        moved += 1
        console.print(f"[green]→[/green] {subdir}/{path.name}")

    console.print(
        f"\n[green]Listo[/green] — movidos={moved} skippeados={skipped}"
    )


if __name__ == "__main__":
    typer.run(download)
