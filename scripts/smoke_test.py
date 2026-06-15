"""Smoke test: verifica que todo el stack esté funcional antes de correr el pipeline."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

QUICK = "--quick" in sys.argv or os.environ.get("SMOKE_QUICK") == "1"


def check(label: str, fn) -> tuple[bool, str]:
    try:
        msg = fn() or "ok"
        return True, str(msg)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    # 1. Imports
    def _imports():
        import anthropic, faster_whisper, pypdf, qdrant_client, sentence_transformers, streamlit  # noqa: F401
        return f"anthropic, faster_whisper, pypdf, qdrant_client, sentence_transformers, streamlit"

    results.append(("imports", *check("imports", _imports)))

    # 2. Settings + .env
    def _settings():
        from coach_brain.settings import settings

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY no está en .env")
        return f"API key: {settings.anthropic_api_key[:10]}...  model: {settings.model_main}"

    results.append(("settings & .env", *check("settings", _settings)))

    # 3. ffmpeg
    def _ffmpeg():
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg no está en PATH")
        return shutil.which("ffmpeg")

    results.append(("ffmpeg", *check("ffmpeg", _ffmpeg)))

    # 4. CUDA / GPU
    def _cuda():
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() == False")
        return f"{torch.cuda.get_device_name(0)} ({torch.version.cuda})"

    results.append(("CUDA", *check("cuda", _cuda)))

    # 5. Anthropic API ping (rápido)
    def _anthropic():
        from anthropic import Anthropic

        from coach_brain.settings import settings

        client = Anthropic(api_key=settings.anthropic_api_key)
        r = client.messages.create(
            model=settings.model_fast,
            max_tokens=20,
            messages=[{"role": "user", "content": "Decí 'ok' y nada más."}],
        )
        text = "".join(b.text for b in r.content if hasattr(b, "text"))
        return f"response: {text.strip()[:30]}"

    results.append(("Anthropic API ping", *check("anthropic", _anthropic)))

    # 6. Embedding model (descarga si hace falta — puede tardar la 1ra vez)
    if QUICK:
        results.append(("Embeddings (bge-m3)", True, "skipped (--quick)"))
    else:
        def _embed():
            from coach_brain.kb.embed import embed_dim, embed_one

            _ = embed_one("hola mundo")
            return f"bge-m3 carga ok, dim={embed_dim()}"

        results.append(("Embeddings (bge-m3)", *check("embed", _embed)))

    # 7. Qdrant local
    def _qdrant():
        from coach_brain.kb.index import get_qdrant

        client = get_qdrant()
        collections = client.get_collections()
        return f"qdrant ok, collections={len(collections.collections)}"

    results.append(("Qdrant", *check("qdrant", _qdrant)))

    # Render
    table = Table(title="Smoke test")
    table.add_column("Check")
    table.add_column("OK")
    table.add_column("Detalle", overflow="fold")
    for name, ok, msg in results:
        table.add_row(name, "[green]OK[/green]" if ok else "[red]FAIL[/red]", msg)
    console.print(table)

    fails = [r for r in results if not r[1]]
    if fails:
        console.print(f"\n[red]{len(fails)} check(s) fallaron.[/red]")
        return 1
    console.print("\n[green]Todo en orden.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
