"""Smoke test: verifica que todo el stack esté funcional antes de correr el pipeline."""

from __future__ import annotations

import importlib
import os
import shutil
import sys

from rich.console import Console
from rich.table import Table

console = Console()

QUICK = "--quick" in sys.argv or os.environ.get("SMOKE_QUICK") == "1"

# Los seis paquetes que el pipeline necesita para arrancar. El chequeo importa
# uno por uno y nombra los que fallan: antes importaba solo streamlit y el
# informe listaba los seis igual, asi que faltar faster-whisper daba OK verde.
PAQUETES = (
    "anthropic",
    "faster_whisper",
    "pypdf",
    "qdrant_client",
    "sentence_transformers",
    "streamlit",
)


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
        faltan = []
        for nombre in PAQUETES:
            try:
                importlib.import_module(nombre)
            except Exception as e:  # noqa: BLE001
                faltan.append(f"{nombre} ({type(e).__name__}: {e})")
        if faltan:
            raise RuntimeError("no se pudieron importar: " + "; ".join(faltan))
        return ", ".join(PAQUETES)

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
        from coach_brain.kb.index import close_qdrant, get_qdrant

        # El cierre es explícito a propósito: get_qdrant() cachea el cliente, y
        # si se lo deja al __del__ del intérprete el script termina imprimiendo
        # cinco líneas de "Exception ignored in" (ver close_qdrant). Va en
        # finally para que también cierre cuando get_collections() falle.
        client = get_qdrant()
        try:
            collections = client.get_collections()
        finally:
            close_qdrant()
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
