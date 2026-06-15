"""
Migra vectores del Qdrant embebido local → Qdrant Cloud.

Uso:
    python scripts/upload_to_qdrant_cloud.py

Requiere en .env:
    QDRANT_URL     = https://xxxxxxxx.qdrant.io:6333
    QDRANT_API_KEY = tu_api_key_de_qdrant_cloud

Lee de QDRANT_PATH (./qdrant_storage por defecto).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rich.console import Console

from coach_brain.settings import settings

console = Console()

COLLECTIONS = ["principles", "situations", "style_phrases"]
BATCH = 200


def main() -> None:
    if not settings.qdrant_url:
        console.print("[red]QDRANT_URL no configurada. Ponela en .env y volvé a correr.[/red]")
        sys.exit(1)

    local_path = settings.qdrant_storage
    if not local_path.exists():
        console.print(f"[red]Storage local no existe: {local_path}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Local :[/cyan] {local_path}")
    console.print(f"[cyan]Cloud :[/cyan] {settings.qdrant_url}")
    if not settings.qdrant_api_key:
        console.print("[yellow]QDRANT_API_KEY vacía (OK si Qdrant sin auth).[/yellow]")

    local = QdrantClient(path=str(local_path))

    cloud_kwargs: dict = {"url": settings.qdrant_url}
    if settings.qdrant_api_key:
        cloud_kwargs["api_key"] = settings.qdrant_api_key
    cloud = QdrantClient(**cloud_kwargs)

    for col in COLLECTIONS:
        console.print(f"\n[bold]── {col}[/bold]")

        try:
            info = local.get_collection(col)
        except Exception as e:
            console.print(f"  [yellow]No existe localmente, skip: {e}[/yellow]")
            continue

        dim = info.config.params.vectors.size
        total = info.points_count

        try:
            cloud.get_collection(col)
            console.print(f"  Colección ya existe en cloud (dim={dim})")
        except Exception:
            cloud.create_collection(
                collection_name=col,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            console.print(f"  [green]Creada en cloud[/green] (dim={dim})")

        console.print(f"  Subiendo {total} vectores...")
        offset = None
        uploaded = 0

        while True:
            points, next_offset = local.scroll(
                collection_name=col,
                limit=BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break

            cloud.upsert(
                collection_name=col,
                points=[PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                        for p in points],
            )
            uploaded += len(points)
            console.print(f"  {uploaded}/{total}", end="\r")

            if next_offset is None:
                break
            offset = next_offset

        console.print(f"  [green]✓ {uploaded} vectores subidos[/green]")

    console.print("\n[bold green]Migración completa.[/bold green]")
    console.print(
        "Siguiente: actualizá .env con QDRANT_URL + QDRANT_API_KEY y quitá QDRANT_PATH."
    )


if __name__ == "__main__":
    main()
