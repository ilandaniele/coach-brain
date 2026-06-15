"""Indexador en Qdrant: carga principios/situaciones/frases desde extracted/."""

from __future__ import annotations

# IMPORTANTE (no reordenar): en Windows con torch cu130, importar qdrant_client
# ANTES que sentence_transformers/torch produce un access violation (0xC0000005)
# por conflicto de libs nativas. Forzamos la carga de torch primero.
import sentence_transformers  # noqa: F401,E402  (orden de carga — ver comentario)

import json
import uuid
from pathlib import Path

import typer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rich.console import Console
from rich.progress import track

from coach_brain.kb.embed import embed, embed_dim
from coach_brain.settings import settings

console = Console()

COL_PRINCIPLES = "principles"
COL_SITUATIONS = "situations"
COL_STYLE = "style_phrases"


def get_qdrant() -> QdrantClient:
    """Cliente Qdrant: usa URL si está configurada, si no modo local embebido."""
    if settings.qdrant_url:
        console.print(f"[cyan]Qdrant remoto: {settings.qdrant_url}[/cyan]")
        kwargs: dict = {"url": settings.qdrant_url}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
        return QdrantClient(**kwargs)
    storage = settings.qdrant_storage
    storage.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Qdrant local embebido: {storage}[/cyan]")
    return QdrantClient(path=str(storage))


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Crea la colección si no existe."""
    try:
        client.get_collection(name)
    except Exception:  # noqa: BLE001
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        console.print(f"[green]Creada colección[/green] {name} (dim={dim})")


def _text_for_principle(p: dict) -> str:
    return p.get("texto", "")


def _text_for_situation(s: dict) -> str:
    parts = [s.get("descripcion", ""), s.get("diagnostico", "")]
    return " — ".join(x for x in parts if x)


def _text_for_phrase(f: dict) -> str:
    return f.get("texto", "")


def _point_id() -> str:
    return str(uuid.uuid4())


def index_extracted(client: QdrantClient, extracted_path: Path, source_meta: dict) -> dict:
    """Indexa un archivo .extracted.json en las 3 colecciones."""
    data = json.loads(extracted_path.read_text(encoding="utf-8"))

    counts = {"principles": 0, "situations": 0, "style_phrases": 0}

    # Principios
    principios = data.get("principios") or []
    if principios:
        texts = [_text_for_principle(p) for p in principios]
        vecs = embed(texts)
        points = [
            PointStruct(
                id=_point_id(),
                vector=v,
                payload={
                    **source_meta,
                    "kind": "principle",
                    "principle_id": p.get("id"),
                    "texto": p.get("texto"),
                    "tipo": p.get("tipo"),
                    "absoluto": p.get("absoluto"),
                    "tags": p.get("tags") or [],
                },
            )
            for p, v in zip(principios, vecs, strict=False)
        ]
        client.upsert(collection_name=COL_PRINCIPLES, points=points)
        counts["principles"] = len(points)

    # Situaciones
    situaciones = data.get("situaciones") or []
    if situaciones:
        texts = [_text_for_situation(s) for s in situaciones]
        vecs = embed(texts)
        points = [
            PointStruct(
                id=_point_id(),
                vector=v,
                payload={
                    **source_meta,
                    "kind": "situation",
                    "situation_id": s.get("id"),
                    "descripcion": s.get("descripcion"),
                    "diagnostico": s.get("diagnostico"),
                    "respuesta_recomendada": s.get("respuesta_recomendada"),
                    "respuesta_evitar": s.get("respuesta_evitar"),
                    "tono": s.get("tono"),
                    "riesgo": s.get("riesgo"),
                    "principios_aplicados": s.get("principios_aplicados") or [],
                    "tags": s.get("tags") or [],
                },
            )
            for s, v in zip(situaciones, vecs, strict=False)
        ]
        client.upsert(collection_name=COL_SITUATIONS, points=points)
        counts["situations"] = len(points)

    # Frases
    frases = data.get("frases") or []
    if frases:
        texts = [_text_for_phrase(f) for f in frases]
        vecs = embed(texts)
        points = [
            PointStruct(
                id=_point_id(),
                vector=v,
                payload={
                    **source_meta,
                    "kind": "phrase",
                    "phrase_id": f.get("id"),
                    "texto": f.get("texto"),
                    "uso": f.get("uso"),
                    "tono": f.get("tono"),
                },
            )
            for f, v in zip(frases, vecs, strict=False)
        ]
        client.upsert(collection_name=COL_STYLE, points=points)
        counts["style_phrases"] = len(points)

    return counts


def run(reset: bool = False) -> None:
    extracted_dir = settings.extracted_path
    extracts = sorted(extracted_dir.glob("*.extracted.json"))
    if not extracts:
        console.print(f"[yellow]No hay extractos en {extracted_dir}. Corré `coach extract` primero.[/yellow]")
        return

    client = get_qdrant()
    dim = embed_dim()

    if reset:
        for name in [COL_PRINCIPLES, COL_SITUATIONS, COL_STYLE]:
            try:
                client.delete_collection(name)
                console.print(f"[yellow]Borrada colección {name}[/yellow]")
            except Exception:  # noqa: BLE001
                pass

    for name in [COL_PRINCIPLES, COL_SITUATIONS, COL_STYLE]:
        ensure_collection(client, name, dim)

    totals = {"principles": 0, "situations": 0, "style_phrases": 0}

    for epath in track(extracts, description="Indexando"):
        try:
            source_meta = {
                "source_id": epath.stem.replace(".extracted", ""),
            }
            counts = index_extracted(client, epath, source_meta)
            for k, v in counts.items():
                totals[k] += v
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error indexando {epath.name}: {e}[/red]")

    console.print(
        f"[green]Indexado completo[/green] — "
        f"principles={totals['principles']} "
        f"situations={totals['situations']} "
        f"style={totals['style_phrases']}"
    )


if __name__ == "__main__":
    typer.run(run)
