"""Retrieval desde Qdrant: traer principios/situaciones/frases relevantes a una query."""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.console import Console

from coach_brain.kb.embed import embed_one
from coach_brain.kb.index import (
    COL_PRINCIPLES,
    COL_SITUATIONS,
    COL_STYLE,
    get_qdrant,
)

console = Console()


@dataclass
class RetrievalResult:
    principles: list[dict]
    situations: list[dict]
    style: list[dict]

    def format_for_prompt(self) -> dict[str, str]:
        """Convierte resultados en strings listos para inyectar en el system prompt."""
        principles_str = "\n".join(
            f"- {p.get('texto', '')}" for p in self.principles
        ) or "(ninguno recuperado)"

        situations_str = "\n\n".join(
            f"### {s.get('descripcion', '')}\n"
            f"Diagnóstico: {s.get('diagnostico', '')}\n"
            f"Recomendado: {s.get('respuesta_recomendada', '')}\n"
            f"Evitar: {s.get('respuesta_evitar', '')}\n"
            f"Tono: {s.get('tono', '')} | Riesgo: {s.get('riesgo', '')}"
            for s in self.situations
        ) or "(ninguna recuperada)"

        style_str = "\n".join(
            f"- \"{f.get('texto', '')}\" — uso: {f.get('uso', '')} ({f.get('tono', '')})"
            for f in self.style
        ) or "(ninguna recuperada)"

        return {
            "principles_retrieved": principles_str,
            "situations_retrieved": situations_str,
            "style_retrieved": style_str,
        }


# Coseno sobre bge-m3: por debajo de esto el match es ruido y no material del coach.
# Sin umbral, query_points siempre devuelve top-N aunque nada sea relevante, y el
# system prompt le dice al modelo que el material recuperado le gana a su criterio.
MIN_SCORE = 0.45


def retrieve(query: str, top_principles: int = 10, top_situations: int = 6, top_style: int = 8,
             min_score: float = MIN_SCORE) -> RetrievalResult:
    """Busca en las 3 colecciones y devuelve resultados agregados."""
    # Instrumentado: el retrieval tarda minutos en CPU compartida y hay que saber
    # si el costo está en el embedding o en la búsqueda (la solución difiere).
    _t0 = time.perf_counter()
    client = get_qdrant()
    _t1 = time.perf_counter()
    vec = embed_one(query)
    _t2 = time.perf_counter()
    console.print(f"[cyan]retrieve: qdrant_client={_t1-_t0:.2f}s embed={_t2-_t1:.2f}s[/cyan]")

    def _search(collection: str, limit: int) -> list[dict]:
        # qdrant-client >=1.12 reemplazó .search() por .query_points().
        try:
            _s = time.perf_counter()
            resp = client.query_points(
                collection_name=collection, query=vec, limit=limit,
                score_threshold=min_score,
            )
            console.print(f"[cyan]retrieve: search {collection}={time.perf_counter()-_s:.2f}s[/cyan]")
            return [p.payload for p in resp.points]
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]retrieve: fallo en {collection}: {type(e).__name__}: {e}[/yellow]")
            return []

    return RetrievalResult(
        principles=_search(COL_PRINCIPLES, top_principles),
        situations=_search(COL_SITUATIONS, top_situations),
        style=_search(COL_STYLE, top_style),
    )
