"""Extracción LLM: transcripts → principios/situaciones/frases JSON estructurado."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer
from anthropic import Anthropic
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from coach_brain.settings import settings

console = Console()

CHUNK_CHARS = 5_000  # ~1500 tokens; chunk chico = menos salida = sin OOM de Ollama
CHUNK_OVERLAP_CHARS = 400

# Cuántos chunks se piden a la vez. Queda en 1 porque se midió y NO sirve, pero el
# código queda: en otra placa sí sirve, y la medición está acá para no repetirla.
#
# Medido el 18/11 sobre LOS MISMOS 4 chunks de un PDF, en una RTX 5070 Ti Laptop
# de 12 GB. 2 slots: 82,6 s. 1 slot: 82,8 s. Ganancia 1,00x. En tok/s de salida,
# 21,0 contra 18,6-19,2, que está dentro de la variación entre corridas.
#
# La teoría decía que tenía que servir: decodificar lee los 9 GB de pesos enteros
# por token, y hacerlo para N secuencias a la vez los lee UNA sola vez. No sirvió
# porque esta placa no está limitada por el ancho de banda: a 672 GB/s teóricos,
# un modelo de 9 GB debería dar ~74 tok/s y da 19, el 26 %. Lo que la frena es que
# los clocks no suben (ver abajo), y batchear no arregla un clock bajo.
#
# Si alguna vez se vuelve a probar con más de 1, hay dos condiciones:
#  - OLLAMA_NUM_PARALLEL >= este número en el SERVER, o Ollama encola igual.
#  - Que entre en la VRAM. `ollama ps` tiene que decir `100% GPU`; cualquier
#    `N%/M% CPU/GPU` manda capas a la CPU y queda MUCHO más lento que en serie.
#    Con KV fp16 no entra: 3 slots a num_ctx 4096 dieron `11%/89%` y 2 slots a
#    4096 dieron `6%/94%`. Entra cuantizando el KV, con dos variables del server
#    (OLLAMA_FLASH_ATTENTION=1 y OLLAMA_KV_CACHE_TYPE=q8_0): así 2 slots a 4096
#    dan `100% GPU`. Pero con num_ctx 4096 el margen de salida queda corto, así
#    que esa combinación no es gratis; ver el comentario de num_ctx más abajo.
EXTRACT_PARALLEL = int(os.environ.get("COACH_EXTRACT_PARALLEL", "1"))


def _client() -> Anthropic:
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY no configurada. Edita .env y poné tu key."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def _load_prompt() -> str:
    p = settings.prompts_path / "extractor.md"
    return p.read_text(encoding="utf-8")


def _chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Chunking simple por chars con overlap. Suficiente para extracción."""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    # El `while` nunca sale por su condicion: el ciclo siempre corta con el
    # `break` de abajo, porque llega ahi en cuanto `end` toca el final del
    # texto. De ahi el `no branch`: la salida de arriba no es alcanzable.
    while start < len(text):  # pragma: no branch
        end = min(start + size, len(text))
        # Tratá de cortar en punto/salto de línea cercano
        if end < len(text):
            for sep in ["\n\n", ". ", "\n", " "]:
                idx = text.rfind(sep, start + size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end])
        start = end - overlap
        if start <= 0 or end >= len(text):
            break
    return chunks


def _repair_json(s: str) -> str:
    """Repara JSON parcial/truncado: saca comas colgantes y cierra
    strings/brackets/braces que quedaron abiertos (caso típico: la respuesta
    del modelo se cortó por max_tokens a mitad de un objeto)."""
    # Recorrer ignorando el contenido de strings para saber qué quedó abierto.
    stack: list[str] = []
    in_str = False
    escaped = False
    for ch in s:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        s += '"'
    # Sacar coma colgante antes de cerrar (",  }" / ", ]") que el truncado deja.
    s = re.sub(r",\s*$", "", s.rstrip())
    # Cerrar lo que quedó abierto, en orden inverso.
    for opener in reversed(stack):
        s += "}" if opener == "{" else "]"
    return s


def _extract_json(s: str) -> dict | None:
    """Intenta parsear JSON robustamente (con markdown fences, basura, truncado)."""
    # Buscar primer { hasta último } balanceado-ish
    s = s.strip()
    # Quitar fences ```json ... ```
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Fallback: buscar bloque {...} (greedy) y, si no parsea, repararlo.
    m = re.search(r"\{[\s\S]*\}", s)
    candidate = m.group(0) if m else (s[s.find("{"):] if "{" in s else None)
    if not candidate:
        return None
    # Quitar comas colgantes ", }" / ", ]" en cualquier parte.
    fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
    for attempt in (candidate, fixed, _repair_json(fixed)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


_SYSTEM = "Sos un extractor de conocimiento. Devolvés solo JSON válido, nada más."

# Schema laxo: garantiza las 3 claves como arrays de objetos (Ollama lo fuerza →
# el output SIEMPRE parsea). El contenido de cada item lo guía el prompt.
_OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "principios": {"type": "array", "items": {"type": "object"}},
        "situaciones": {"type": "array", "items": {"type": "object"}},
        "frases": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["principios", "situaciones", "frases"],
}


def _extract_anthropic(client: Anthropic, prompt: str) -> str:
    resp = client.messages.create(
        model=settings.model_extractor,
        max_tokens=8192,
        temperature=0.1,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


def _ollama_alive() -> bool:
    import httpx

    try:
        httpx.get(f"{settings.ollama_host}/api/tags", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


_LOCK_OLLAMA = threading.Lock()


def _ensure_ollama(wait: int = 60) -> bool:
    """Si el server de Ollama se cayó (típico tras un OOM), lo relanza y espera.

    El lock existe porque ahora varios chunks corren a la vez: si el server se cae,
    los N hilos entran acá juntos y sin el lock cada uno lanzaría su propio
    `ollama serve`. El primero que entra lo levanta; los demás encuentran el server
    vivo en el chequeo de arriba y salen sin hacer nada.
    """
    with _LOCK_OLLAMA:
        return _ensure_ollama_locked(wait)


def _ensure_ollama_locked(wait: int = 60) -> bool:
    if _ollama_alive():
        return True
    exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    try:
        creationflags = 0x00000008 if os.name == "nt" else 0  # DETACHED_PROCESS
        subprocess.Popen([str(exe), "serve"], creationflags=creationflags,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001
        console.print(f"[yellow]No pude relanzar Ollama: {e}[/yellow]")
    for _ in range(max(1, wait // 2)):
        time.sleep(2)
        if _ollama_alive():
            console.print("[green]Ollama recuperado.[/green]")
            return True
    return False


def _extract_ollama(prompt: str) -> str:
    """Extracción local con Ollama + structured output (format=schema).

    El parámetro format obliga al modelo a devolver JSON válido (schema), así que
    el parseo nunca falla por formato. Reintenta ante 500/conexión y SI el server
    se cayó (OOM) lo relanza solo -> el job se auto-recupera y no se frena.
    """
    import httpx

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": _OLLAMA_SCHEMA,
        # num_ctx 6144 con CHUNK_CHARS 5000, que es con lo que se extrajeron los
        # 8548 archivos que ya están hechos. El margen sobra a propósito.
        #
        # Medido el 18/11 sobre chunks reales de un PDF: el prompt pesa 2107-2155
        # tokens y la salida llegó a 892, o sea ~3050 en el peor caso visto. Se
        # probaron 4096 y 3072 para ahorrar KV cache, y 3072 se descartó porque un
        # chunk tocó 2999 de 3072. Pasado el borde Ollama NO falla: desplaza el
        # contexto y descarta la parte vieja del prompt, así que el chunk se
        # extrae a medias y nada lo avisa (el mismo defecto silencioso de los 32
        # PDFs que extrajeron cero). Bajarlo solo servía para meter 2 slots en
        # paralelo en la VRAM, y el paralelo midió 1,00x, así que no hay nada que
        # comprar con ese riesgo.
        #
        # Si se sube CHUNK_CHARS o se agranda prompts/extractor.md, hay que volver
        # a medir prompt_eval_count y eval_count de un chunk real antes de tocar
        # esto.
        "options": {"temperature": 0.1, "num_ctx": 6144},
    }
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            r = httpx.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=600)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            _ensure_ollama()  # si murió (OOM), lo levanta; si está vivo, no hace nada
            time.sleep(min(2 ** attempt * 3, 20))
    raise last_err if last_err else RuntimeError("ollama: fallo desconocido")


def extract_chunk(client, prompt_template: str, source_id: str, chunk_idx: int, chunk: str) -> dict:
    """Extrae un chunk con el backend configurado (anthropic | ollama)."""
    prompt = prompt_template.replace("{source_id}", f"{source_id}_c{chunk_idx}").replace("{chunk}", chunk)

    if settings.extractor_backend == "ollama":
        text = _extract_ollama(prompt)
    else:
        text = _extract_anthropic(client, prompt)

    data = _extract_json(text)
    if not data:
        console.print(f"[yellow]No se pudo parsear JSON del chunk {chunk_idx} ({source_id})[/yellow]")
        return {"principios": [], "situaciones": [], "frases": []}
    return data


def process_transcript(client: Anthropic, prompt_template: str, transcript_path: Path) -> dict:
    """Procesa un transcript completo, combinando todos los chunks."""
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    source_id = data["source_id"]
    full_text = data.get("full_text", "")

    if not full_text.strip():
        console.print(f"[yellow]Transcript vacío: {transcript_path.name}[/yellow]")
        return {"source_id": source_id, "principios": [], "situaciones": [], "frases": []}

    chunks = _chunk_text(full_text)

    # Los chunks se piden de a EXTRACT_PARALLEL a la vez, pero se combinan EN ORDEN:
    # `executor.map` devuelve los resultados en el orden de entrada, no en el de
    # terminación, así que el archivo de salida sale idéntico a como salía en serie.
    # Cada chunk es independiente (el prompt lleva su propio source_id_cN), así que
    # el paralelismo no cambia lo que extrae, solo cuándo.
    fallidos: list[int] = []

    def _uno(par: tuple[int, str]) -> dict:
        idx, chunk = par
        try:
            return extract_chunk(client, prompt_template, source_id, idx, chunk)
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Error en chunk {idx} de {source_id}: {e}[/red]")
            fallidos.append(idx)
            return {"principios": [], "situaciones": [], "frases": []}

    all_principios: list[dict] = []
    all_situaciones: list[dict] = []
    all_frases: list[dict] = []

    if EXTRACT_PARALLEL > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=EXTRACT_PARALLEL) as pool:
            resultados = list(pool.map(_uno, enumerate(chunks)))
    else:
        resultados = [_uno(par) for par in enumerate(chunks)]

    for extracted in resultados:
        all_principios.extend(extracted.get("principios", []) or [])
        all_situaciones.extend(extracted.get("situaciones", []) or [])
        all_frases.extend(extracted.get("frases", []) or [])

    # Un chunk que falla después de sus 4 reintentos deja un agujero en el archivo
    # que igual se escribe. Antes eso quedaba en una línea roja perdida entre miles;
    # ahora el total viaja con el resultado y la línea de OK lo dice, para que un
    # archivo incompleto no se confunda con uno completo (es el mismo defecto que
    # tuvieron los 32 PDFs que extrajeron cero sin que nadie se enterara).
    return {
        "source_id": source_id,
        "source_type": data.get("source_type"),
        "title": data.get("title"),
        "principios": all_principios,
        "situaciones": all_situaciones,
        "frases": all_frases,
        "chunks_totales": len(chunks),
        "chunks_fallidos": sorted(fallidos),
    }


def run(force: bool = False) -> None:
    transcripts_dir = settings.transcripts_path
    out_dir = settings.extracted_path
    out_dir.mkdir(parents=True, exist_ok=True)

    transcripts = sorted(transcripts_dir.glob("*.json"))
    if not transcripts:
        console.print(f"[yellow]No hay transcripts en {transcripts_dir}[/yellow]")
        return

    client = None if settings.extractor_backend == "ollama" else _client()
    if settings.extractor_backend == "ollama":
        console.print(f"[cyan]Backend extracción: Ollama ({settings.ollama_model}) — local, gratis[/cyan]")
    prompt_template = _load_prompt()

    console.print(f"[cyan]Extrayendo de {len(transcripts)} transcript(s)...[/cyan]")
    ok = skip = err = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extrayendo", total=len(transcripts))
        for tpath in transcripts:
            out_path = out_dir / f"{tpath.stem}.extracted.json"
            if out_path.exists() and not force:
                skip += 1
                progress.advance(task)
                continue
            try:
                result = process_transcript(client, prompt_template, tpath)
                out_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                ok += 1
                fallidos = result.get("chunks_fallidos") or []
                aviso = (
                    f" [red]INCOMPLETO: {len(fallidos)} de "
                    f"{result.get('chunks_totales')} chunks fallaron {fallidos}[/red]"
                    if fallidos
                    else ""
                )
                console.print(
                    f"[green]OK[/green] {tpath.name} → "
                    f"p={len(result['principios'])} s={len(result['situaciones'])} "
                    f"f={len(result['frases'])}{aviso}"
                )
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Error procesando {tpath.name}: {e}[/red]")
                err += 1
            progress.advance(task)

    console.print(f"[green]Listo[/green] — ok={ok} skip={skip} errores={err}")


if __name__ == "__main__":
    typer.run(run)
