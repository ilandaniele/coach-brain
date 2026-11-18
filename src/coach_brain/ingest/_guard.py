"""Guarda compartida para los transcripts que se escriben desde dos lados.

`whatsapp_images.py` (vision con Haiku) y `whatsapp_images_ocr.py` (OCR local)
escriben **el mismo archivo** —`transcripts/whatsapp_images_chats.json`— con el
mismo `source_id`. Es a propósito: así el resto del pipeline no tiene que saber
con qué motor se sacó el texto.

El problema es que la corrida de vision quedó a medias (4074 imágenes de 8978,
2873 chats) porque fallaba y encima cuesta plata, y la de OCR terminó entera
(8953 imágenes, 6336 chats). Un `python -m coach_brain.ingest.whatsapp_images
aggregate` hoy sobreescribiría el transcript bueno con el parcial: 6336
segmentos y 2,6 MB de texto pasarían a 2873, sin un error y sin backup. Después
`kb.extract` lo re-extrae y `kb.index --reset` lo publica, así que la pérdida
llega hasta el índice que sirve las respuestas.

Esta función corta eso: si el archivo que está en disco tiene más segmentos que
el que se va a escribir, no lo pisa y dice por qué. `force=True` lo pisa igual,
para cuando la decisión es a propósito.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

console = Console()


def write_transcript(out_path: Path, data: dict, *, force: bool = False) -> bool:
    """Escribe `data` en `out_path` salvo que ahí ya haya un transcript más grande.

    Devuelve True si escribió. Un archivo ilegible o sin segmentos no bloquea
    nada: lo que protege es un transcript existente que de verdad tiene más
    contenido que el nuevo.
    """
    nuevos = len(data.get("segments") or [])
    if not force and out_path.exists():
        try:
            viejo = json.loads(out_path.read_text(encoding="utf-8"))
            previos = len(viejo.get("segments") or [])
        except Exception:  # noqa: BLE001
            previos = 0
        if previos > nuevos:
            console.print(
                f"[red]No escribo {out_path.name}[/red]: el que está en disco tiene "
                f"{previos} segmentos y este tiene {nuevos}. "
                f"Sería perder {previos - nuevos}. Usá force=True si es a propósito."
            )
            return False

    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return True
