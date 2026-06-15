"""Ingestor de export de WhatsApp (formato iOS, español).

La conversación es coaching 1-a-1: Ema Pereyra (coach) <-> Ilan (alumno).
Parsea el `_chat.txt`, separa mensajes, detecta adjuntos ("imagen omitida",
"audio omitido", etc.) y arma un transcript unificado con roles etiquetados
para que el extractor sepa de quién es cada consejo.

Mapeo de los archivos de media reales (8k imágenes / 8k audios) a su posición
en el chat = v2 (se hace por timestamp embebido en el nombre del archivo).
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from coach_brain.settings import REPO_ROOT, settings

console = Console()

# Quién es quién (ajustable). Cualquier remitente no listado -> "otro".
ROLE_BY_SENDER = {
    "Ema Pereyra": "coach",
    "Ilan Daniele": "alumno",
}

# Separadores invisibles que mete iOS: U+200E (LRM), U+202F (narrow NBSP),
# U+00A0 (NBSP), U+200B (zero width). Los normalizamos a espacio normal.
_INVISIBLE = dict.fromkeys(map(ord, "‎‏  ​﻿"), " ")

# [6/5/23, 7:27:32 p. m.] Nombre Apellido: mensaje
_LINE_RE = re.compile(
    r"^\[(\d{1,2})/(\d{1,2})/(\d{2,4}),\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})\s*"
    r"(?:([ap])\.\s*m\.)?\]\s*"
    r"([^:]+?):\s?(.*)$",
    re.DOTALL,
)

# Marcadores de adjunto: "imagen omitida", "audio omitido", "video omitido",
# "documento omitido", "sticker omitido", "GIF omitido". Puede venir precedido
# del nombre de archivo (ej. "10 de julio.mp4 documento omitido").
_ATTACH_RE = re.compile(
    r"^(?P<fname>.*?)\s*(?P<kind>imagen|audio|video|documento|sticker|GIF|Contacto)\s+omitid[oa]\s*$",
    re.IGNORECASE,
)

_ATTACH_LABEL = {
    "imagen": "[imagen]",
    "audio": "[nota de voz]",
    "video": "[video]",
    "documento": "[documento]",
    "sticker": "[sticker]",
    "gif": "[gif]",
    "contacto": "[contacto]",
}

# Frases de sistema que no son conversación.
_SYSTEM_SUBSTRINGS = (
    "Los mensajes y las llamadas están cifrados",
    "Las llamadas y los mensajes están cifrados",
    "se eliminó este mensaje",
    "Eliminaste este mensaje",
    "Se eliminó este mensaje",
    "cambió su número de teléfono",
    "cambió al grupo",
    "<Multimedia omitido>",
)


# Marcadores inline de sistema (mensaje editado/eliminado) -> se quitan del texto.
_INLINE_NOISE_RE = re.compile(
    r"\s*<\s*(?:se (?:editó|eliminó) este mensaje|this message was edited|"
    r"you deleted this message|multimedia omitido)\s*\.?\s*>\s*",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Saca caracteres invisibles de iOS y normaliza."""
    text = text.translate(_INVISIBLE)
    text = unicodedata.normalize("NFC", text)
    text = _INLINE_NOISE_RE.sub(" ", text)
    return text.strip()


def _parse_dt(d: str, mo: str, y: str, hh: str, mm: str, ss: str, ampm: str | None) -> str | None:
    try:
        year = int(y)
        if year < 100:
            year += 2000
        hour = int(hh)
        if ampm:
            ap = ampm.lower()
            if ap == "p" and hour != 12:
                hour += 12
            elif ap == "a" and hour == 12:
                hour = 0
        return datetime(year, int(mo), int(d), hour, int(mm), int(ss)).isoformat()
    except (ValueError, TypeError):
        return None


def _is_system(text: str) -> bool:
    return any(s.lower() in text.lower() for s in _SYSTEM_SUBSTRINGS)


def parse_chat(chat_path: Path) -> list[dict]:
    """Devuelve lista de mensajes {ts, sender, role, text, attachment}."""
    raw = chat_path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    messages: list[dict] = []
    for line in raw.split("\n"):
        clean_line = _clean(line)
        if not clean_line:
            continue

        m = _LINE_RE.match(clean_line)
        if not m:
            # Continuación del mensaje anterior (multi-línea).
            if messages:
                messages[-1]["text"] = (messages[-1]["text"] + "\n" + clean_line).strip()
            continue

        d, mo, y, hh, mm, ss, ampm, sender, body = m.groups()
        sender = sender.strip()
        body = body.strip()

        if _is_system(body):
            continue

        attachment = None
        am = _ATTACH_RE.match(body)
        if am:
            kind = am.group("kind").lower()
            fname = am.group("fname").strip()
            attachment = kind
            label = _ATTACH_LABEL.get(kind, f"[{kind}]")
            body = f"{fname} {label}".strip() if fname else label

        if not body and not attachment:
            continue

        messages.append({
            "ts": _parse_dt(d, mo, y, hh, mm, ss, ampm),
            "sender": sender,
            "role": ROLE_BY_SENDER.get(sender, "otro"),
            "text": body,
            "attachment": attachment,
        })

    return messages


def _render_full_text(messages: list[dict]) -> str:
    """Diálogo legible con roles, para que el extractor entienda el coaching."""
    label = {"coach": "Coach (Ema)", "alumno": "Alumno (Ilan)"}
    parts = []
    for msg in messages:
        who = label.get(msg["role"], msg["sender"])
        parts.append(f"{who}: {msg['text']}")
    return "\n".join(parts)


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def run(force: bool = False) -> None:
    wa_dir = settings.raw_path / "whatsapp"
    chat_path = wa_dir / "_chat.txt"
    if not chat_path.exists():
        console.print(f"[yellow]No existe {chat_path}. Dejá el export de WhatsApp ahí.[/yellow]")
        return

    out_dir = settings.transcripts_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "whatsapp_ema_coaching.json"
    if out_path.exists() and not force:
        console.print(f"[yellow]Ya existe {out_path.name} (usá --force para rehacer).[/yellow]")
        return

    console.print(f"[cyan]Parseando {chat_path}...[/cyan]")
    messages = parse_chat(chat_path)

    n_att = {}
    for msg in messages:
        if msg["attachment"]:
            n_att[msg["attachment"]] = n_att.get(msg["attachment"], 0) + 1
    n_coach = sum(1 for m in messages if m["role"] == "coach")
    n_alumno = sum(1 for m in messages if m["role"] == "alumno")

    full_text = _render_full_text(messages)

    data = {
        "source_id": "whatsapp_ema_coaching",
        "source_type": "whatsapp",
        "source_path": _relpath(chat_path),
        "title": "Coaching WhatsApp — Ema Pereyra / Ilan",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": "es",
        "metadata": {
            "messages": len(messages),
            "coach_messages": n_coach,
            "alumno_messages": n_alumno,
            "attachments": n_att,
            "first_ts": next((m["ts"] for m in messages if m["ts"]), None),
            "last_ts": next((m["ts"] for m in reversed(messages) if m["ts"]), None),
        },
        "segments": [
            {"id": i, "text": m["text"], "sender": m["sender"], "role": m["role"],
             "ts": m["ts"], "attachment": m["attachment"], "page": None}
            for i, m in enumerate(messages)
        ],
        "full_text": full_text,
    }

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(
        f"[green]OK[/green] {len(messages)} mensajes "
        f"(coach={n_coach}, alumno={n_alumno}, adjuntos={n_att}) -> {out_path.name}"
    )


if __name__ == "__main__":
    typer.run(run)
