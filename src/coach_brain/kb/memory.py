"""Memoria por contacto (chica): perfil + historial en SQLite local.

$0, sin nube. Clave = id único (no el nombre) → puede haber varias "Mica".
Cada contacto = nombre + apellido(opc) + tag distintivo(opc, ej "gimnasio").
Cada análisis actualiza un perfil corrido con Ollama. Interfaz pensada para
swappear a Supabase después.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from coach_brain.settings import settings

DB_PATH: Path = settings.processed_path / "contacts.db"

_PROFILE_PROMPT = """\
Sos el memorista del coach. Mantenés un PERFIL corrido de una chica con la que
{user} está hablando. Integrá el nuevo intercambio al perfil previo. Sé conciso,
factual, sin relleno. Markdown con estas secciones (omití la que no tenga datos):

**Quién es**: nombre, qué hace, datos duros (trabajo, estudio, ciudad, edad si aparece).
**Intereses / ganchos**: temas que le interesan, cosas que mencionó.
**Dinámica**: cómo viene el vínculo (interés, distancia, tests, ritmo de respuesta).
**Qué funcionó**: movidas/líneas que dieron resultado.
**Qué evitar**: lo que la enfrió o no pegó.
**Estado**: una línea con dónde está la cosa hoy.

PERFIL PREVIO:
{prev}

NUEVO INTERCAMBIO (chat + veredicto del coach):
{new}

Devolvé SOLO el perfil actualizado en markdown, nada más."""


def _create(c: sqlite3.Connection) -> None:
    c.execute(
        "CREATE TABLE IF NOT EXISTS contacts ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "name TEXT NOT NULL, apellido TEXT DEFAULT '', tag TEXT DEFAULT '', "
        "profile TEXT DEFAULT '', history TEXT DEFAULT '[]', updated_at TEXT)"
    )


def _conn() -> sqlite3.Connection:
    settings.processed_path.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    cols = [r[1] for r in c.execute("PRAGMA table_info(contacts)")]
    if cols and "id" not in cols:
        # Migrar esquema viejo (name como PK) -> nuevo (id).
        c.execute("ALTER TABLE contacts RENAME TO contacts_old")
        _create(c)
        for name, profile, history, updated in c.execute(
            "SELECT name, profile, history, updated_at FROM contacts_old"
        ):
            c.execute(
                "INSERT INTO contacts(name, apellido, tag, profile, history, updated_at) "
                "VALUES(?,?,?,?,?,?)", (name, "", "", profile, history, updated))
        c.execute("DROP TABLE contacts_old")
        c.commit()
    else:
        _create(c)
    return c


def label_of(row: dict) -> str:
    """Etiqueta legible: 'Mica Wurch' / 'Mica (gimnasio)' / 'Mica'."""
    parts = [row["name"]]
    if row.get("apellido"):
        parts.append(row["apellido"])
    base = " ".join(parts)
    if row.get("tag"):
        base += f" ({row['tag']})"
    return base


def list_contacts() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, name, apellido, tag FROM contacts ORDER BY updated_at DESC"
        ).fetchall()
    return [{"id": r[0], "name": r[1], "apellido": r[2], "tag": r[3],
             "label": label_of({"name": r[1], "apellido": r[2], "tag": r[3]})} for r in rows]


def find_by_name(name: str) -> list[dict]:
    """Contactos con el mismo nombre (para avisar colisión de 'Mica')."""
    n = name.strip().lower()
    return [c for c in list_contacts() if c["name"].strip().lower() == n]


def get_contact(cid: int) -> dict | None:
    with _conn() as c:
        r = c.execute(
            "SELECT id, name, apellido, tag, profile, history, updated_at "
            "FROM contacts WHERE id=?", (cid,)).fetchone()
    if not r:
        return None
    d = {"id": r[0], "name": r[1], "apellido": r[2], "tag": r[3],
         "profile": r[4], "history": json.loads(r[5] or "[]"), "updated_at": r[6]}
    d["label"] = label_of(d)
    return d


def create_contact(name: str, apellido: str = "", tag: str = "") -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO contacts(name, apellido, tag, updated_at) VALUES(?,?,?,?)",
            (name.strip(), apellido.strip(), tag.strip(), datetime.now(timezone.utc).isoformat()))
        c.commit()
        return cur.lastrowid


def _save(cid: int, profile: str, history: list) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE contacts SET profile=?, history=?, updated_at=? WHERE id=?",
            (profile, json.dumps(history, ensure_ascii=False),
             datetime.now(timezone.utc).isoformat(), cid))
        c.commit()


def _summarize_profile(user: str, prev: str, chat_text: str, verdict: str) -> str:
    prompt = (_PROFILE_PROMPT
              .replace("{user}", user)
              .replace("{prev}", prev or "(no hay perfil previo — es nueva)")
              .replace("{new}", f"CHAT:\n{chat_text[:1800]}\n\nVEREDICTO:\n{verdict[:1000]}"))

    # Ollama (local, gratis)
    if settings.ollama_enabled:
        try:
            import httpx
            r = httpx.post(
                f"{settings.ollama_host}/api/chat",
                json={"model": settings.ollama_model,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False, "options": {"temperature": 0.2, "num_ctx": 8192}},
                timeout=300)
            r.raise_for_status()
            out = r.json()["message"]["content"].strip()
            if out.startswith("```"):
                out = out.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return out
        except Exception:  # noqa: BLE001
            pass

    # Fallback: Anthropic Haiku (cloud)
    if settings.anthropic_api_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=settings.anthropic_api_key)
            resp = client.messages.create(
                model=settings.model_fast,
                max_tokens=800,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        except Exception:  # noqa: BLE001
            pass

    return prev


def update_after_analysis(cid: int, chat_text: str, verdict: str, user: str = "Ilan") -> str:
    """Integra el intercambio al perfil + historial del contacto (por id)."""
    cur = get_contact(cid)
    if not cur:
        return ""
    history = list(cur.get("history", []))
    history.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "chat": chat_text[:1500], "verdict": verdict[:800],
    })
    history = history[-20:]
    new_profile = _summarize_profile(user, cur.get("profile", ""), chat_text, verdict)
    _save(cid, new_profile, history)
    return new_profile
