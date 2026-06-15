"""MVP Streamlit: subí screenshot o pegá chat → Claude analiza con RAG sobre el coach."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import streamlit as st
from anthropic import Anthropic
from PIL import Image

from coach_brain.kb.memory import (
    create_contact,
    find_by_name,
    get_contact,
    list_contacts,
    update_after_analysis,
)
from coach_brain.kb.retrieve import retrieve
from coach_brain.settings import settings


PRINCIPLES_CORE = """\
- No subas el esfuerzo cuando ella lo baja.
- Match the energy; nunca persigas.
- No te justifiques, no expliques de más.
- El mejor mensaje a veces es no mandar nada.
- Concretá planes; no dejes la pelota en su cancha sin necesidad.
- Si está testeando, no defiendas: redirigí con humor o calma.
- Un solo mensaje por turno. Nunca doble texto sin respuesta.
- Calidad sobre velocidad; tu tiempo vale lo mismo que el de ella.
- Si la conversación muere, soltá limpio; no la mendigues.
- Tu frame primero; lo demás se acomoda alrededor.
"""


# Modos de respuesta: cada uno tuerce el TONO de las 3 líneas. Mantienen el frame
# del coach (nunca necesitado/justificativo), solo cambian el color.
MODES: dict[str, str] = {
    "😎 Normal": "Tono balanceado y calmo. Líneas naturales, sin forzar.",
    "🔥 Picante": "Subí la tensión: provocá, jugá al borde, doble sentido, atrevido pero con clase. Nada vulgar ni necesitado.",
    "❤️ Enamorar": "Buscá conexión real: una pizca de vulnerabilidad calibrada, profundidad, hacé que se sienta vista. Sin cursilería ni sobre-inversión.",
    "💬 Chamuyo": "Labia pura: halago ingenioso, seductor verbal, encanto rioplatense. Que tenga calle, no que parezca guion.",
    "🌫️ Misterio": "Generá intriga: líneas cortas, no reveles todo, dejá que ella quiera más. Menos es más.",
    "😂 Divertido": "Todo con humor: chistes, banter, juego, exageración cómica, teasing. Que se ría sí o sí.",
}
_HUMOR_NUDGE = "En lo posible meté humor/juego en las líneas; al coach le funciona el banter."


def _client() -> Anthropic:
    if not settings.anthropic_api_key:
        st.error("ANTHROPIC_API_KEY no configurada en .env")
        st.stop()
    return Anthropic(api_key=settings.anthropic_api_key)


def _load_system_prompt() -> str:
    p = settings.prompts_path / "system_coach.md"
    return p.read_text(encoding="utf-8")


def _image_to_b64(img: Image.Image) -> tuple[str, str]:
    """Devuelve (media_type, base64_str)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "image/png", base64.standard_b64encode(buf.getvalue()).decode("ascii")


@st.cache_resource
def _get_ocr():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def ocr_chat_from_image(img: Image.Image) -> tuple[str, str]:
    """Lee el chat del screenshot con OCR LOCAL (RapidOCR) — gratis, sin API.

    Devuelve (chat_text con [yo]/[ella], raw_text crudo incluyendo el encabezado).
    El crudo sirve para autodetectar el nombre del contacto (header de WhatsApp).
    """
    import numpy as np

    from coach_brain.ingest.whatsapp_images_ocr import _is_noise

    ocr = _get_ocr()
    rgb = img.convert("RGB")
    width = rgb.size[0]
    res, _ = ocr(np.array(rgb))
    if not res:
        return "", ""
    items = sorted(res, key=lambda r: r[0][0][1])  # arriba -> abajo
    raw_lines = [(t or "").strip() for _b, t, _c in items if (t or "").strip()]
    turns: list[list] = []
    for box, txt, _conf in items:
        txt = (txt or "").strip()
        if not txt or _is_noise(txt):
            continue
        who = "yo" if box[0][0] > width * 0.5 else "ella"
        if turns and turns[-1][0] == who:
            turns[-1][1] += " " + txt
        else:
            turns.append([who, txt])
    chat = "\n".join(f"[{w}] {t}" for w, t in turns)
    return chat, "\n".join(raw_lines)


def guess_contact_name(text: str) -> str:
    """Adivina el nombre de la CHICA (no Ilan) desde el chat/header. Ollama → Haiku fallback."""
    if not text.strip():
        return ""
    prompt = (
        "Abajo hay texto de un chat (puede incluir el encabezado del contacto arriba).\n"
        "¿Cuál es el NOMBRE de la otra persona (la chica), NO de Ilan/yo?\n"
        "Si hay un nombre de contacto en el encabezado, usá ese. Si no se puede saber, respondé 'desconocido'.\n"
        "Devolvé SOLO el nombre (1-2 palabras), sin comillas ni explicación.\n\n"
        f"TEXTO:\n{text[:1500]}"
    )

    def _clean(raw: str) -> str:
        name = raw.strip().strip('"').split("\n")[0].strip()
        return name if name and "desconoc" not in name.lower() and len(name) <= 40 else ""

    # Ollama (local, gratis)
    if settings.ollama_enabled:
        try:
            import httpx
            r = httpx.post(
                f"{settings.ollama_host}/api/chat",
                json={"model": settings.ollama_model,
                      "messages": [{"role": "user", "content": prompt}],
                      "stream": False, "options": {"temperature": 0, "num_ctx": 4096}},
                timeout=30)
            r.raise_for_status()
            name = _clean(r.json()["message"]["content"])
            if name:
                return name
        except Exception:  # noqa: BLE001
            pass

    # Fallback: Anthropic Haiku (cloud)
    if settings.anthropic_api_key:
        try:
            resp = _client().messages.create(
                model=settings.model_fast,
                max_tokens=30,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            name = _clean("".join(b.text for b in resp.content if hasattr(b, "text")))
            if name:
                return name
        except Exception:  # noqa: BLE001
            pass

    return ""


def parse_chat_from_image(client: Anthropic, img: Image.Image) -> dict:
    """Pasa la imagen a Haiku para extraer el chat como JSON estructurado."""
    media_type, b64 = _image_to_b64(img)
    resp = client.messages.create(
        model=settings.model_fast,
        max_tokens=2000,
        temperature=0,
        system=(
            "Sos un extractor de chats de WhatsApp/Telegram a partir de screenshots. "
            "Devolvés SOLO JSON con la forma "
            '{"messages":[{"sender":"ella"|"yo","text":"..."}], "notas":"..."}. '
            "Detectá quién es 'ella' por color/posición de burbuja. "
            "Si ves emojis, audios, stickers, marcalo. Si hay 'visto'/'escribiendo', anotalo en notas."
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": "Extraé el chat como JSON."},
                ],
            }
        ],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    # Limpiar fences
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"messages": [], "notas": text}


def build_chat_text(parsed: dict) -> str:
    msgs = parsed.get("messages", [])
    lines = [f"[{m.get('sender', '?')}] {m.get('text', '')}" for m in msgs]
    notas = parsed.get("notas", "")
    out = "\n".join(lines)
    if notas:
        out += f"\n\n(notas del extractor: {notas})"
    return out


def _coach_ollama(system_prompt: str, user_content: str) -> str:
    """Respuesta local con Ollama (gratis). Texto libre, sin schema."""
    import httpx

    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0.7, "num_ctx": 10240},
    }
    r = httpx.post(f"{settings.ollama_host}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"]


def call_coach(backend: str, model: str, system_prompt: str, chat_text: str, user_q: str,
               retrieved: dict | None = None, mode_directive: str = "") -> str:
    """Genera la respuesta del coach con el backend/modelo elegido en la UI.

    backend: "ollama" (local, gratis) | "anthropic" (API paga, model = Sonnet o Haiku).
    `retrieved` se repite acá (mensaje de usuario) para que el modelo lo use sí o sí.
    """
    coach_block = ""
    if retrieved:
        coach_block = f"""
Material del coach recuperado para ESTE caso (fundamentá tu respuesta acá, no en consejos genéricos):

PRINCIPIOS:
{retrieved.get('principles_retrieved', '')}

SITUACIONES SIMILARES (con diagnóstico/recomendación del coach):
{retrieved.get('situations_retrieved', '')}

FRASES DE ESTILO (tono):
{retrieved.get('style_retrieved', '')}
"""

    mode_block = ""
    if mode_directive:
        mode_block = f"\nMODO DE RESPUESTA: {mode_directive}\nLas 3 líneas tienen que respetar ESTE modo (manteniendo el frame del coach).\n"

    user_content = f"""\
Conversación actual:
```
{chat_text}
```
{coach_block}{mode_block}
Pregunta de Ilan: {user_q or "¿qué le respondo?"}

Respondé con el formato del coach. El Diagnóstico y las 3 líneas deben apoyarse en el material de arriba.
"""

    if backend == "ollama":
        return _coach_ollama(system_prompt, user_content)

    client = _client()  # backend pago
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0.7,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))


# ============ UI ============

st.set_page_config(page_title="Coach Asistente", layout="wide")
st.title("🎯 Coach Asistente")
st.caption("Subí screenshot o pegá texto del chat. Te devuelvo diagnóstico + opciones de respuesta.")

with st.sidebar:
    st.subheader("Configuración")
    st.text_input("Tu nombre", value="Ilan", key="user_name")
    use_rag = st.checkbox("Usar RAG (recuperar del coach)", value=True)
    st.divider()
    st.subheader("Modelo de respuesta")
    _resp_opts: dict = {}
    if settings.ollama_enabled:
        _resp_opts[f"🟢 Ollama (gratis) — {settings.ollama_model}"] = ("ollama", settings.ollama_model)
    _resp_opts[f"💳 Sonnet — {settings.model_main}"] = ("anthropic", settings.model_main)
    _resp_opts[f"💳 Haiku — {settings.model_fast}"] = ("anthropic", settings.model_fast)
    _resp_keys = list(_resp_opts)
    _use_ollama = settings.ollama_enabled and settings.response_backend == "ollama"
    _default_idx = 0 if _use_ollama else (0 if not settings.ollama_enabled else 1)
    resp_choice = st.radio("¿Con qué genero la respuesta?", _resp_keys, index=_default_idx)
    resp_backend, resp_model = _resp_opts[resp_choice]

    st.divider()
    st.subheader("Leer screenshot")
    vision_choice = st.radio(
        "¿Cómo leo las imágenes?",
        ["🟢 OCR local (gratis)", "💳 Haiku vision (pago)"],
        index=0,
    )
    vision_local = vision_choice.startswith("🟢")

col_in, col_out = st.columns([1, 1.3])

with col_in:
    st.subheader("1. Input")
    uploaded = st.file_uploader("Screenshot del chat", type=["png", "jpg", "jpeg", "webp"])
    chat_text_manual = st.text_area(
        "O pegá el texto del chat",
        height=180,
        placeholder="[ella] hola\n[yo] hey\n[ella] qué hacés?",
    )
    # Memoria por contacto (id único: puede haber varias "Mica")
    _saved = list_contacts()
    _sel = st.selectbox("Chica (memoria)", ["➕ Nueva"] + [c["label"] for c in _saved])
    contact_id = None
    new_contact = None
    auto_detect = False
    if _sel == "➕ Nueva":
        auto_detect = st.checkbox("🔎 Detectar la chica sola (del chat/foto)", value=True)
        _nn = st.text_input("Nombre de la chica (vacío = autodetectar)", placeholder="Mica")
        _ca, _cb = st.columns(2)
        _ape = _ca.text_input("Apellido (opc)")
        _tag = _cb.text_input("Tag (opc)", placeholder="gimnasio / IG / Tinder")
        if _nn.strip():
            _dups = find_by_name(_nn)
            if _dups and not (_ape.strip() or _tag.strip()):
                st.warning(
                    f"Ya hay {len(_dups)} '{_nn.strip()}': "
                    f"{', '.join(d['label'] for d in _dups)}. "
                    "Poné apellido o tag para distinguir, o elegila arriba.")
        new_contact = (_nn.strip(), _ape.strip(), _tag.strip()) if _nn.strip() else None
    else:
        contact_id = next((c["id"] for c in _saved if c["label"] == _sel), None)
        _m = get_contact(contact_id) if contact_id else None
        if _m and _m.get("profile"):
            with st.expander(f"🧠 Memoria de {_m['label']} ({len(_m.get('history', []))} chats)"):
                st.markdown(_m["profile"])

    user_q = st.text_input("Pregunta opcional", placeholder="¿qué le respondo?")
    st.write("**Modo de respuesta**")
    mode_choice = st.radio("Modo", list(MODES), horizontal=True, label_visibility="collapsed")
    st.caption(MODES[mode_choice])
    go = st.button("🚀 Analizar", type="primary", use_container_width=True)

with col_out:
    st.subheader("2. Análisis")
    if go:
        client = None  # se crea solo si hace falta (vision de imagen o backend pago)
        chat_text = ""
        detect_source = ""

        if uploaded is not None:
            img = Image.open(uploaded)
            st.image(img, caption="Chat detectado", width=350)
            if vision_local:
                with st.spinner("Leyendo screenshot (OCR local, gratis)..."):
                    chat_text, detect_source = ocr_chat_from_image(img)
            else:
                client = _client()  # vision con Haiku (API)
                with st.spinner("Leyendo screenshot (Haiku)..."):
                    parsed = parse_chat_from_image(client, img)
                    chat_text = build_chat_text(parsed)
                    detect_source = chat_text + " " + str(parsed.get("notas", ""))
            with st.expander("Chat extraído"):
                st.code(chat_text or "(no se detectó texto)")
        elif chat_text_manual.strip():
            chat_text = chat_text_manual.strip()
            detect_source = chat_text
        else:
            st.warning("Subí una imagen o pegá texto.")
            st.stop()

        # Auto-mapear contacto desde el contenido (si no se eligió/escribió uno)
        if contact_id is None and new_contact is None and auto_detect and detect_source.strip():
            with st.spinner("Detectando quién es..."):
                _cand = guess_contact_name(detect_source)
            if _cand:
                _matches = find_by_name(_cand)
                if len(_matches) == 1:
                    contact_id = _matches[0]["id"]
                    st.info(f"Detecté **{_cand}** → uso su memoria ({_matches[0]['label']}).")
                elif len(_matches) > 1:
                    st.warning(
                        f"Detecté **{_cand}** pero hay {len(_matches)} guardadas: "
                        f"{', '.join(m['label'] for m in _matches)}. Elegila arriba para no mezclar.")
                else:
                    new_contact = (_cand, "", "")
                    st.info(f"Detecté **{_cand}** (nueva) → la creo y arranco su memoria.")
            else:
                st.caption("No detecté el nombre. Poné uno arriba si querés memoria.")

        # Retrieval
        retrieved_str = {
            "principles_retrieved": "(RAG desactivado)",
            "situations_retrieved": "(RAG desactivado)",
            "style_retrieved": "(RAG desactivado)",
        }
        if use_rag:
            with st.spinner("Recuperando del coach..."):
                try:
                    result = retrieve(chat_text + " " + user_q)
                    retrieved_str = result.format_for_prompt()
                    with st.expander(f"Recuperado: {len(result.principles)} ppios | {len(result.situations)} sits | {len(result.style)} frases"):
                        st.json({
                            "principles": result.principles,
                            "situations": result.situations,
                            "style": result.style,
                        })
                except Exception as e:
                    st.warning(f"RAG no disponible aún: {e}")

        # Memoria del contacto (por id)
        _cmem = "(sin historial de esta persona)"
        if contact_id:
            _m = get_contact(contact_id)
            if _m and _m.get("profile"):
                _cmem = f"Perfil de {_m['label']}:\n{_m['profile']}"

        # System prompt
        system_prompt = _load_system_prompt() \
            .replace("{user_name}", st.session_state.get("user_name", "Ilan")) \
            .replace("{principles_core}", PRINCIPLES_CORE) \
            .replace("{principles_retrieved}", retrieved_str["principles_retrieved"]) \
            .replace("{situations_retrieved}", retrieved_str["situations_retrieved"]) \
            .replace("{style_retrieved}", retrieved_str["style_retrieved"]) \
            .replace("{contact_memory}", _cmem)

        _spin = "Pensando (Ollama local)..." if resp_backend == "ollama" else f"Pensando ({resp_model})..."
        with st.spinner(_spin):
            try:
                answer = call_coach(resp_backend, resp_model, system_prompt, chat_text, user_q,
                                    retrieved=retrieved_str if use_rag else None,
                                    mode_directive=f"{MODES[mode_choice]} {_HUMOR_NUDGE}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Error generando la respuesta ({resp_backend}): {e}")
                st.stop()

        st.markdown(answer)

        # Actualizar memoria del contacto (agente local, gratis)
        cid = contact_id
        if cid is None and new_contact and new_contact[0]:
            cid = create_contact(*new_contact)  # crear recién al analizar
        if cid:
            _lbl = (get_contact(cid) or {}).get("label", "contacto")
            with st.spinner(f"Guardando contexto de {_lbl}..."):
                try:
                    prof = update_after_analysis(cid, chat_text, answer,
                                                 user=st.session_state.get("user_name", "Ilan"))
                    with st.expander(f"🧠 Memoria de {_lbl} actualizada"):
                        st.markdown(prof)
                except Exception as e:  # noqa: BLE001
                    st.caption(f"No pude actualizar memoria: {e}")
        else:
            st.caption("Tip: elegí o creá la chica arriba para que recuerde el contexto entre análisis.")
