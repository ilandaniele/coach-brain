"""MVP Streamlit: análisis de chats + generador de líneas situacionales + notas personales."""

from __future__ import annotations

import base64
import hmac
import io
import json
import re
import time
from pathlib import Path

import streamlit as st
from anthropic import Anthropic
from PIL import Image

from coach_brain.app.pwa import inject as inject_pwa
from coach_brain.kb.memory import (
    create_contact,
    find_by_name,
    get_contact,
    list_contacts,
    update_after_analysis,
)
from coach_brain.kb.retrieve import retrieve
from coach_brain.llm import available_providers, complete as llm_complete, complete_fast
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

MODES: dict[str, str] = {
    "😎 Normal": "Tono balanceado y calmo. Líneas naturales, sin forzar.",
    "🔥 Picante": "Subí la tensión: provocá, jugá al borde, doble sentido, atrevido pero con clase. Nada vulgar ni necesitado.",
    "❤️ Enamorar": "Buscá conexión real: una pizca de vulnerabilidad calibrada, profundidad, hacé que se sienta vista. Sin cursilería ni sobre-inversión.",
    "💬 Chamuyo": "Labia pura: halago ingenioso, seductor verbal, encanto rioplatense. Que tenga calle, no que parezca guion.",
    "🌫️ Misterio": "Generá intriga: líneas cortas, no reveles todo, dejá que ella quiera más. Menos es más.",
    "😂 Divertido": "Todo con humor: chistes, banter, juego, exageración cómica, teasing. Que se ría sí o sí.",
}
_HUMOR_NUDGE = "En lo posible meté humor/juego en las líneas; al coach le funciona el banter."

NEED_TYPES: dict[str, str] = {
    "💬 Opener / primer mensaje": "Necesito un primer mensaje o frase de apertura para iniciar la conversación o el contacto.",
    "⚡ Escalación": "Necesito subir la tensión/interés en la conversación. Algo que lleve la dinámica a otro nivel.",
    "🔥 Sexualizar": "Quiero llevar la conversación a territorio sexual/físico de forma calibrada, sin ser torpe.",
    "🎯 Cerrar / proponer algo": "Necesito proponer un plan concreto: salir, encontrarnos, algo específico.",
    "🔄 Retomar / reiniciar": "La conversación quedó muerta o fría. Necesito reiniciarla sin parecer necesitado.",
    "📍 En persona (IRL)": "Estoy frente a ella o la voy a ver. Necesito algo para decir en persona.",
}


# ============ helpers ============

# ── Rate limit del login ───────────────────────────────────────────────────
# Un PIN de 4 dígitos son 10.000 combinaciones: sin throttling se rompe en
# minutos. El estado va a nivel proceso (no session_state) porque una sesión
# nueva / incógnito resetearía el contador y el límite no serviría de nada.
_MAX_FAILS = 5            # intentos libres por ventana
_WINDOW = 15 * 60         # ventana de conteo (s)
_MAX_LOCK = 15 * 60       # techo del backoff (s)


@st.cache_resource
def _login_attempts() -> dict[str, list[float]]:
    """{ip: [timestamps de fallos]} compartido por todas las sesiones del proceso."""
    return {}


def _client_ip() -> str:
    """IP real detrás del proxy de Fly. Si no se puede leer, cae a una clave global."""
    try:
        h = st.context.headers or {}
        ip = (h.get("Fly-Client-IP")
              or (h.get("X-Forwarded-For") or "").split(",")[0].strip())
        if ip:
            return ip
    except Exception:  # noqa: BLE001
        pass
    return "_global"


def _lock_remaining(ip: str) -> int:
    """Segundos que faltan para poder reintentar. 0 = libre."""
    now = time.time()
    fails = [t for t in _login_attempts().get(ip, []) if now - t < _WINDOW]
    _login_attempts()[ip] = fails
    if len(fails) < _MAX_FAILS:
        return 0
    # Backoff exponencial desde el último fallo: 30s, 60s, 120s... tope 15 min.
    penalty = min(30 * (2 ** (len(fails) - _MAX_FAILS)), _MAX_LOCK)
    return max(0, int(fails[-1] + penalty - now))


def require_login() -> None:
    """Gate de PIN con throttling. Sin esto la app queda abierta a internet."""
    if not settings.app_password:
        return  # sin PIN configurado (uso local)
    if st.session_state.get("_authed"):
        return

    st.title("🔒 Coach Asistente")
    ip = _client_ip()
    waiting = _lock_remaining(ip)

    if waiting:
        mins, secs = divmod(waiting, 60)
        st.error(f"Demasiados intentos fallidos. Probá de nuevo en {mins}m {secs:02d}s.")
        st.stop()

    pin = st.text_input("PIN", type="password", key="_pin_input")
    if st.button("Entrar", type="primary"):
        if hmac.compare_digest(pin.strip(), settings.app_password):
            _login_attempts().pop(ip, None)
            st.session_state["_authed"] = True
            st.rerun()
        else:
            _login_attempts().setdefault(ip, []).append(time.time())
            time.sleep(1)  # frena el guessing automatizado
            left = _MAX_FAILS - len(_login_attempts()[ip])
            if left > 0:
                st.error(f"PIN incorrecto. Te quedan {left} intento/s.")
            else:
                st.error("PIN incorrecto. Cuenta bloqueada temporalmente.")
    st.stop()


@st.cache_resource(show_spinner="⏳ Cargando el cerebro del coach (solo la primera vez, ~3 min). Después cada consulta tarda segundos.")
def warmup() -> bool:
    """Carga Qdrant + bge-m3 una sola vez, antes de que el usuario pida nada.

    st.cache_resource serializa las llamadas concurrentes, así que garantiza una
    única construcción aunque haya varias sesiones abiertas a la vez.
    """
    if not settings.qdrant_url:
        from coach_brain.kb.index import get_qdrant
        get_qdrant()
    from coach_brain.kb.embed import embed_one, get_model
    get_model()
    # Construir el modelo NO materializa los pesos: la primera inferencia real
    # los pagina desde la imagen y tardaba ~145s, dentro de la primera consulta
    # del usuario. Con un encode dummy ese costo se paga acá (con spinner) y
    # las consultas pasan a ~0.4s.
    embed_one("calentando el modelo")
    return True


def _client() -> Anthropic:
    if not settings.anthropic_api_key:
        st.error("ANTHROPIC_API_KEY no configurada en .env")
        st.stop()
    return Anthropic(api_key=settings.anthropic_api_key)


# Delimitan la seccion de sugerencias dentro de la respuesta del coach.
_SUGG_START = re.compile(r"l[íi]neas posibles|opciones", re.I)
_SUGG_END = re.compile(
    r"cu[áa]ndo usar|qu[ée] evitar|c[óo]mo entregarlo|nivel de riesgo|veredicto|riesgo\s*:", re.I)


def extract_quoted_lines(answer: str) -> list[str]:
    """Saca SOLO las frases sugeridas, para mostrarlas con botón de copiar.

    Escanear la respuesta entera traía comillas de dos lugares equivocados: los
    mensajes de ella que el diagnóstico cita como evidencia, y los ejemplos de
    "qué evitar". En el celular este bloque es la interfaz principal, así que
    ofrecer para copiar una línea que hay que evitar es peor que no ofrecer nada.
    """
    scope = answer
    m = _SUGG_START.search(answer)
    if m:
        rest = answer[m.end():]
        end = _SUGG_END.search(rest)
        scope = rest[:end.start()] if end else rest
    found = re.findall(r'["“]([^"”\n]{12,300})["”]', scope)
    seen, out = set(), []
    for line in found:
        s = line.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:6]


def render_answer(answer: str, key_prefix: str) -> None:
    """Muestra la respuesta + las líneas sueltas en bloques copiables (1 tap en celu)."""
    lines = extract_quoted_lines(answer)
    if lines:
        st.markdown("#### 📋 Líneas para copiar")
        for i, line in enumerate(lines):
            st.code(line, language=None)
        st.divider()
    st.markdown(answer)


def _load_system_prompt() -> str:
    return (settings.prompts_path / "system_coach.md").read_text(encoding="utf-8")


def _load_situational_prompt() -> str:
    return (settings.prompts_path / "situational_lines.md").read_text(encoding="utf-8")


def _notes_path() -> Path:
    p = settings.data_path
    p.mkdir(parents=True, exist_ok=True)
    return p / "personal_notes.txt"


def _load_personal_notes() -> str:
    p = _notes_path()
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _save_personal_note(text: str) -> None:
    note = text.strip()
    if not note:
        return
    p = _notes_path()
    existing = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    sep = "\n\n---\n\n"
    p.write_text((existing + sep + note) if existing else note, encoding="utf-8")


def _delete_personal_notes() -> None:
    p = _notes_path()
    if p.exists():
        p.write_text("", encoding="utf-8")


def _image_to_b64(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "image/png", base64.standard_b64encode(buf.getvalue()).decode("ascii")


@st.cache_resource
def _get_ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr_chat_from_image(img: Image.Image) -> tuple[str, str]:
    import numpy as np
    from coach_brain.ingest.whatsapp_images_ocr import _is_noise

    ocr = _get_ocr()
    rgb = img.convert("RGB")
    width = rgb.size[0]
    res, _ = ocr(np.array(rgb))
    if not res:
        return "", ""
    items = sorted(res, key=lambda r: r[0][0][1])
    raw_lines = [(t or "").strip() for _b, t, _c in items if (t or "").strip()]
    turns: list[list] = []
    for box, txt, _conf in items:
        txt = (txt or "").strip()
        if not txt or _is_noise(txt):
            continue
        # El borde DERECHO decide, no el izquierdo: un mensaje propio largo envuelve
        # y arranca a la izquierda del medio, quedando mal etiquetado como de ella.
        right_edge = max(pt[0] for pt in box)
        who = "yo" if right_edge > width * 0.82 else "ella"
        if turns and turns[-1][0] == who:
            turns[-1][1] += " " + txt
        else:
            turns.append([who, txt])
    chat = "\n".join(f"[{w}] {t}" for w, t in turns)
    return chat, "\n".join(raw_lines)


def guess_contact_name(text: str) -> str:
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

    # complete_fast elige solo: Ollama local -> proveedor gratis -> Haiku.
    return _clean(complete_fast(
        system="Sos un extractor. Respondés únicamente con el dato pedido.",
        user=prompt, max_tokens=30, temperature=0.0))


def parse_chat_from_image(client: Anthropic, img: Image.Image) -> dict:
    media_type, b64 = _image_to_b64(img)
    resp = client.messages.create(
        model=settings.model_fast,
        max_tokens=2000, temperature=0,
        system=(
            "Sos un extractor de chats de WhatsApp/Telegram a partir de screenshots. "
            "Devolvés SOLO JSON con la forma "
            '{"messages":[{"sender":"ella"|"yo","text":"..."}], "notas":"..."}. '
            "Detectá quién es 'ella' por color/posición de burbuja. "
            "Si ves emojis, audios, stickers, marcalo. Si hay 'visto'/'escribiendo', anotalo en notas."
        ),
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
            {"type": "text", "text": "Extraé el chat como JSON."},
        ]}],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
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


def call_coach(backend: str, model: str, system_prompt: str, chat_text: str, user_q: str,
               retrieved: dict | None = None, mode_directive: str = "") -> str:
    coach_block = ""
    if retrieved:
        coach_block = f"""
Material del coach recuperado para ESTE caso:

PRINCIPIOS:
{retrieved.get('principles_retrieved', '')}

SITUACIONES SIMILARES:
{retrieved.get('situations_retrieved', '')}

FRASES DE ESTILO:
{retrieved.get('style_retrieved', '')}
"""
    mode_block = ""
    if mode_directive:
        mode_block = f"\nMODO DE RESPUESTA: {mode_directive}\nLas 3 líneas tienen que respetar ESTE modo.\n"

    user_content = f"""\
Conversación actual:
```
{chat_text}
```
{coach_block}{mode_block}
Pregunta de Ilan: {user_q or "¿qué le respondo?"}

Respondé con el formato del coach. El Diagnóstico y las 3 líneas deben apoyarse en el material de arriba.
"""
    return llm_complete(system=system_prompt, user=user_content,
                        backend=backend, model=model,
                        max_tokens=2000, temperature=0.7)


def call_situational(backend: str, model: str, situation: str, need_type: str,
                     mode_directive: str, retrieved: dict | None,
                     contact_profile: str, personal_notes: str) -> str:
    """Genera líneas/frases para una situación concreta (sin chat que analizar)."""
    coach_block = ""
    if retrieved:
        coach_block = f"""
Material del coach recuperado:

PRINCIPIOS:
{retrieved.get('principles_retrieved', '')}

SITUACIONES SIMILARES:
{retrieved.get('situations_retrieved', '')}

FRASES DE ESTILO (tono de referencia):
{retrieved.get('style_retrieved', '')}
"""
    notes_block = ""
    if personal_notes.strip():
        notes_block = f"\nFRASES/NOTAS PERSONALES DE ILAN (úsalas si encajan):\n{personal_notes[:6000]}\n"

    user_content = f"""\
SITUACIÓN:
{situation}

QUÉ NECESITO:
{need_type}

MODO: {mode_directive}
{coach_block}
PERFIL DE ELLA:
{contact_profile or "(sin información guardada)"}
{notes_block}
Generá las líneas siguiendo el formato del sistema.
"""
    system_prompt = _load_situational_prompt() \
        .replace("{user_name}", "Ilan") \
        .replace("{principles_core}", PRINCIPLES_CORE)

    return llm_complete(system=system_prompt, user=user_content,
                        backend=backend, model=model,
                        max_tokens=2000, temperature=0.8)


# ============ UI ============

st.set_page_config(page_title="Coach Asistente", layout="centered",
                   initial_sidebar_state="collapsed")

# Antes del login: el manifest tiene que estar tambien en la pantalla del PIN,
# si no el celular no ofrece "Agregar a inicio" hasta despues de entrar.
inject_pwa()

require_login()
st.title("🎯 Coach Asistente")
warmup()

# ── Sidebar (aplica a todos los tabs) ──────────────────────────────────────
with st.sidebar:
    st.subheader("Configuración")
    st.text_input("Tu nombre", value="Ilan", key="user_name")
    use_rag = st.checkbox("Usar RAG (recuperar del coach)", value=True)
    st.divider()
    st.subheader("Modelo")
    _resp_opts: dict = {}
    if settings.ollama_enabled:
        _resp_opts[f"🟢 Ollama (local, gratis) — {settings.ollama_model}"] = ("ollama", settings.ollama_model)
    # Proveedores gratis con key cargada: van primero para que sean el default.
    for _p in available_providers():
        _resp_opts[f"🆓 {_p.label}"] = (_p.key, _p.model)
    if settings.anthropic_api_key:
        _resp_opts[f"💳 Sonnet — {settings.model_main}"] = ("anthropic", settings.model_main)
        _resp_opts[f"💳 Haiku — {settings.model_fast}"] = ("anthropic", settings.model_fast)

    if not _resp_opts:
        st.error(
            "No hay ningún modelo configurado. Cargá una API key gratis "
            "(Gemini, Groq, Cerebras u OpenRouter) o créditos de Anthropic."
        )
        st.stop()

    _resp_keys = list(_resp_opts)
    resp_choice = st.radio("¿Con qué genero la respuesta?", _resp_keys, index=0)
    resp_backend, resp_model = _resp_opts[resp_choice]
    st.divider()
    st.subheader("Leer screenshot")
    _vision_opts = ["🟢 OCR local (gratis)"]
    if settings.anthropic_api_key:
        # La visión por API es solo de Anthropic; sin key no tiene sentido
        # ofrecerla porque falla al momento de usarla.
        _vision_opts.append("💳 Haiku vision (pago)")
    vision_choice = st.radio("¿Cómo leo las imágenes?", _vision_opts, index=0)
    vision_local = vision_choice.startswith("🟢")

# ── Tabs ───────────────────────────────────────────────────────────────────
# Etiquetas cortas: con los nombres largos las 3 tabs median 407px y no
# entraban en los 349 utiles de un celular de 375px.
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🎯 Líneas", "📝 Frases"])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Analizar Chat (lógica original, sin cambios)
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    # El resultado va ARRIBA: en celular las columnas se apilan y si no,
    # habría que scrollear todo el formulario cada vez para leer la respuesta.
    out_slot = st.container()

    with st.container():
        st.subheader("Input")
        uploaded = st.file_uploader("Screenshot del chat", type=["png", "jpg", "jpeg", "webp"],
                                    accept_multiple_files=True, key="upload_tab1")
        chat_text_manual = st.text_area(
            "O pegá el texto del chat",
            height=180,
            placeholder="[ella] hola\n[yo] hey\n[ella] qué hacés?",
            key="chat_manual_tab1",
        )
        _saved = list_contacts()
        _sel = st.selectbox("Chica (memoria)", ["➕ Nueva"] + [c["label"] for c in _saved],
                            key="contact_sel_tab1")
        contact_id = None
        new_contact = None
        auto_detect = False
        if _sel == "➕ Nueva":
            auto_detect = st.checkbox("🔎 Detectar la chica sola (del chat/foto)", value=True,
                                      key="autodetect_tab1")
            _nn = st.text_input("Nombre de la chica (vacío = autodetectar)",
                                placeholder="Mica", key="name_tab1")
            _ca, _cb = st.columns(2)
            _ape = _ca.text_input("Apellido (opc)", key="ape_tab1")
            _tag = _cb.text_input("Tag (opc)", placeholder="gimnasio / IG / Tinder", key="tag_tab1")
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

        user_q = st.text_input("Pregunta opcional", placeholder="¿qué le respondo?",
                               key="userq_tab1")
        st.write("**Modo de respuesta**")
        mode_choice = st.radio("Modo", list(MODES), horizontal=True,
                               label_visibility="collapsed", key="mode_tab1")
        st.caption(MODES[mode_choice])
        go = st.button("🚀 Analizar", type="primary", use_container_width=True, key="go_tab1")

    with out_slot:
        if go:
            st.subheader("Análisis")
            client = None
            chat_text = ""
            detect_source = ""

            if uploaded:
                # Varios screenshots = una conversación larga, en orden de subida.
                parts, raws = [], []
                for f in uploaded:
                    img = Image.open(f)
                    if vision_local:
                        with st.spinner(f"Leyendo {f.name} (OCR local, gratis)..."):
                            c, raw = ocr_chat_from_image(img)
                    else:
                        client = client or _client()
                        with st.spinner(f"Leyendo {f.name} (Haiku)..."):
                            parsed = parse_chat_from_image(client, img)
                            c = build_chat_text(parsed)
                            raw = c + " " + str(parsed.get("notas", ""))
                    if c:
                        parts.append(c)
                    raws.append(raw)
                chat_text = "\n".join(parts)
                detect_source = " ".join(raws)
                with st.expander(f"Chat extraído ({len(uploaded)} captura/s)"):
                    st.code(chat_text or "(no se detectó texto)")
            elif chat_text_manual.strip():
                chat_text = chat_text_manual.strip()
                detect_source = chat_text
            else:
                st.warning("Subí una imagen o pegá texto.")

            if chat_text:
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
                            with st.expander(
                                f"Recuperado: {len(result.principles)} ppios | "
                                f"{len(result.situations)} sits | {len(result.style)} frases"
                            ):
                                st.json({"principles": result.principles,
                                         "situations": result.situations,
                                         "style": result.style})
                        except Exception as e:
                            st.warning(f"RAG no disponible aún: {e}")

                _cmem = "(sin historial de esta persona)"
                if contact_id:
                    _m = get_contact(contact_id)
                    if _m and _m.get("profile"):
                        _cmem = f"Perfil de {_m['label']}:\n{_m['profile']}"

                system_prompt = _load_system_prompt() \
                    .replace("{user_name}", st.session_state.get("user_name", "Ilan")) \
                    .replace("{principles_core}", PRINCIPLES_CORE) \
                    .replace("{principles_retrieved}", retrieved_str["principles_retrieved"]) \
                    .replace("{situations_retrieved}", retrieved_str["situations_retrieved"]) \
                    .replace("{style_retrieved}", retrieved_str["style_retrieved"]) \
                    .replace("{contact_memory}", _cmem)

                _spin = "Pensando (Ollama local)..." if resp_backend == "ollama" else f"Pensando ({resp_model})..."
                answer = None
                with st.spinner(_spin):
                    try:
                        answer = call_coach(resp_backend, resp_model, system_prompt,
                                            chat_text, user_q,
                                            retrieved=retrieved_str if use_rag else None,
                                            mode_directive=f"{MODES[mode_choice]} {_HUMOR_NUDGE}")
                    except Exception as e:
                        st.error(f"Error generando la respuesta ({resp_backend}): {e}")

                if answer:
                    st.session_state["last_chat_answer"] = answer
                    render_answer(answer, "tab1")

                    cid = contact_id
                    if cid is None and new_contact and new_contact[0]:
                        cid = create_contact(*new_contact)
                    if cid:
                        _lbl = (get_contact(cid) or {}).get("label", "contacto")
                        with st.spinner(f"Guardando contexto de {_lbl}..."):
                            try:
                                prof = update_after_analysis(cid, chat_text, answer,
                                                             user=st.session_state.get("user_name", "Ilan"))
                                with st.expander(f"🧠 Memoria de {_lbl} actualizada"):
                                    st.markdown(prof)
                            except Exception as e:
                                st.caption(f"No pude actualizar memoria: {e}")
                    else:
                        st.caption("Tip: elegí o creá la chica arriba para que recuerde el contexto entre análisis.")

        elif st.session_state.get("last_chat_answer"):
            # Persistido: sin esto, tocar cualquier widget borraba la respuesta.
            st.subheader("Análisis")
            st.caption("↻ Último análisis")
            render_answer(st.session_state["last_chat_answer"], "tab1_cached")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Líneas & Situaciones
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Describí la situación → recibís líneas exactas para usar")
    st.caption("No necesitás tener un chat. Contame dónde estás / qué pasó / qué querés lograr.")

    res_slot = st.container()

    with st.container():
        situation = st.text_area(
            "🎬 Situación",
            height=180,
            placeholder=(
                "Ej: Estoy en un bar, es la primera vez que la veo. Nos presentaron hace 10 min, "
                "hay química pero no arrancó nada. Quiero romper el hielo de forma interesante.\n\n"
                "O: Le escribo por primera vez después de cruzarnos en el gym hace dos días. "
                "Me miró bastante. No tenemos contacto previo."
            ),
            key="situation_input",
        )

        need_type_key = st.radio(
            "¿Qué necesitás?",
            list(NEED_TYPES),
            key="need_type",
        )
        st.caption(NEED_TYPES[need_type_key])

        st.write("**Tono / Modo**")
        sit_mode = st.radio("Modo líneas", list(MODES), horizontal=True,
                            label_visibility="collapsed", key="sit_mode")
        st.caption(MODES[sit_mode])

        _saved_sit = list_contacts()
        _sel_sit = st.selectbox(
            "¿Con quién? (opcional — para usar su perfil)",
            ["— Sin seleccionar —"] + [c["label"] for c in _saved_sit],
            key="contact_sit",
        )
        sit_contact_id = None
        if _sel_sit != "— Sin seleccionar —":
            sit_contact_id = next((c["id"] for c in _saved_sit if c["label"] == _sel_sit), None)

        go_sit = st.button("🔥 Generar líneas", type="primary",
                           use_container_width=True, key="go_sit")

    with res_slot:
        if go_sit and not situation.strip():
            st.warning("Describí la situación primero.")
        elif go_sit:
            st.subheader("Líneas")
            sit_retrieved = None
            if use_rag:
                with st.spinner("Buscando en el material del coach..."):
                    try:
                        result = retrieve(f"{situation} {need_type_key}")
                        sit_retrieved = result.format_for_prompt()
                        with st.expander(
                            f"Coach RAG: {len(result.principles)} ppios | "
                            f"{len(result.situations)} sits | {len(result.style)} frases"
                        ):
                            st.json({"principles": result.principles,
                                     "situations": result.situations,
                                     "style": result.style})
                    except Exception as e:
                        st.warning(f"RAG no disponible: {e}")

            sit_profile = ""
            if sit_contact_id:
                _m = get_contact(sit_contact_id)
                if _m and _m.get("profile"):
                    sit_profile = f"Perfil de {_m['label']}:\n{_m['profile']}"

            personal_notes = _load_personal_notes()

            _spin2 = "Pensando (Ollama local)..." if resp_backend == "ollama" else f"Generando ({resp_model})..."
            lines_answer = None
            with st.spinner(_spin2):
                try:
                    lines_answer = call_situational(
                        resp_backend, resp_model,
                        situation=situation,
                        need_type=f"{need_type_key}: {NEED_TYPES[need_type_key]}",
                        mode_directive=f"{MODES[sit_mode]} {_HUMOR_NUDGE}",
                        retrieved=sit_retrieved,
                        contact_profile=sit_profile,
                        personal_notes=personal_notes,
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

            if lines_answer:
                st.session_state["last_lines_answer"] = lines_answer
                render_answer(lines_answer, "tab2")
                if personal_notes.strip():
                    st.caption("💡 Tus frases personales (📝 Mis Frases) fueron incluidas en el contexto.")

        elif st.session_state.get("last_lines_answer"):
            st.subheader("Líneas")
            st.caption("↻ Últimas generadas")
            render_answer(st.session_state["last_lines_answer"], "tab2_cached")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Mis Frases (notas personales / iCloud Notes import)
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📝 Mis Frases & Notas Personales")
    st.caption(
        "Guardá acá tus frases favoritas, openers que te funcionaron, estilos de texto, "
        "cosas del coach que querés tener a mano. "
        "Se incluyen automáticamente en el Tab 🎯 al generar líneas. "
        "**¿Tenés notas en iCloud?** Abrí Notes en el celu / Mac, copiá el texto y pegalo abajo."
    )

    col_n1, col_n2 = st.columns([1, 1])

    with col_n1:
        st.write("**Agregar nueva nota / frase**")
        new_note = st.text_area(
            "Pegá texto de iCloud Notes, frases propias, openers, etc.",
            height=220,
            placeholder=(
                "Ej:\n"
                "- 'Mirá, te lo digo de una: me gustaste. Qué hacemos con eso.'\n"
                "- Opener que me funcionó en el gym: llegué y le pregunté por los auriculares.\n"
                "- Del coach: cuando hay frialdad, silencio primero. No llenar el espacio.\n\n"
                "Podés pegar nota entera de iCloud, bullets, lo que sea."
            ),
            key="new_note_input",
        )
        save_btn = st.button("💾 Guardar nota", use_container_width=True, key="save_note")
        if save_btn:
            if new_note.strip():
                _save_personal_note(new_note)
                st.success("✅ Guardado. Se usará en el próximo análisis de situación.")
                st.rerun()
            else:
                st.warning("Escribí algo primero.")

    with col_n2:
        st.write("**Notas guardadas**")
        current_notes = _load_personal_notes()
        if current_notes:
            st.text_area(
                "Contenido actual (solo lectura acá)",
                value=current_notes,
                height=220,
                disabled=True,
                key="notes_display",
            )
            st.caption(f"Total: {len(current_notes)} caracteres")
            if st.button("🗑️ Borrar todas las notas", key="clear_notes"):
                _delete_personal_notes()
                st.success("Notas borradas.")
                st.rerun()
        else:
            st.info("No hay notas guardadas todavía. Agregá algo en el panel izquierdo.")

    st.divider()
    st.caption(
        "**Tip:** Para importar notas de iCloud en iPhone → abrí la nota → tocá ··· (arriba a la derecha) → "
        "Compartir → Copiar → pegá acá. En Mac: Editar → Seleccionar todo → Copiar."
    )
