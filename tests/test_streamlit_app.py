"""La interfaz: los helpers de `app/streamlit_app.py`.

El archivo no se puede importar: abajo del marcador `# ============ UI ===`
corre la app entera (`require_login()`, `warmup()`), y `warmup()` carga
Qdrant y bge-m3. Así que se compila **solo la mitad de arriba** en un módulo
propio. Una prueba se encarga de que el marcador siga existiendo una sola
vez, para que este archivo no quede probando otra cosa en silencio.

Lo que se cubre: la firma de la cookie y el throttling del PIN (la app está
publicada en internet, es lo único que la separa de cualquiera), el OCR del
chat, los errores de proveedor traducidos, y el armado de los prompts.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import time
import types
from pathlib import Path

import pytest
from PIL import Image

from coach_brain.settings import settings

FUENTE = Path(__file__).resolve().parents[1] / "src" / "coach_brain" / "app" / "streamlit_app.py"
MARCA = "# ============ UI ============"


@pytest.fixture(scope="module")
def app():
    """El módulo con los helpers, sin la UI."""
    texto = FUENTE.read_text(encoding="utf-8")
    cabeza = texto.split(MARCA, 1)[0]
    mod = types.ModuleType("streamlit_app_helpers")
    mod.__file__ = str(FUENTE)
    exec(compile(cabeza, str(FUENTE), "exec"), mod.__dict__)  # noqa: S102
    return mod


def test_el_marcador_de_la_ui_sigue_siendo_unico():
    """Si se mueve o se duplica, el resto de este archivo prueba otra cosa."""
    assert FUENTE.read_text(encoding="utf-8").count(MARCA) == 1


def test_la_mitad_de_arriba_no_ejecuta_la_app(app):
    """Nada de `st.set_page_config` ni `warmup()` del lado de los helpers."""
    assert "set_page_config" not in FUENTE.read_text(encoding="utf-8").split(MARCA)[0]


# ===========================================================================
# La cookie firmada
# ===========================================================================

@pytest.fixture
def pin(monkeypatch):
    monkeypatch.setattr(settings, "app_password", "1234")
    return "1234"


def test_un_token_recien_firmado_vale(app, pin):
    assert app._token_valid(app._sign_token(int(time.time()) + 60)) is True


def test_un_token_vencido_no_vale(app, pin):
    """El vencimiento va DENTRO de la firma; acá se comprueba que además se
    compara contra el reloj."""
    assert app._token_valid(app._sign_token(int(time.time()) - 1)) is False


def test_estirar_el_vencimiento_invalida_la_firma(app, pin):
    """El ataque obvio: editar la cookie para que no venza nunca."""
    token = app._sign_token(int(time.time()) + 60)
    _exp, sig = token.split(".", 1)
    estirado = f"{int(time.time()) + 10 ** 9}.{sig}"
    assert app._token_valid(estirado) is False


def test_una_firma_inventada_no_vale(app, pin):
    assert app._token_valid(f"{int(time.time()) + 60}.{'0' * 32}") is False


def test_un_token_firmado_con_otro_pin_no_vale(app, monkeypatch):
    """Cambiar el PIN tiene que invalidar las sesiones viejas."""
    monkeypatch.setattr(settings, "app_password", "1234")
    token = app._sign_token(int(time.time()) + 60)
    monkeypatch.setattr(settings, "app_password", "9999")
    assert app._token_valid(token) is False


@pytest.mark.parametrize("basura", ["", "abc", "....", "99999999", None, "x.y", "abc.def"])
def test_una_cookie_con_basura_no_rompe(app, pin, basura):
    assert app._token_valid(basura) is False


def test_la_firma_es_hmac_del_vencimiento(app, pin):
    """Clavado contra la construcción, no contra sí misma: si alguien cambiara
    el hash por algo reversible, esto falla."""
    exp = int(time.time()) + 60
    esperado = hmac.new(b"1234", str(exp).encode(), hashlib.sha256).hexdigest()[:32]
    assert app._sign_token(exp) == f"{exp}.{esperado}"


# ===========================================================================
# El throttling del PIN
# ===========================================================================

@pytest.fixture
def intentos(app):
    """La tabla de fallos, vacía antes y después (es a nivel proceso)."""
    app._login_attempts().clear()
    yield app._login_attempts()
    app._login_attempts().clear()


def test_sin_fallos_no_hay_espera(app, intentos):
    assert app._lock_remaining("1.2.3.4") == 0


def test_cuatro_fallos_todavia_dejan_entrar(app, intentos):
    """Un PIN de 4 dígitos son 10.000 combinaciones: el límite es lo único que
    lo hace impracticable, pero un tipeo mal no puede bloquear."""
    intentos["1.2.3.4"] = [time.time()] * 4
    assert app._lock_remaining("1.2.3.4") == 0


@pytest.mark.parametrize("fallos,esperado", [(5, 30), (6, 60), (7, 120), (8, 240)])
def test_el_backoff_duplica_desde_el_quinto_fallo(app, intentos, fallos, esperado):
    intentos["1.2.3.4"] = [time.time()] * fallos
    assert app._lock_remaining("1.2.3.4") == pytest.approx(esperado, abs=1)


def test_el_backoff_tiene_techo_de_quince_minutos(app, intentos):
    """Sin techo, veinte fallos bloquearían la IP por años."""
    intentos["1.2.3.4"] = [time.time()] * 40
    assert app._lock_remaining("1.2.3.4") == pytest.approx(900, abs=1)


def test_los_fallos_viejos_se_olvidan(app, intentos):
    """La ventana es de 15 minutos: fallos de ayer no bloquean hoy."""
    intentos["1.2.3.4"] = [time.time() - 3600] * 10
    assert app._lock_remaining("1.2.3.4") == 0
    assert intentos["1.2.3.4"] == []  # y se limpian de la tabla


def test_el_bloqueo_es_por_ip(app, intentos):
    intentos["1.2.3.4"] = [time.time()] * 10
    assert app._lock_remaining("5.6.7.8") == 0


def test_la_espera_baja_con_el_tiempo(app, intentos):
    intentos["1.2.3.4"] = [time.time() - 20] * 5
    assert app._lock_remaining("1.2.3.4") == pytest.approx(10, abs=1)


# ===========================================================================
# La IP y la cookie del request
# ===========================================================================

def fake_st(headers):
    """Un `st` mínimo: `_client_ip` y `_cookie_token` solo tocan el contexto."""
    return types.SimpleNamespace(context=types.SimpleNamespace(headers=headers))


def test_la_ip_sale_del_header_de_fly(app, monkeypatch):
    monkeypatch.setattr(app, "st", fake_st({"Fly-Client-IP": "1.2.3.4"}))
    assert app._client_ip() == "1.2.3.4"


def test_sin_fly_usa_el_primero_del_x_forwarded_for(app, monkeypatch):
    """El resto de la cadena lo pone el cliente: solo el primero sirve."""
    monkeypatch.setattr(app, "st", fake_st({"X-Forwarded-For": "9.9.9.9, 10.0.0.1"}))
    assert app._client_ip() == "9.9.9.9"


def test_fly_le_gana_al_x_forwarded_for(app, monkeypatch):
    monkeypatch.setattr(app, "st", fake_st(
        {"Fly-Client-IP": "1.2.3.4", "X-Forwarded-For": "9.9.9.9"}))
    assert app._client_ip() == "1.2.3.4"


def test_sin_headers_el_limite_pasa_a_ser_global(app, monkeypatch):
    """Peor es no limitar: todos comparten el contador."""
    monkeypatch.setattr(app, "st", fake_st({}))
    assert app._client_ip() == "_global"


def test_si_el_contexto_explota_tambien_cae_a_global(app, monkeypatch):
    class Explota:
        @property
        def headers(self):
            raise RuntimeError("sin runtime")

    monkeypatch.setattr(app, "st", types.SimpleNamespace(context=Explota()))
    assert app._client_ip() == "_global"


def test_la_cookie_se_saca_de_entre_las_demas(app, monkeypatch):
    monkeypatch.setattr(app, "st", fake_st({"Cookie": "otra=x; coach_auth=tok123; mas=y"}))
    assert app._cookie_token() == "tok123"


def test_sin_cookie_devuelve_vacio(app, monkeypatch):
    monkeypatch.setattr(app, "st", fake_st({"Cookie": "otra=x"}))
    assert app._cookie_token() == ""


def test_sin_header_cookie_devuelve_vacio(app, monkeypatch):
    monkeypatch.setattr(app, "st", fake_st({}))
    assert app._cookie_token() == ""


# ===========================================================================
# Los errores del proveedor
# ===========================================================================

@pytest.mark.parametrize("crudo,pista", [
    ("your credit balance is too low", "créditos"),
    ("billing error", "créditos"),
    ("429 Too Many Requests", "cuota"),
    ("RESOURCE_EXHAUSTED", "cuota"),
    ("quota exceeded", "cuota"),
    ("rate limit reached", "cuota"),
    ("invalid api key", "API key"),
    ("401 Unauthorized", "API key"),
    ("INVALID_ARGUMENT: key", "API key"),
])
def test_el_error_del_proveedor_dice_que_hacer(app, crudo, pista):
    """El caso de siempre (sin créditos) llegaba como un volcado de JSON donde
    no se veía que la solución es elegir otro modelo en el panel."""
    msg = app.friendly_llm_error("anthropic", RuntimeError(crudo))
    assert pista in msg
    assert "anthropic" in msg


def test_un_error_desconocido_se_muestra_crudo(app):
    """No se traga el detalle: sin él no habría con qué debuggear."""
    msg = app.friendly_llm_error("groq", RuntimeError("boom raro"))
    assert "boom raro" in msg and "groq" in msg


# ===========================================================================
# Las líneas para copiar
# ===========================================================================

def test_solo_salen_las_lineas_de_la_seccion_de_sugerencias(app):
    """Escanear la respuesta entera ofrecía para copiar los mensajes de ella y
    los ejemplos de 'qué evitar', que es peor que no ofrecer nada."""
    respuesta = (
        'Diagnóstico: ella dijo "esto no lo tendría que ofrecer nunca jamás".\n'
        "Líneas posibles:\n"
        '1. "Te hago una propuesta medio audaz igual"\n'
        "Qué evitar:\n"
        '- "esta frase no se puede ofrecer para copiar"\n'
    )
    assert app.extract_quoted_lines(respuesta) == ["Te hago una propuesta medio audaz igual"]


def test_sin_encabezado_de_sugerencias_escanea_todo(app):
    """Un formato distinto no puede dejar la sección vacía."""
    assert app.extract_quoted_lines('dijo "una frase bastante larga acá"') == \
        ["una frase bastante larga acá"]


def test_las_comillas_curvas_tambien_cuentan(app):
    """El LLM mezcla rectas y curvas en la misma respuesta."""
    assert app.extract_quoted_lines("opciones: “una linea con curvas”") == \
        ["una linea con curvas"]


def test_una_frase_corta_no_es_una_linea(app):
    """Menos de 12 caracteres es una palabra citada, no una línea para mandar."""
    assert app.extract_quoted_lines('opciones: "corto"') == []


def test_las_repetidas_salen_una_sola_vez(app):
    r = 'opciones: "una misma linea igual" y "Una Misma Linea Igual"'
    assert app.extract_quoted_lines(r) == ["una misma linea igual"]


def test_como_maximo_seis_lineas(app):
    r = "opciones:" + "".join(f'"linea numero {i} de prueba" ' for i in range(10))
    assert len(app.extract_quoted_lines(r)) == 6


def test_una_cita_de_varias_lineas_no_se_toma(app):
    """El regex corta en el salto: una comilla sin cerrar se tragaba media
    respuesta como si fuera una línea."""
    assert app.extract_quoted_lines('opciones: "arranca acá\ny sigue allá"') == []


def test_el_corte_respeta_el_veredicto(app):
    r = 'opciones:\n"una linea buena para usar"\nVeredicto: "esto no va"'
    assert app.extract_quoted_lines(r) == ["una linea buena para usar"]


# ===========================================================================
# El encabezado de WhatsApp y el OCR
# ===========================================================================

@pytest.mark.parametrize("texto", [
    "Meli en línea", "Meli en linea", "online", "últ. vez hoy",
    "ultima vez a las 3", "escribiendo...", "typing", "last seen today",
    "activo hace 2 h",
])
def test_el_regex_reconoce_el_estado_del_contacto(app, texto):
    assert app._HEADER_STATUS.search(texto)


@pytest.mark.parametrize("texto", ["Meli", "hola como estas", "mañana te escribo"])
def test_un_mensaje_cualquiera_no_parece_encabezado(app, texto):
    assert not app._HEADER_STATUS.search(texto)


def caja(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def poner_ocr(app, monkeypatch, items):
    monkeypatch.setattr(app, "_get_ocr", lambda: (lambda _arr: (items, None)))


def imagen(w=100, h=100):
    return Image.new("RGB", (w, h), "white")


def test_el_borde_derecho_decide_de_quien_es_el_mensaje(app, monkeypatch):
    """Un mensaje propio largo envuelve y arranca a la izquierda del medio: con
    el borde izquierdo quedaba etiquetado como de ella."""
    poner_ocr(app, monkeypatch, [
        (caja(5, 20, 95, 30), "mensaje propio que envuelve toda la pantalla", 0.9),
        (caja(5, 40, 50, 50), "mensaje corto de ella", 0.9),
    ])
    chat, _crudo = app.ocr_chat_from_image(imagen())
    assert chat == ("[yo] mensaje propio que envuelve toda la pantalla\n"
                    "[ella] mensaje corto de ella")


def test_el_encabezado_se_descarta_si_dice_en_linea(app, monkeypatch):
    poner_ocr(app, monkeypatch, [
        (caja(10, 2, 60, 9), "Meli en línea", 0.9),
        (caja(5, 40, 50, 50), "hola que haces", 0.9),
    ])
    chat, _ = app.ocr_chat_from_image(imagen())
    assert chat == "[ella] hola que haces"


def test_sin_marcador_de_estado_el_encabezado_se_conserva(app, monkeypatch):
    """Perder un mensaje real de una captura recortada es peor que dejar colado
    el nombre del contacto."""
    poner_ocr(app, monkeypatch, [
        (caja(5, 2, 50, 9), "no me contestaste nada", 0.9),
        (caja(5, 40, 50, 50), "hola que haces", 0.9),
    ])
    chat, _ = app.ocr_chat_from_image(imagen())
    assert "no me contestaste nada" in chat


def test_un_texto_abajo_nunca_cuenta_como_encabezado(app, monkeypatch):
    """La franja es el 12% de arriba: un 'en línea' al pie no borra nada."""
    poner_ocr(app, monkeypatch, [
        (caja(5, 40, 50, 50), "hola que haces", 0.9),
        (caja(5, 90, 50, 98), "en línea", 0.9),
    ])
    chat, _ = app.ocr_chat_from_image(imagen())
    assert "hola que haces" in chat


def test_las_lineas_seguidas_del_mismo_lado_se_unen(app, monkeypatch):
    poner_ocr(app, monkeypatch, [
        (caja(5, 20, 50, 30), "primera parte del mensaje", 0.9),
        (caja(5, 32, 50, 42), "segunda parte del mensaje", 0.9),
    ])
    chat, _ = app.ocr_chat_from_image(imagen())
    assert chat == "[ella] primera parte del mensaje segunda parte del mensaje"


def test_el_orden_sale_por_la_coordenada_y(app, monkeypatch):
    """RapidOCR no garantiza orden: sin ordenar, el chat salía mezclado."""
    poner_ocr(app, monkeypatch, [
        (caja(5, 60, 50, 70), "esto va segundo del todo", 0.9),
        (caja(5, 20, 50, 30), "esto va primero del todo", 0.9),
    ])
    chat, _ = app.ocr_chat_from_image(imagen())
    assert chat.index("primero") < chat.index("segundo")


def test_el_texto_crudo_guarda_el_ruido_que_el_chat_descarta(app, monkeypatch):
    """El etiquetado de hablante es aproximado; el crudo es lo fiable."""
    poner_ocr(app, monkeypatch, [
        (caja(5, 20, 50, 30), "8:33", 0.9),
        (caja(5, 40, 50, 50), "hola que haces", 0.9),
    ])
    chat, crudo = app.ocr_chat_from_image(imagen())
    assert "8:33" in crudo
    assert "8:33" not in chat


def test_una_imagen_sin_texto_no_devuelve_nada(app, monkeypatch):
    poner_ocr(app, monkeypatch, [])
    assert app.ocr_chat_from_image(imagen()) == ("", "")


# ===========================================================================
# El nombre del contacto
# ===========================================================================

def test_un_texto_vacio_no_llama_al_modelo(app, monkeypatch):
    """Cuesta plata (o una carga de Ollama) y no hay nada que mirar."""
    def explota(**kw):
        raise AssertionError("no tendría que llamar al modelo")

    monkeypatch.setattr(app, "complete_fast", explota)
    assert app.guess_contact_name("   ") == ""


@pytest.mark.parametrize("respuesta,esperado", [
    ("Meli", "Meli"),
    ('"Meli"', "Meli"),
    ("  Meli  ", "Meli"),
    ("Meli\nes la chica del gym", "Meli"),
    ("desconocido", ""),
    ("Desconocido.", ""),
    ("no se puede desconocer el nombre", ""),
    ("", ""),
    ("x" * 41, ""),
])
def test_el_nombre_se_limpia(app, monkeypatch, respuesta, esperado):
    monkeypatch.setattr(app, "complete_fast", lambda **kw: respuesta)
    assert app.guess_contact_name("un chat cualquiera") == esperado


def test_el_prompt_recorta_el_chat(app, monkeypatch):
    """1500 caracteres alcanzan para el encabezado y sobra: mandar el chat
    entero multiplica el costo de cada análisis."""
    visto = {}
    monkeypatch.setattr(app, "complete_fast", lambda **kw: visto.update(kw) or "Meli")
    app.guess_contact_name("a" * 5000)
    assert "a" * 1500 in visto["user"]
    assert "a" * 1501 not in visto["user"]
    assert visto["max_tokens"] == 30
    assert visto["temperature"] == 0.0


# ===========================================================================
# El chat extraído por vision
# ===========================================================================

class FakeBlock:
    def __init__(self, text):
        self.text = text


class FakeClient:
    """Doble del cliente de Anthropic: devuelve lo que se le pase y guarda los
    kwargs de la llamada."""

    def __init__(self, texto):
        self.texto = texto
        self.kwargs: dict = {}
        self.messages = types.SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.kwargs = kw
        return types.SimpleNamespace(content=[FakeBlock(self.texto)])


def test_el_json_del_modelo_se_parsea(app):
    c = FakeClient('{"messages": [{"sender": "ella", "text": "hola"}], "notas": "visto"}')
    assert app.parse_chat_from_image(c, imagen()) == {
        "messages": [{"sender": "ella", "text": "hola"}], "notas": "visto"}


@pytest.mark.parametrize("envuelto", [
    '```json\n{"messages": [], "notas": "x"}\n```',
    '```\n{"messages": [], "notas": "x"}\n```',
])
def test_el_bloque_de_codigo_se_desenvuelve(app, envuelto):
    """El modelo devuelve el JSON en un fence la mitad de las veces."""
    assert app.parse_chat_from_image(FakeClient(envuelto), imagen())["notas"] == "x"


def test_un_json_roto_no_pierde_la_respuesta(app):
    """Queda en `notas` en vez de levantar: así se ve qué contestó el modelo."""
    salida = app.parse_chat_from_image(FakeClient("esto no es JSON"), imagen())
    assert salida == {"messages": [], "notas": "esto no es JSON"}


def test_la_imagen_va_como_png_en_base64(app):
    c = FakeClient('{"messages": []}')
    app.parse_chat_from_image(c, imagen())
    bloque = c.kwargs["messages"][0]["content"][0]
    assert bloque["source"]["media_type"] == "image/png"
    datos = base64.standard_b64decode(bloque["source"]["data"])
    assert Image.open(io.BytesIO(datos)).format == "PNG"


def test_la_extraccion_usa_el_modelo_rapido_y_sin_temperatura(app):
    c = FakeClient('{"messages": []}')
    app.parse_chat_from_image(c, imagen())
    assert c.kwargs["model"] == settings.model_fast
    assert c.kwargs["temperature"] == 0


def test_la_imagen_a_base64_devuelve_un_png(app):
    media, b64 = app._image_to_b64(imagen())
    assert media == "image/png"
    assert Image.open(io.BytesIO(base64.standard_b64decode(b64))).size == (100, 100)


# ===========================================================================
# El texto del chat para el prompt
# ===========================================================================

def test_el_chat_se_arma_con_el_hablante_adelante(app):
    parsed = {"messages": [{"sender": "ella", "text": "hola"},
                           {"sender": "yo", "text": "que tal"}]}
    assert app.build_chat_text(parsed) == "[ella] hola\n[yo] que tal"


def test_un_mensaje_sin_hablante_no_rompe(app):
    assert app.build_chat_text({"messages": [{"text": "hola"}]}) == "[?] hola"


def test_las_notas_del_extractor_van_al_final(app):
    salida = app.build_chat_text({"messages": [], "notas": "hay un audio"})
    assert salida.endswith("(notas del extractor: hay un audio)")


def test_sin_notas_no_se_agrega_el_bloque(app):
    assert "notas del extractor" not in app.build_chat_text({"messages": [], "notas": ""})


def test_un_dict_vacio_devuelve_un_texto_vacio(app):
    assert app.build_chat_text({}) == ""


# ===========================================================================
# Los prompts que se le mandan al coach
# ===========================================================================

@pytest.fixture
def llm(app, monkeypatch):
    """Espía de `llm_complete`: devuelve los kwargs de la última llamada."""
    visto: dict = {}
    monkeypatch.setattr(app, "llm_complete", lambda **kw: visto.update(kw) or "respuesta")
    return visto


RECUPERADO = {"principles_retrieved": "PRINCIPIO-X",
              "situations_retrieved": "SITUACION-X",
              "style_retrieved": "FRASE-X"}


def test_el_material_recuperado_entra_en_el_prompt(app, llm):
    app.call_coach("ollama", "m", "SISTEMA", "chat", "que digo", retrieved=RECUPERADO)
    for marca in ("PRINCIPIO-X", "SITUACION-X", "FRASE-X"):
        assert marca in llm["user"]


def test_sin_rag_no_se_mencionan_secciones_vacias(app, llm):
    """Un bloque 'PRINCIPIOS:' vacío le dice al modelo que no encontró nada."""
    app.call_coach("ollama", "m", "SISTEMA", "chat", "que digo", retrieved=None)
    assert "PRINCIPIOS:" not in llm["user"]


def test_el_modo_obliga_a_las_tres_lineas(app, llm):
    app.call_coach("ollama", "m", "SISTEMA", "chat", "q", mode_directive="DIRECTO")
    assert "MODO DE RESPUESTA: DIRECTO" in llm["user"]


def test_sin_modo_no_hay_bloque_de_modo(app, llm):
    app.call_coach("ollama", "m", "SISTEMA", "chat", "q")
    assert "MODO DE RESPUESTA" not in llm["user"]


def test_sin_pregunta_se_pone_la_de_siempre(app, llm):
    app.call_coach("ollama", "m", "SISTEMA", "chat", "")
    assert "¿qué le respondo?" in llm["user"]


def test_el_chat_va_en_un_fence(app, llm):
    """Sin delimitar, el chat se mezcla con las instrucciones del prompt."""
    app.call_coach("ollama", "m", "SISTEMA", "[ella] hola", "q")
    assert "```\n[ella] hola\n```" in llm["user"]


def test_el_backend_y_el_modelo_se_pasan_tal_cual(app, llm):
    app.call_coach("groq", "llama-x", "SISTEMA", "chat", "q")
    assert llm["backend"] == "groq"
    assert llm["model"] == "llama-x"
    assert llm["system"] == "SISTEMA"
    assert llm["max_tokens"] == 2000


def test_el_analisis_usa_temperatura_media(app, llm):
    app.call_coach("ollama", "m", "SISTEMA", "chat", "q")
    assert llm["temperature"] == 0.7


# --- situacional -----------------------------------------------------------

@pytest.fixture
def situacional(app, monkeypatch):
    monkeypatch.setattr(app, "_load_situational_prompt",
                        lambda: "Para {user_name}:\n{principles_core}")


def test_el_prompt_situacional_se_rellena(app, llm, situacional):
    """Si quedara `{principles_core}` sin reemplazar, el modelo recibe la
    plantilla cruda y responde cualquier cosa."""
    app.call_situational("ollama", "m", "en el gym", "una apertura", "JUGUETON",
                         None, "", "")
    assert "{user_name}" not in llm["system"]
    assert "{principles_core}" not in llm["system"]
    assert "Ilan" in llm["system"]
    assert llm["system"].count(app.PRINCIPLES_CORE) == 1


def test_la_situacion_y_el_modo_van_en_el_prompt(app, llm, situacional):
    app.call_situational("ollama", "m", "en el gym", "una apertura", "JUGUETON",
                         None, "", "")
    assert "en el gym" in llm["user"]
    assert "una apertura" in llm["user"]
    assert "MODO: JUGUETON" in llm["user"]


def test_sin_perfil_de_ella_se_dice_que_no_hay(app, llm, situacional):
    """Un bloque vacío hacía que el modelo inventara un perfil."""
    app.call_situational("ollama", "m", "s", "n", "m", None, "", "")
    assert "(sin información guardada)" in llm["user"]


def test_el_perfil_guardado_se_usa(app, llm, situacional):
    app.call_situational("ollama", "m", "s", "n", "m", None, "PERFIL-X", "")
    assert "PERFIL-X" in llm["user"]


def test_las_notas_personales_se_recortan(app, llm, situacional):
    """Sin tope, el archivo de notas crece y se come la ventana de contexto."""
    app.call_situational("ollama", "m", "s", "n", "m", None, "", "n" * 9000)
    assert llm["user"].count("n" * 6000) == 1
    assert "n" * 6001 not in llm["user"]


def test_sin_notas_no_hay_bloque_de_notas(app, llm, situacional):
    app.call_situational("ollama", "m", "s", "n", "m", None, "", "   ")
    assert "NOTAS PERSONALES" not in llm["user"]


def test_el_situacional_usa_mas_temperatura_que_el_analisis(app, llm, situacional):
    """Generar líneas pide más variedad que diagnosticar un chat."""
    app.call_situational("ollama", "m", "s", "n", "m", None, "", "")
    assert llm["temperature"] == 0.8


# ===========================================================================
# Las notas personales
# ===========================================================================

@pytest.fixture
def datos(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    return tmp_path


def test_el_archivo_cuelga_de_data_dir(app, datos):
    assert app._notes_path() == datos / "personal_notes.txt"


def test_la_carpeta_se_crea_si_no_esta(app, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", str(tmp_path / "nueva"))
    app._notes_path()
    assert (tmp_path / "nueva").is_dir()


def test_sin_archivo_las_notas_son_vacias(app, datos):
    assert app._load_personal_notes() == ""


def test_una_nota_se_guarda_y_se_lee(app, datos):
    app._save_personal_note("  una frase propia  ")
    assert app._load_personal_notes() == "una frase propia"


def test_la_segunda_nota_no_pisa_la_primera(app, datos):
    """Es el único lugar donde Ilan escribe a mano: perder una nota es perder
    algo que no está en ningún otro archivo."""
    app._save_personal_note("primera")
    app._save_personal_note("segunda")
    assert app._load_personal_notes() == "primera\n\n---\n\nsegunda"


def test_una_nota_vacia_no_escribe_nada(app, datos):
    app._save_personal_note("   \n  ")
    assert not app._notes_path().exists()


def test_una_nota_vacia_no_agrega_un_separador(app, datos):
    app._save_personal_note("primera")
    app._save_personal_note("")
    assert app._load_personal_notes() == "primera"


def test_borrar_vacia_el_archivo_pero_lo_deja(app, datos):
    app._save_personal_note("algo")
    app._delete_personal_notes()
    assert app._notes_path().exists()
    assert app._load_personal_notes() == ""


def test_borrar_sin_archivo_no_rompe(app, datos):
    app._delete_personal_notes()


# ===========================================================================
# Los prompts del disco
# ===========================================================================

def test_los_dos_prompts_del_sistema_existen(app):
    """Se leen en cada corrida: si falta uno, la app muere al arrancar."""
    assert app._load_system_prompt().strip()
    assert app._load_situational_prompt().strip()


def test_el_prompt_situacional_tiene_los_placeholders(app):
    """`call_situational` los reemplaza: si se renombran en el .md, el modelo
    recibe la plantilla cruda."""
    texto = app._load_situational_prompt()
    assert "{user_name}" in texto
    assert "{principles_core}" in texto


def test_el_nucleo_de_principios_no_quedo_vacio(app):
    assert len(app.PRINCIPLES_CORE.strip()) > 100

